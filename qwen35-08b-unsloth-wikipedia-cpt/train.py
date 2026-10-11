"""Continued pretraining on English Wikipedia with the dataset kept in a volume.

The first run copies all 41 Parquet shards (11.6 GB) of wikimedia/wikipedia
20231101.en into the volume. Every later run reads them from the volume instead
of the Hub; with load-on-open machines only the shards a run opens download.
Checkpoints go to the same volume, so a stopped run resumes on any machine.
Training follows Unsloth's continued pretraining notebook.
"""
import os

# Faster Hugging Face downloads; set before the hub libraries load.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

import unsloth  # noqa: F401  Patch the training stack before Transformers.
import torch
from datasets import load_dataset
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported


VOLUME = Path("/vkong/storage/unsloth-wiki-lab")
DATASET_REPO = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"
DATA_DIR = VOLUME / "datasets" / f"wikipedia-{DATASET_CONFIG}"
DATA_MANIFEST = DATA_DIR / "dataset.json"
TRAIN_SHARDS = 4          # Parquet shards read for training; the rest stay untouched
DOWNLOAD_WORKERS = 4      # shards downloaded at the same time on the first run
STAGING = Path.home() / ".cache" / "wikipedia-staging"  # rental disk; never saved
TRAIN_ARTICLES = 40_000
MODEL = "unsloth/Qwen3.5-0.8B"
MAX_SEQUENCE_LENGTH = 2048
MAX_STEPS = 300
SAVE_STEPS = 50
SEED = 3407
OUTPUT = VOLUME / "runs" / "qwen35-08b-wikipedia-cpt"
CHECKPOINTS = OUTPUT / "checkpoints"
FINAL_ADAPTER = OUTPUT / "final-adapter"
MARKER = OUTPUT / "training-complete.json"
SETTINGS = {"model": MODEL, "dataset": f"{DATASET_REPO}:{DATASET_CONFIG}", "train_shards": TRAIN_SHARDS,
            "train_articles": TRAIN_ARTICLES, "max_steps": MAX_STEPS, "max_sequence_length": MAX_SEQUENCE_LENGTH}


def log(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, value) -> None:
    # Write to a temporary name first so a stop never leaves half a file.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare_dataset() -> list[Path]:
    """Copy the dataset into the volume once; later runs only check sizes."""
    if DATA_MANIFEST.is_file():
        manifest = json.loads(DATA_MANIFEST.read_text())
        shards = [DATA_DIR / name for name in manifest["files"]]
        # stat() reads the listing only; it does not download a file.
        if all(path.is_file() and path.stat().st_size == manifest["files"][path.name] for path in shards):
            log(f"dataset already in the volume: {len(shards)} shards, {manifest['bytes'] / 1e9:.1f} GB")
            return shards
    api = HfApi()
    remote = sorted(
        (item for item in api.list_repo_tree(DATASET_REPO, path_in_repo=DATASET_CONFIG, repo_type="dataset")
         if item.path.endswith(".parquet")),
        key=lambda item: item.path,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(item) -> int:
        target = DATA_DIR / Path(item.path).name
        if target.is_file() and target.stat().st_size == item.size:
            return 0
        # Download to the rental disk with the Hub's parallel transfer, then copy
        # into the volume under a name ending in .incomplete, which is never
        # saved, and rename it when the copy is complete.
        local = Path(hf_hub_download(DATASET_REPO, item.path, repo_type="dataset", local_dir=STAGING))
        partial = target.with_name(target.name + ".incomplete")
        shutil.copyfile(local, partial)
        if partial.stat().st_size != item.size:
            raise RuntimeError(f"incomplete copy: {item.path}")
        partial.replace(target)
        local.unlink()
        log(f"copied {target.name} ({item.size / 1e6:.0f} MB)")
        return item.size

    started = time.time()
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        copied = sum(pool.map(fetch, remote))
    files = {Path(item.path).name: item.size for item in remote}
    write_json(DATA_MANIFEST, {"repo": DATASET_REPO, "config": DATASET_CONFIG, "files": files,
                               "bytes": sum(files.values())})
    elapsed = time.time() - started
    log(f"dataset in the volume: {len(files)} shards, {sum(files.values()) / 1e9:.1f} GB; "
        f"copied {copied / 1e9:.1f} GB in {elapsed:.0f} s")
    return [DATA_DIR / name for name in files]


def verify_checkpoint(checkpoint: Path) -> None:
    # A resumable checkpoint needs the Trainer state, not only the adapter.
    required = ["adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"]
    missing = [name for name in required if not (checkpoint / name).is_file()
               or (checkpoint / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"incomplete checkpoint {checkpoint}: {', '.join(missing)}")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if checkpoint.name != f"checkpoint-{state['global_step']}":
        raise RuntimeError(f"checkpoint step mismatch: {checkpoint}")


class SaveFinalCheckpoint(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Save the final step even between save intervals.
        if state.global_step >= state.max_steps:
            control.should_save = True
        return control


if MARKER.is_file():
    metadata = json.loads(MARKER.read_text())
    if any(metadata.get(key) != value for key, value in SETTINGS.items()):
        raise RuntimeError("completed run used different settings; choose a new OUTPUT directory")
    log("training already complete; nothing to do")
    raise SystemExit(0)

shards = prepare_dataset()
OUTPUT.mkdir(parents=True, exist_ok=True)
prompt = "Wikipedia Article\n### Title: {}\n\n### Article:\n{}"

resume_from = get_last_checkpoint(str(CHECKPOINTS)) if CHECKPOINTS.is_dir() else None
resume_step = 0
if resume_from:
    verify_checkpoint(Path(resume_from))
    resume_step = json.loads((Path(resume_from) / "trainer_state.json").read_text())["global_step"]

if resume_step >= MAX_STEPS:
    # Training finished before the last run stopped. Resuming the Trainer here
    # would run one extra step, so load the final checkpoint instead.
    log(f"training already reached step {resume_step}; loading {resume_from}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resume_from,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        load_in_4bit=True,
    )
    final_step, train_loss = resume_step, None
else:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head"],  # Add embeddings for continued pretraining
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=True,
    )

    # Opening these shards is what downloads them on a load-on-open machine.
    read_started = time.time()
    dataset = load_dataset("parquet", data_files=[str(path) for path in shards[:TRAIN_SHARDS]], split="train")
    dataset = dataset.shuffle(seed=SEED).select(range(TRAIN_ARTICLES))
    log(f"read {TRAIN_SHARDS} of {len(shards)} shards from the volume in {time.time() - read_started:.0f} s; "
        f"{len(dataset)} articles selected")

    def format_articles(batch):
        return {"text": [prompt.format(title, text) + tokenizer.eos_token
                         for title, text in zip(batch["title"], batch["text"])]}

    dataset = dataset.map(format_articles, batched=True, remove_columns=dataset.column_names)

    trainer = UnslothTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        callbacks=[SaveFinalCheckpoint()],
        args=UnslothTrainingArguments(
            dataset_text_field="text",
            max_length=MAX_SEQUENCE_LENGTH,
            packing=False,
            dataset_num_proc=None,  # Let Unsloth choose compatible tokenizer workers
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            max_steps=MAX_STEPS,
            warmup_steps=10,
            learning_rate=5e-5,
            embedding_learning_rate=1e-5,  # 2 to 10x smaller for the embeddings
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=2,
            output_dir=str(CHECKPOINTS),
            seed=SEED,
            report_to="none",
        ),
    )
    if resume_from:
        log(f"resuming training from {resume_from}")
    result = trainer.train(resume_from_checkpoint=resume_from)
    final_step, train_loss = int(trainer.state.global_step), float(result.training_loss)
    if final_step < MAX_STEPS:
        raise RuntimeError("training ended before max_steps")
    verify_checkpoint(CHECKPOINTS / f"checkpoint-{final_step}")

model.save_pretrained(str(FINAL_ADAPTER))
tokenizer.save_pretrained(str(FINAL_ADAPTER))

FastLanguageModel.for_inference(model)
# Qwen3.5 loads a multimodal processor: pass text by keyword, or it is read as images.
inputs = tokenizer(text=[prompt.format("Ha Long Bay", "")], return_tensors="pt").to("cuda")
sample = tokenizer.decode(model.generate(**inputs, max_new_tokens=64)[0], skip_special_tokens=True)
log(f"sample:\n{sample}")

# The marker is written last: its presence means everything above is saved.
write_json(MARKER, {**SETTINGS, "steps": final_step, "train_loss": train_loss,
                    "resumed_from": resume_from, "gpu": torch.cuda.get_device_name(0),
                    "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
                    "sample": sample})
log(f"training complete: adapter={FINAL_ADAPTER}")
