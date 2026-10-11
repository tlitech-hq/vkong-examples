import json
from pathlib import Path
import tarfile

import unsloth  # Patch the training stack before importing TRL or Transformers.
import torch
from datasets import load_dataset
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


MODEL = "unsloth/SmolLM2-135M-Instruct"
DATASET = "yahma/alpaca-cleaned"
DATASET_SPLIT = "train"
NUM_TRAIN_EPOCHS = 1
OUTPUT = Path("/vkong/storage/unsloth-smollm2-lab/smollm2-unsloth-full")
CHECKPOINTS = OUTPUT / "checkpoints"
FINAL_ADAPTER = OUTPUT / "final-adapter"
MARKER = OUTPUT / "training-complete.json"
DOWNLOAD = OUTPUT / "download" / "smollm2-unsloth-checkpoint.tar.gz"
MAX_SEQUENCE_LENGTH = 512
SAVE_STEPS = 500


def verify_checkpoint(checkpoint: Path, require_scaler: bool = False) -> None:
    # Adapter weights alone cannot resume training. This single-GPU recipe also
    # requires the optimizer, scheduler, RNG and Trainer state from Trainer.save.
    required = [
        "adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
        "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin",
        "tokenizer_config.json",
    ]
    if require_scaler:
        required.append("scaler.pt")
    missing = [name for name in required if not (checkpoint / name).is_file()
               or (checkpoint / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"incomplete checkpoint {checkpoint}: {', '.join(missing)}")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if checkpoint.name != f"checkpoint-{state['global_step']}":
        raise RuntimeError(f"checkpoint step mismatch: {checkpoint}")


class SaveFinalCheckpoint(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Save all Trainer state at the final step even between save intervals.
        if state.global_step >= state.max_steps:
            control.should_save = True
        return control


def prepare_download(final_checkpoint: Path) -> None:
    DOWNLOAD.parent.mkdir(parents=True, exist_ok=True)
    temporary = DOWNLOAD.with_name(f".{DOWNLOAD.name}.tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(final_checkpoint, arcname=final_checkpoint.name)
        archive.add(FINAL_ADAPTER, arcname=FINAL_ADAPTER.name)
        archive.add(MARKER, arcname=MARKER.name)
    temporary.replace(DOWNLOAD)
    print(f"checkpoint archive ready: {DOWNLOAD}", flush=True)


if MARKER.is_file():
    metadata = json.loads(MARKER.read_text())
    expected = {"model": MODEL, "dataset": DATASET, "split": DATASET_SPLIT,
                "epochs": NUM_TRAIN_EPOCHS, "max_sequence_length": MAX_SEQUENCE_LENGTH}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("completed run uses different settings; choose a new OUTPUT directory")
    final_checkpoint = CHECKPOINTS / f"checkpoint-{metadata['steps']}"
    verify_checkpoint(final_checkpoint, require_scaler=metadata.get("amp_scaler", False))
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        path = FINAL_ADAPTER / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"incomplete final adapter: {path}")
    prepare_download(final_checkpoint)
    print("training already complete; saved checkpoint verified", flush=True)
    raise SystemExit(0)

OUTPUT.mkdir(parents=True, exist_ok=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    dtype=None,
    load_in_4bit=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

dataset = load_dataset(DATASET, split=DATASET_SPLIT)


def format_examples(batch):
    texts = []
    for instruction, context, response in zip(batch["instruction"], batch["input"], batch["output"]):
        prompt = instruction if not context else f"{instruction}\n\nContext:\n{context}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts}


dataset = dataset.map(format_examples, batched=True, remove_columns=dataset.column_names)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    callbacks=[SaveFinalCheckpoint()],
    args=SFTConfig(
        dataset_text_field="text",
        max_length=MAX_SEQUENCE_LENGTH,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_ratio=0.03,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_only_model=False,
        save_total_limit=2,
        output_dir=str(CHECKPOINTS),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        # Let the installed Unsloth version choose its compatible tokenizer
        # workers. Forcing one worker can hit a patched-config pickle failure.
        dataset_num_proc=None,
        packing=False,
    ),
)
require_scaler = trainer.accelerator.scaler is not None
resume_from = get_last_checkpoint(str(CHECKPOINTS)) if CHECKPOINTS.is_dir() else None
if resume_from:
    verify_checkpoint(Path(resume_from), require_scaler=require_scaler)
    print(f"resuming training from {resume_from}", flush=True)
result = trainer.train(resume_from_checkpoint=resume_from)
# Resuming from the final checkpoint can run one extra step; use the actual last step.
final_step = int(trainer.state.global_step)
final_checkpoint = CHECKPOINTS / f"checkpoint-{final_step}"
if final_step < trainer.state.max_steps:
    raise RuntimeError("training ended before all configured epochs completed")
verify_checkpoint(final_checkpoint, require_scaler=require_scaler)
model.save_pretrained(FINAL_ADAPTER)
tokenizer.save_pretrained(FINAL_ADAPTER)
temporary_marker = MARKER.with_suffix(".json.tmp")
temporary_marker.write_text(
    json.dumps(
        {
            "model": MODEL,
            "dataset": DATASET,
            "split": DATASET_SPLIT,
            "epochs": NUM_TRAIN_EPOCHS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "examples": len(dataset),
            "steps": final_step,
            "checkpoint": str(final_checkpoint),
            "train_loss": float(result.training_loss),
            "amp_scaler": require_scaler,
            "cuda": torch.cuda.get_device_name(0),
        },
        indent=2,
    )
    + "\n"
)
temporary_marker.replace(MARKER)
print(
    f"training complete: checkpoint={final_checkpoint} adapter={FINAL_ADAPTER}",
    flush=True,
)
prepare_download(final_checkpoint)
