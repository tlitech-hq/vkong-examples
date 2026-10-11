"""Train Qwen3.5-4B as an Unsloth decision model on typed-decisions.

Follows the official Unsloth decision notebook, trains for two full epochs,
saves resumable checkpoints to the attached volume, calibrates on held-out rows,
scores the dataset's test split and packages the calibrated adapter.
"""
import json
import os
import tarfile
from pathlib import Path

# Faster Hugging Face downloads; set before the hub libraries load.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import unsloth  # noqa: F401  Patch the training stack before Transformers.
import torch
from datasets import load_dataset
from transformers import TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported


MODEL = "unsloth/Qwen3.5-4B"
DATASET = "LocalLLaMA/typed-decisions"
DATASET_CONFIG = "all"
NUM_TRAIN_EPOCHS = 2
MAX_SEQUENCE_LENGTH = 2048
SAVE_STEPS = 20
SEED = 3407
OUTPUT = Path("/vkong/storage/unsloth-decision-lab/qwen35-4b-decisions")
CHECKPOINTS = OUTPUT / "checkpoints"
FINAL_ADAPTER = OUTPUT / "final-adapter"
METRICS = OUTPUT / "metrics.json"
MARKER = OUTPUT / "training-complete.json"
DOWNLOAD = OUTPUT / "download" / "qwen35-4b-decisions.tar.gz"
SETTINGS = {"model": MODEL, "dataset": DATASET, "config": DATASET_CONFIG,
            "epochs": NUM_TRAIN_EPOCHS, "max_sequence_length": MAX_SEQUENCE_LENGTH}

# The notebook's example: one support ticket, three kinds of decision.
EXAMPLE_INPUT = "Hi, I was charged twice for invoice #4411. Please refund the duplicate today."
EXAMPLE_QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, errors",
            "sales": "pricing, new plans",
        },
    },
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "soon", "today"],
    },
}


def plain(value):
    # Metrics can hold numpy or torch numbers and arrays.
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_json(path: Path, value) -> None:
    # Write to a temporary name first so a stop never leaves half a file.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=plain) + "\n")
    temporary.replace(path)


def verify_checkpoint(checkpoint: Path) -> None:
    # A resumable checkpoint needs the Trainer state, not only the adapter.
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"]
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


def prepare_download() -> None:
    DOWNLOAD.parent.mkdir(parents=True, exist_ok=True)
    temporary = DOWNLOAD.with_name(f".{DOWNLOAD.name}.tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(FINAL_ADAPTER, arcname=FINAL_ADAPTER.name)
        archive.add(METRICS, arcname=METRICS.name)
        archive.add(MARKER, arcname=MARKER.name)
    temporary.replace(DOWNLOAD)
    print(f"adapter archive ready: {DOWNLOAD}", flush=True)


# A finished run only rebuilds the archive.
if MARKER.is_file():
    metadata = json.loads(MARKER.read_text())
    if any(metadata.get(key) != value for key, value in SETTINGS.items()):
        raise RuntimeError("completed run used different settings; choose a new OUTPUT directory")
    if not FINAL_ADAPTER.is_dir() or not any(FINAL_ADAPTER.iterdir()) or not METRICS.is_file():
        raise RuntimeError(f"incomplete final adapter or metrics in {OUTPUT}")
    prepare_download()
    print("training already complete; saved adapter verified", flush=True)
    raise SystemExit(0)

OUTPUT.mkdir(parents=True, exist_ok=True)

model, tokenizer = FastDecisionModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQUENCE_LENGTH,  # Long inputs keep the question and options.
    load_in_4bit=True,
)
model = FastDecisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

# Each row has a state (the input), its questions and the gold answers.
dataset = load_dataset(DATASET, DATASET_CONFIG, split="train")
items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
print(f"skipped {report['skipped']} of {report['total']} decisions", flush=True)
train_items, eval_items = FastDecisionModel.split_holdout(items, seed=SEED)
print(f"{len(train_items)} training rows, {len(eval_items)} held out", flush=True)

metrics = json.loads(METRICS.read_text()) if METRICS.is_file() else {"settings": SETTINGS}
resume_from = get_last_checkpoint(str(CHECKPOINTS)) if CHECKPOINTS.is_dir() else None
if resume_from:
    verify_checkpoint(Path(resume_from))
    print(f"resuming training from {resume_from}", flush=True)
else:
    # The decision head is new, so this is close to guessing.
    metrics["holdout_before_training"] = FastDecisionModel.evaluate(model, tokenizer, eval_items)
    print(f"holdout before training: {metrics['holdout_before_training']}", flush=True)
    write_json(METRICS, metrics)

trainer = DecisionTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_items,
    eval_dataset=eval_items,
    callbacks=[SaveFinalCheckpoint()],
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        eval_strategy="epoch",
        logging_steps=5,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        output_dir=str(CHECKPOINTS),
        report_to="none",
        seed=SEED,
    ),
)
result = trainer.train(resume_from_checkpoint=resume_from)
if int(result.global_step) != trainer.state.max_steps:
    raise RuntimeError("training ended before all configured epochs completed")
verify_checkpoint(CHECKPOINTS / f"checkpoint-{int(result.global_step)}")

# Calibration fits the probabilities, so a 90% answer is right about 90% of the time.
metrics["holdout_calibrated"] = FastDecisionModel.calibrate(model, tokenizer, eval_items)
print(f"holdout after calibration: {metrics['holdout_calibrated']}", flush=True)
test = load_dataset(DATASET, DATASET_CONFIG, split="test")
test_items, _ = FastDecisionModel.build_dataset(test, tokenizer, model)
metrics["test"] = FastDecisionModel.evaluate(model, tokenizer, test_items)
print(f"test split: {metrics['test']}", flush=True)

FastDecisionModel.for_inference(model)
answers = FastDecisionModel.predict(model, tokenizer, EXAMPLE_INPUT, EXAMPLE_QUESTIONS)
metrics["example"] = {
    name: {"answer": result_["answer"], "probabilities": result_["probabilities"]}
    for name, result_ in answers.items()
}
for name, result_ in answers.items():
    print(name, result_["answer"], {k: round(v, 3) for k, v in result_["probabilities"].items()}, flush=True)

# Saves the LoRA adapters, the decision head and its calibration.
model.save_pretrained(str(FINAL_ADAPTER))
tokenizer.save_pretrained(str(FINAL_ADAPTER))
metrics["training"] = {
    "steps": int(result.global_step),
    "train_loss": float(result.training_loss),
    "train_runtime_seconds": float(result.metrics.get("train_runtime", 0.0)),
    "train_rows": len(train_items),
    "holdout_rows": len(eval_items),
    "gpu": torch.cuda.get_device_name(0),
    "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
}
write_json(METRICS, metrics)
# The marker is written last: its presence means everything above is saved.
write_json(MARKER, {**SETTINGS, "steps": int(result.global_step), "adapter": str(FINAL_ADAPTER)})
print(f"training complete: adapter={FINAL_ADAPTER}", flush=True)
prepare_download()
