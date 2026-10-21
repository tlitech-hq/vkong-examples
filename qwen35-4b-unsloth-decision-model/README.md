# Train a decision model with Unsloth

Fine-tune `unsloth/Qwen3.5-4B` into a **decision model**: instead of writing text,
it reads an input, looks at the options you give it and picks one with a
calibrated probability. This follows Unsloth's
[decision model guide](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)
and its official notebook, run as one VKong task.

The task trains for two full epochs on
[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
calibrates the probabilities on held-out rows, scores the dataset's `test` split,
saves the calibrated adapter to a volume, packages it for download and exits.

Each row has a `state` (the input), the `questions` to decide about it and the
`gold` answers. There are three kinds of question:

- `choice`: pick one option, such as which team should handle a ticket.
- `noul`: yes or no.
- `score`: pick a level, such as how urgent something is.

## Before you run

Create a secret bundle named `huggingface` with your `HF_TOKEN` in
**Dashboard → Secrets**. The recipe references it with `secrets: [huggingface]`;
authenticated downloads are faster and not rate limited. A read-only token is
enough. Without a token, remove that line; downloads can then stall.

## Run

From the examples repository:

```bash
cd qwen35-4b-unsloth-decision-model
vkong run --detach --auto-stop
```

The job continues after the CLI returns. When it finishes, VKong saves the volume
and releases compute. To follow training, use the instance ID printed by the run
command:

```bash
vkong attach <instance-id>
```

Closing an attached session leaves the job running. To stop early, use
`vkong app stop qwen35-4b-unsloth-decision-model`; running the same command again
resumes from the last saved checkpoint.

## Outputs

Everything is written to `/vkong/storage/unsloth-decision-lab/qwen35-4b-decisions`,
the mount of volume `unsloth-decision-lab`:

- `checkpoints/checkpoint-<step>/`: LoRA adapters, decision head, optimizer,
  scheduler, RNG and Trainer state. Saved every 20 steps and at the final step;
  the latest two are kept.
- `final-adapter/`: the calibrated adapter and decision head for inference with
  the original base model, plus the tokenizer.
- `metrics.json`: held-out accuracy before training, after calibration, the `test`
  split result, the example predictions and training details.
- `training-complete.json`: written last, after everything above is saved.
- `download/qwen35-4b-decisions.tar.gz`: final adapter, metrics and summary.

A finished run verifies its files, rebuilds the archive and exits without
training again. An unfinished run resumes from the newest complete checkpoint and
fails instead of starting over if that checkpoint is incomplete.

## Download the result

Check that the volume's last save completed, then download the archive without a
GPU:

```bash
vkong storage list
vkong storage download unsloth-decision-lab qwen35-4b-decisions/download/qwen35-4b-decisions.tar.gz
```

## Use the model

Load the saved adapter with the same base model and make decisions:

```python
from unsloth import FastDecisionModel

model, tokenizer = FastDecisionModel.from_pretrained(
    model_name="final-adapter",  # the folder from the archive
    max_seq_length=2048,
    load_in_4bit=True,
)
FastDecisionModel.for_inference(model)
answers = FastDecisionModel.predict(
    model,
    tokenizer,
    "Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
    {
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
    },
)
for name, result in answers.items():
    print(name, result["answer"], result["probabilities"])
```

`answer` is the option for `choice`, `True` or `False` for `noul`, and the level
number for `score`; `probabilities` covers every option.

## Results

Measured on one RTX 4090 (24 GB) on 11 October 2026 with the settings in this folder:

| | Accuracy | Calibration error (ECE) |
|---|---:|---:|
| Held-out rows before training | 30.8% | 0.057 |
| Held-out rows after training and calibration | 81.5% | 0.048 |
| Dataset `test` split | 77.7% | 0.036 |

These match Unsloth's notebook, which reports about 78% on the `test` split after a
full run. Training took 70 steps in about 10 minutes and peaked at 7.5 GB of GPU
memory. The example ticket was sent to `billing` (1.00), recognised as a refund
request (0.98) and rated urgent today (0.99).

The task ran about 16 minutes once the workload started: about 4 to download the
model and prepare data, 10 to train and 2 to calibrate, score the test split and
save. The image is 7.8 GB; on a machine that does not have it yet, the first start
can take another 15 to 20 minutes before the workload begins.

## Adjust training

Edit `train.py`:

- `MODEL`: Unsloth also documents `unsloth/Llama-3.2-3B-Instruct` and
  `unsloth/gemma-4-E4B-it`.
- `DATASET` / `DATASET_CONFIG`: your own data works if each row has the same
  `state`, `questions` and `gold` fields.
- `NUM_TRAIN_EPOCHS`, `SAVE_STEPS`, batch size and learning rate in
  `TrainingArguments`.

Keep the same settings when resuming. For a new experiment, choose a fresh
`OUTPUT` directory and update the download path.

The base model and dataset download into the Hugging Face cache on the rental
disk; only the volume is saved. The image `unsloth/unsloth:core-nightly-2026.10.10`
includes `FastDecisionModel`, which Unsloth added in October 2026.
