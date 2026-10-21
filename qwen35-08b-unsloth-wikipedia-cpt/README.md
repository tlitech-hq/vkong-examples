# Continued pretraining on Wikipedia with Unsloth

Continue pretraining `unsloth/Qwen3.5-0.8B` on English Wikipedia articles, with
the dataset and the checkpoints kept in one VKong volume. This follows Unsloth's
[continued pretraining guide](https://unsloth.ai/docs/basics/continued-pretraining):
LoRA on every projection plus the embeddings and the LM head, with a smaller
learning rate for the embeddings.

The first run copies all 41 Parquet shards (11.6 GB) of
[`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
`20231101.en` from Hugging Face into volume `unsloth-wiki-lab`. It then trains on
40,000 articles from the first four shards for 300 steps, saves a checkpoint to
the same volume every 50 steps, writes the final adapter and exits.

Every later run reads the dataset from the volume instead of Hugging Face. On
machines that load files when opened, the Run starts without downloading the
volume, and only the files the code opens download: the four training shards and
the checkpoint it resumes from, not the other 37 shards or older checkpoints.

## Before you run

Create a secret bundle named `huggingface` with your `HF_TOKEN` in
**Dashboard → Secrets**. The recipe references it with `secrets: [huggingface]`;
authenticated downloads are faster and not rate limited. A read-only token is
enough. Without a token, remove that line; downloads can then stall.

## Run

From the examples repository:

```bash
cd qwen35-08b-unsloth-wikipedia-cpt
vkong run --detach --auto-stop
```

The job continues after the CLI returns. When it finishes, VKong saves the volume
and releases compute. To follow training, use the instance ID printed by the run
command:

```bash
vkong attach <instance-id>
```

To stop early, use `vkong app stop qwen35-08b-unsloth-wikipedia-cpt`. Running the
same command again on any machine resumes from the newest complete checkpoint and
does not download the dataset from Hugging Face again.

## Outputs

Volume `unsloth-wiki-lab` is mounted at `/vkong/storage/unsloth-wiki-lab`:

- `datasets/wikipedia-20231101.en/`: the 41 shards and `dataset.json`, which
  lists every shard with its size. A shard is copied under a temporary name and
  renamed when complete, so an interrupted copy never looks finished.
- `runs/qwen35-08b-wikipedia-cpt/checkpoints/checkpoint-<step>/`: LoRA adapter,
  optimizer, scheduler, RNG and Trainer state. Saved every 50 steps and at the
  final step; the latest two are kept.
- `runs/qwen35-08b-wikipedia-cpt/final-adapter/`: the adapter for inference with
  the base model, plus the tokenizer.
- `runs/qwen35-08b-wikipedia-cpt/training-complete.json`: written last, after
  everything above is saved. It records the settings, the final step, the loss
  and a sample article.

A finished run exits without training again. An unfinished run checks that its
newest checkpoint is complete before resuming, and fails instead of starting over
if it is not.

## Download the result

Check that the volume's last save completed, then download the adapter without a
GPU:

```bash
vkong storage list
vkong storage download unsloth-wiki-lab runs/qwen35-08b-wikipedia-cpt/final-adapter -o ./final-adapter
```

## Use the model

Load the adapter with the same base model and continue an article:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="final-adapter",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
prompt = "Wikipedia Article\n### Title: Ha Long Bay\n\n### Article:\n"
# Qwen3.5 loads a multimodal processor: pass the prompt as text=.
inputs = tokenizer(text=[prompt], return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Results

Measured on one RTX 4090 (24 GB) on 11 October 2026 with the settings in this folder.

First run, starting from an empty volume:

| Step | Time |
|---|---:|
| Copy 41 shards (11.6 GB) from Hugging Face into the volume | 2 min 23 s |
| Read the four training shards | 7 s |
| Train 300 steps | 11 min |
| Whole task, once the workload started | 16 min 30 s |

The loss fell from 2.67 at step 10 to about 2.4 (2.45 averaged over the run), and
GPU memory peaked at 6.9 GB. Each checkpoint is 3.3 GB: the adapter with the
embeddings and LM head is 1.2 GB and the optimizer state 2.1 GB.

Resuming on a new machine, with 18.4 GB in the volume (the dataset and two
checkpoints), the volume was ready in about two seconds and the run downloaded
4.8 GB: the four training shards (1.4 GB) and the checkpoint it resumed from
(3.3 GB). Training continued from step 151.

The sample article follows Wikipedia's format and tone, but some of its facts are
wrong: 300 steps teach a 0.8B model the style of the text, not reliable knowledge.
Train longer, on more shards or with a larger model for better content.

The image is 7.8 GB; on a machine that does not have it yet, the first start can
take another 15 to 20 minutes before the workload begins.

## Adjust training

Edit `train.py`:

- `TRAIN_SHARDS` and `TRAIN_ARTICLES`: how much of the dataset to read. More
  shards download on first use and stay on the rental disk.
- `MAX_STEPS`, `SAVE_STEPS`, batch size and learning rates in
  `UnslothTrainingArguments`.
- `MODEL`: another Unsloth model; raise `gpu` in `vkong.yaml` for larger ones.
- `DATASET_CONFIG`: another language, such as `20231101.vi`, with a new
  `DATA_DIR`.

Keep the same settings when resuming. For a new experiment, choose a fresh `OUTPUT`
directory; the dataset in the volume is reused.

`disk_gb` must hold the whole volume, the base model, caches and the image. Only
one App can write to a volume at a time; to train several models from the same
dataset at once, keep the dataset in its own volume.
