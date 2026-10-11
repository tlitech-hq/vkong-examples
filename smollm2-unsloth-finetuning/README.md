# Fine-tune SmolLM2 with Unsloth

Train a LoRA adapter for `unsloth/SmolLM2-135M-Instruct` on the full `train`
split of `yahma/alpaca-cleaned`. The default runs **one complete epoch**, saves
resumable checkpoints and the final adapter, packages the result, then exits.
There is no download server or fixed 60-step test limit.

## Run

From the examples repository:

```bash
cd smollm2-unsloth-finetuning
vkong run --detach --auto-stop
```

The job continues after the CLI returns. On completion, VKong saves the attached
volume and releases compute. Saving takes time and remains billable until the
machine is released; storage billing continues separately.

To follow training, use the instance ID printed by the run command:

```bash
vkong attach <instance-id>
```

Closing an established attach session leaves the detached job running. To stop
early, use `vkong app stop smollm2-unsloth-finetuning`. Resume recovers from the
last complete saved checkpoint, not from unsaved GPU memory.

## Checkpoints and final adapter

All outputs live in `/vkong/storage/unsloth-smollm2-lab/smollm2-unsloth-full`,
the default mount path of volume `unsloth-smollm2-lab`:

- `checkpoints/checkpoint-<step>/`: LoRA weights, tokenizer, optimizer, scheduler,
  RNG and Trainer state. Saves every 500 optimizer steps and at the final step;
  retains the latest two checkpoints. Trainer also saves the AMP scaler when used.
- `final-adapter/`: adapter and tokenizer for inference with the original base model.
- `training-complete.json`: summary written after final checkpoint validation.
- `download/smollm2-unsloth-checkpoint.tar.gz`: final checkpoint, adapter and summary.

The checkpoint is for resuming this LoRA run. It does not contain a merged copy
of the base model. The final adapter alone does not contain optimizer state.

Running again with the same settings resumes an unfinished run. If the latest
checkpoint is incomplete, the script fails instead of silently starting over;
recover an earlier complete checkpoint before retrying. A completed run verifies
its files, rebuilds the archive and exits without training again.

## Download after saving

First check that the volume's latest save has completed successfully:

```bash
vkong storage show unsloth-smollm2-lab
```

Then download the archive without keeping a GPU online:

```bash
vkong storage download unsloth-smollm2-lab smollm2-unsloth-full/download/smollm2-unsloth-checkpoint.tar.gz
```

A stop acknowledgment is not confirmation that saving finished. Downloads read
the latest completed save, so check its time and status before downloading.

## Adjust training

Edit `train.py`: `NUM_TRAIN_EPOCHS` controls full passes through the selected data,
`DATASET` / `DATASET_SPLIT` select the data, and `SAVE_STEPS` controls checkpoint
frequency. The default uses the whole training split; sequences are limited to
512 tokens. Longer examples are subject to the trainer's sequence-length handling.

Keep the same model, data and training settings when resuming. For a new experiment,
choose a fresh `OUTPUT` directory, and update the download path accordingly. The new
directory is separate from the old `smollm2-unsloth-60-step` test output in the volume.

Hugging Face downloads the model and dataset into its normal cache on the rental's
disk. Only the volume is saved; VKong writes back changes under the mount path
automatically. The recipe requests 82 GB disk; VKong checks available storage
headroom before renting.

This full-dataset task has been checked locally; its new training duration,
GPU memory use and end-to-end resume still require a GPU run. The old 60-step
service test is not evidence for this updated recipe.
