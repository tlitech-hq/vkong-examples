# DiffusionGemma decisions with Jev and vLLM

Run Google's `google/diffusiongemma-26B-A4B-it` behind a Jev API. vLLM serves
the model on a private port and `jev_server.py` exposes `/v1/systemone`.

DiffusionGemma is a text diffusion model. This recipe uses the 26B total / 4B
active MoE checkpoint in BF16 on one H100 with at least 80 GB VRAM.

## Files

- `vkong.yaml` requests the machine and defines the public service.
- `serve.sh` starts vLLM and waits until it is healthy.
- `jev_server.py` is the reduced Jev reference flow: it resolves answer slots,
  reads exact label logprobs from vLLM, normalizes them, and averages samples.
- `client.py` sends and validates one example decision.

vLLM downloads the pinned Hugging Face model automatically on the first start.
`HF_HOME` selects its cache; no separate checkpoint downloader is needed.
The recipe caps the context at 8,192 tokens to keep startup and KV-cache usage
reasonable for this decision API.

## Run privately

Install and log in to the VKong CLI, then run:

```bash
cd diffusiongemma-jev-vllm
vkong run
```

The first run downloads about 50 GB of weights and warms up vLLM. VKong prints a
local URL when the service is ready. Keep that command running and use the URL in
another terminal:

```bash
export VKONG_URL=http://127.0.0.1:<local-port>
curl -fsS "$VKONG_URL/health"
python3 client.py
```

## Deploy with a public URL

Start the GPU service in the background, then publish it:

```bash
vkong run --detach
vkong deploy
```

`vkong deploy` prints the HTTPS URL. Call the same client against it:

```bash
export VKONG_URL=https://<your-app-url>
python3 client.py
```

If the App uses an `API_KEY` secret:

```bash
export VKONG_URL=https://<your-app-url>
read -rs API_KEY && export API_KEY
python3 client.py
```

After editing `serve.sh`, `jev_server.py`, or `vkong.yaml`, update the deployed
App on its current GPU with:

```bash
vkong deploy
```

Stop the GPU when finished:

```bash
vkong app stop diffusiongemma-jev-vllm
```

## Model cache

The framework downloads the model to the rental's local disk. That cache is
removed when the machine is released, and a new machine downloads it again.
Do not add `storage` to this service: workspace volumes are for training tasks
that write checkpoints to `/vkong/storage/<volume>`.

## What the client sends

The example asks whether a production incident is urgent and which team owns it:

```json
{
  "state": {
    "ticket": "Production checkout is unavailable and a launch starts in 30 minutes.",
    "customer_impact": "All purchases fail."
  },
  "questions": {
    "urgent": {
      "type": "noul",
      "instructions": "Does this need a response within the next hour?"
    },
    "owner": {
      "type": "choice",
      "instructions": "Which team should handle this first?",
      "criteria": {
        "payments": "Checkout and payment failures",
        "growth": "Campaign and acquisition issues",
        "data": "Analytics pipeline issues"
      }
    }
  }
}
```

The gateway reads the logprobs of `yes`/`no` and `A`/`B`/`C` directly from the
model's answer slots, then returns a Jev-shaped response:

```json
{
  "model": "dgemma",
  "answers": {
    "urgent": {"type": "noul", "noul": 0.99},
    "owner": {"type": "choice", "choice": "payments"}
  }
}
```

## What vLLM and the Jev gateway do

There is no additional training or fine-tuning.

1. vLLM loads the original DiffusionGemma weights on the GPU.
2. The gateway assigns one-token labels and resolves their exact canvas slots.
3. It seeds a diffusion canvas and calls vLLM with `diffusion_read_only`.
4. vLLM returns the exact logprob for every allowed label at every answer slot.
5. The gateway applies softmax, averages repeated reads, and returns Jev's shape.

The gateway is an adapter. It does not convert or retrain the model. Two server
processes are used because vLLM owns model inference while the gateway owns the
Jev request and response contract. Only the gateway port is public.

## Where probabilities come from

This example keeps the probability path from vLLM's Jev reference:

1. Give each answer a short label, such as `yes`/`no` or `A`/`B`/`C`.
2. Reserve a known decision position in the generated canvas.
3. Read the model's log probability for every allowed label at that position.
4. Convert log probabilities to positive scores with `exp(logprob)`.
5. Normalize the allowed scores so they sum to one.
6. Repeat with multiple noise draws and average the distributions.

For allowed labels `A`, `B`, and `C`, the normalized value is:

```text
P(A | allowed labels, context)
  = exp(logprob(A))
    / (exp(logprob(A)) + exp(logprob(B)) + exp(logprob(C)))
```

This is a conditional probability among the supplied labels. It is not proof
that the available choices are complete. A useful implementation also records
how much probability mass the model assigned to those labels. If the labels are
missing from returned top-k logprobs, the probability cannot be reconstructed
reliably; increase `max_logprobs`, use single-token labels, or reject the read.

## Diffusion models versus ordinary autoregressive LLMs

An ordinary LLM writes from left to right. Given two questions, it first writes:

```text
urgent: yes
```

It then writes `owner: A` after `yes` is already in its context. The second
probability therefore means:

```text
P(owner=A | incident, urgent=yes)
```

Changing the order of the questions can change the result. Ask each question in
a separate request when they should be independent.

DiffusionGemma starts with several unfinished answer slots:

```text
urgent: [unknown]
owner:  [unknown]
```

It refines the whole block through denoising steps. Jev reads the `yes/no`
distribution from the first slot and the `A/B/C` distribution from the second.
The slots can still affect each other, but they are not forced through a strictly
left-to-right answer order.

For one `yes/no` question, both model families work similarly: an ordinary LLM
provides next-token logprobs, while DiffusionGemma provides answer-slot logprobs.
Neither requires retraining, and neither is automatically calibrated.

## Are five answers in one response still valid probabilities?

They can be valid conditional probabilities, but the request structure determines
what each number means:

- Five separate requests estimate each answer from the same original context.
  This is easiest to interpret when the questions should be independent.
- Five answers generated left to right are order-dependent. Question 5 is
  conditioned on the tokens and answers generated for questions 1 through 4.
- Five slots read from one diffusion canvas form a joint structured prediction.
  The answers may influence each other even when they are read in parallel.
- Normalizing only the five listed options reports probability conditional on
  those options. It does not measure the probability that none of them is right.

For production decisions, define the intended conditioning explicitly. Use
separate reads for independent questions, or declare dependencies when later
questions should depend on earlier answers. Keep labels short and tokenization
stable, preserve the raw logprobs, and measure calibration on held-out examples.

Useful calibration checks include Brier score for binary decisions, log loss for
multiple choices, reliability diagrams, expected calibration error, accuracy,
and abstention behavior. A probability is operationally trustworthy only after
these measurements match the target workload.

## Scope of this example

The included gateway preserves vLLM's slot resolution, exact label-logprob read,
softmax, and repeated-sample averaging. It supports `noul`, `choice`, up to ten
questions, one to eight denoising steps, and one to eight samples. The full vLLM
server additionally supports score questions, dependencies, adaptive sampling,
thinking, chunking, images, richer diagnostics, TLS, and OpenAI passthrough.

The recipe pins:

- model revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`;
- vLLM commit `e9757321527ca1ecd514c07c1418dd2c53da3d19`;
- Linux/amd64 image manifest
  `sha256:e0eee5c5506bea9bfe350f7d99b07dc49e37d42647a128c2a57ff184551fba10`.

## Sources

- [Google model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
- [vLLM DiffusionGemma recipe](https://github.com/vllm-project/recipes/blob/main/models/Google/diffusiongemma-26B-A4B-it.yaml)
- [vLLM Jev reference server](https://github.com/vllm-project/vllm/blob/e9757321527ca1ecd514c07c1418dd2c53da3d19/examples/features/structured_diffusion/structured_server.py)
