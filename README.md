<p align="center">
  <a href="https://vkong.tli-tech.com">
    <img src="https://vkong.tli-tech.com/logo-text.png" alt="VKong" height="72">
  </a>
</p>

<h1 align="center">VKong Recipes</h1>

<h3 align="center">Runnable recipes for training, generation, and GPU inference with VKong.</h3>

<p align="center">
  <a href="#-get-started">Get started</a> •
  <a href="#-choose-a-recipe">Recipes</a> •
  <a href="https://vkong.tli-tech.com/docs">Documentation</a> •
  <a href="https://vkong.tli-tech.com/download">Download VKong</a>
</p>

Bring your code. VKong finds matching compute, prepares the machine, syncs the
project, and runs the workload through one CLI workflow. Every folder in this
repository is a complete project with its own `vkong.yaml`.

## ⚡ Get started

Install the [`vkong` CLI](https://vkong.tli-tech.com/download), then run the
smallest example:

```bash
git clone https://github.com/tlitech-hq/vkong-examples.git
cd vkong-examples/hello-world

vkong login
vkong run
```

VKong prints the local URL when the service is ready. Open it in a browser or
test it from another terminal:

```bash
curl http://127.0.0.1:<local-port>/
```

Use `vkong run --detach` when the workload should continue after the terminal
closes. Stop the App when you are finished to release compute and stop billing:

```bash
vkong app stop hello-world
```

## 🧭 Choose a recipe

Recipes are grouped by workload so this catalog can grow across models and
frameworks without becoming one long mixed list.

### Start here

| Recipe | Framework | Compute | What you will run |
|---|---|---:|---|
| [Hello World](hello-world) | FastAPI | CPU | Minimal service and the shortest end-to-end VKong workflow |

### LLM inference

| Recipe | Model | Compute | Interface |
|---|---|---:|---|
| [Qwen3.5 9B with llama.cpp](qwen35-9b-llama-cpp) | Qwen3.5-9B GGUF | 1× RTX 4090 | OpenAI-compatible API |
| [Qwen3.5 4B with SGLang](qwen35-4b-sglang) | Qwen3.5-4B | 1× RTX 4090 | OpenAI-compatible API |
| [DiffusionGemma with Jev and vLLM](diffusiongemma-jev-vllm) | DiffusionGemma 26B-A4B | 1× H100 80 GB | Jev structured decisions + OpenAI-compatible API |
| [MiniMax-M2.7 with vLLM (multi-GPU)](minimax-m27-vllm-multi-gpu) | MiniMax-M2.7 | 4× H100 | OpenAI-compatible API, tensor and expert parallelism |

### Model serving

| Recipe | Framework | Model | Compute |
|---|---|---|---:|
| [MobileNetV2 with NVIDIA Triton](mobilenetv2-triton) | Triton + ONNX Runtime | MobileNetV2 | 1× RTX 4090 |

### Training

| Recipe | Framework | Workload | Compute |
|---|---|---|---:|
| [Fine-tune SmolLM2 with Unsloth](smollm2-unsloth-finetuning) | Unsloth | Full-epoch LoRA training with resumable checkpoints | 1× GPU |
| [Train a decision model with Unsloth](qwen35-4b-unsloth-decision-model) | Unsloth | Qwen3.5-4B picks options with calibrated probabilities; 77.7% on the test split | 1× RTX 4090 |
| [Continued pretraining on Wikipedia with Unsloth](qwen35-08b-unsloth-wikipedia-cpt) | Unsloth | Qwen3.5-0.8B trains from an 11.6 GB dataset kept in a volume with its checkpoints; resumes on any machine | 1× RTX 4090 |
| [GPU training smoke test](pytorch-training-smoke-test) | PyTorch | Verify training, checkpoints, and task lifecycle with a small CNN | 1× GPU |

### Image generation

| Recipe | Framework | Model | Compute |
|---|---|---|---:|
| [SDXL image generation](sdxl-image-generation) | Diffusers | SSD-1B text-to-image task | 1× RTX 4090 |
| [Qwen-Image 2.1 with Viggle Turbo](qwen-image-21-viggle-turbo-sglang) | SGLang Diffusion | 6-step text-to-image and image-editing API | 1× RTX 4090 |

Start with **Hello World** to verify login, provisioning, sync, and connectivity.
Use **llama.cpp** or **SGLang** for a practical single-GPU inference service.
The **vLLM** recipe requests four H100 GPUs and can be expensive; review its
`max_dph` and the displayed hourly price before starting it.

## How a recipe works

Each recipe keeps the workload contract in `vkong.yaml`:

```text
my-app/
├── vkong.yaml          # workload, compute requirements, image, and commands
├── README.md           # run, test, reconnect, and stop instructions
└── application files
```

The important fields are:

- `type: task|service` — whether the workload exits or keeps listening.
- `app` — the stable App name shown in the dashboard.
- `gpu`, `num_gpus`, `cpu_cores`, `ram_gb`, `disk_gb` — compute requirements.
- `max_dph` — the maximum hourly machine price you accept.
- `image` — the container image for the workload.
- `init_cmd` and `start` — setup and execution commands.

Run a recipe from its directory:

```bash
vkong run
```

For a long-running task or service, detach after it starts:

```bash
vkong run --detach
```

For a finite task that should release compute when it exits:

```bash
vkong run --detach --auto-stop
```

See the [Getting Started guide](https://vkong.tli-tech.com/docs/getting-started)
for the complete first-run flow and the
[CLI reference](https://vkong.tli-tech.com/docs/cli/login) for command details.

## Bring your own project

From an existing project directory, generate a config and run it:

```bash
cd my-project
vkong new-config
vkong run
```

Use these recipes as starting points: copy the closest `vkong.yaml`, then change
the App name, image, compute requirements, setup command, and start command for
your workload.

## Contributing a recipe

A useful recipe should:

- run from a fresh clone without committed credentials;
- pin a compatible image or dependency version;
- state the expected GPU count and VRAM;
- include one deterministic smoke test;
- explain how to stop the App and billing;
- keep secret values out of `vkong.yaml`, source files, and logs.

Open an issue or pull request with the workload, expected hardware, and the
command used to verify it.

## Recipe style

Start each README with the result, then show **Run → Try → Stop**. Keep the first
run short. Put optional deployment and tuning instructions afterwards.

- Use an image with the framework already installed.
- Group YAML fields under App, Compute, Budget, Storage (training tasks only), and Runtime.
- Do not add `storage` or `cache` to services or other non-training recipes.
- Keep YAML focused on compute, image, and a short setup/start command.
- Let the framework download its model; set its cache location instead of writing a downloader.
- Use `requirements.txt` for dependency lists and a small Python file for model conversion or preparation.
- Keep necessary server options in a readable `serve.sh`; avoid shell installation fallbacks.
- Explain whether output/cache is temporary or saved in the Storage volume (`/vkong/storage/<volume>`).
- Keep credentials in workspace secrets, and retain the example's price and resource limits.

See the [Hugging Face cache settings](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)
and [llama.cpp server options](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
