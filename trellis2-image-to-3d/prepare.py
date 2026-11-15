"""Populate the Hugging Face cache during setup, before service readiness starts."""
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download, get_hf_file_metadata, hf_hub_url, get_token


def main():
    print("Prefetching TRELLIS.2; HF_TOKEN configured:", bool(os.environ.get("HF_TOKEN")), flush=True)
    config_path = hf_hub_download("microsoft/TRELLIS.2-4B", "pipeline.json")
    config = json.loads(Path(config_path).read_text())
    # Check gated auxiliary models before spending time downloading large weights.
    access_errors = []
    for field in ["image_cond_model", "rembg_model"]:
        repo = config["args"][field]["args"]["model_name"]
        try:
            get_hf_file_metadata(hf_hub_url(repo, "config.json"), token=get_token())
            print("Access OK:", repo, flush=True)
        except Exception as exc:
            print("Access check failed:", repo, type(exc).__name__, flush=True)
            access_errors.append(repo)
    if access_errors:
        raise RuntimeError("Grant the HF_TOKEN account access to: " + ", ".join(access_errors))
    patterns = ["pipeline.json"]
    external = []
    for checkpoint in config["args"]["models"].values():
        if checkpoint.startswith("ckpts/"):
            patterns.extend([checkpoint + ".json", checkpoint + ".safetensors"])
        else:
            parts = checkpoint.split("/")
            external.append(("/".join(parts[:2]), "/".join(parts[2:])))
    snapshot_download("microsoft/TRELLIS.2-4B", allow_patterns=patterns, max_workers=4)
    for repo, checkpoint in external:
        for extension in [".json", ".safetensors"]:
            hf_hub_download(repo, checkpoint + extension)
    for field in ["image_cond_model", "rembg_model"]:
        repo = config["args"][field]["args"]["model_name"]
        print("Prefetching", repo, flush=True)
        snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.py"], max_workers=4)
    print("Model cache prepared", flush=True)


if __name__ == "__main__":
    main()
