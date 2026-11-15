# TRELLIS.2 image-to-3D on VKong

Upload an image, generate a 3D asset in the upstream web UI, and download a GLB.
This is a **draft recipe**: CUDA dependencies built successfully on A100 40 GB,
but inference and browser preview remain unverified because the Hugging Face
token did not yet have DINOv3 access during the GPU run.

Prerequisites: the VKong CLI, `vkong login`, and permission to download the
upstream models. Both recipes use workspace secret `huggingface` containing
`HF_TOKEN`; create it in your own terminal if it does not exist. Never commit the token.
The default is one A100 40 GB, 64 GB host RAM, and 150 GB disk.
The $2/hour `max_price` is a cap, not a quoted rental price.

Before renting, request access with the account that owns `HF_TOKEN` to
[DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
and [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0).
A valid Hugging Face token does not automatically grant access to gated models.
Setup checks these permissions before downloading the main weights. Main TRELLIS.2
weights total approximately 14.82 GB, plus auxiliary models. Downloads happen during
setup so they do not consume the service readiness timeout.

## Run

```bash
cd trellis2-image-to-3d
vkong validate
vkong app list
vkong run --detach
```

Review the actual hourly price before accepting. Setup installs dependencies
and builds CUDA extensions; the first start downloads model weights. Both can
take several minutes, and the machine bills during setup and downloads.
Open the local URL printed by VKong. Reconnect with `vkong attach <machine>`
if the local connection closes.

## Try

1. Upload `artifacts/input.png`, the VKong mascot holding a Hy3D sign.
2. Start with the 512 generation mode and a fixed seed. Generate the asset.
3. Extract the GLB, inspect the 3D preview, and download it.

For a smoke check, use the same example image and seed twice. Confirm each
output downloads, opens as a nonempty mesh in a GLB viewer, and includes
materials/textures. A fixed seed helps comparison but does not promise identical
binary files across machines. Loading the web page alone is not a generation test.

Model downloads and generated files live on the machine disk. There is no Storage
volume: download results before stopping or replacing the machine. Clear old
outputs through the upstream UI where available, or restart on the same machine
when necessary; restarting does not guarantee old output files are removed.

## Stop

```bash
vkong app stop trellis2-image-to-3d
```

The machine keeps billing until stopped. Closing the browser or detaching the CLI
does not stop it.

## Publish and update

```bash
vkong deploy
```

This publishes the UI at the App URL. The UI does not add authentication;
only publish when you intend other users with access to that URL to submit work.
The queue is bounded to four requests with a default concurrency of one.
Update a published service with `vkong deploy`. For an unpublished App, use
`vkong attach <machine> --update`. If `setup.sh` changes, also change `init_cmd`
or explicitly rerun setup with `vkong attach <machine> --setup` before restarting.
Keep the machine after a failed build and fix it there.

## Runtime details

Upstream requires at least 24 GB VRAM and has tested A100/H100. This recipe
uses A100 40 GB to leave room for preview and extraction. Higher resolutions
and complex meshes can still exhaust memory. CUDA extensions compile during setup.

TRELLIS.2 uses the upstream MIT license; its downloaded dependencies and models
retain their own licenses.

Source is checked out at [`75fbf0183001`](https://github.com/microsoft/TRELLIS.2/tree/75fbf0183001ed9876c8dbb35de6b68552ee08bd).
The recipe uses upstream UI/model loading rather than vendoring the project.
`launch.py` binds it to `0.0.0.0:$PORT` and limits the queue. `setup.sh` needs
root in the supplied CUDA development image. Requirements pin direct Python
dependencies; transitive dependencies and automatically downloaded model revisions
are not a complete reproducibility lock.

If setup fails, inspect `vkong attach <machine>`. Common areas to check are CUDA
compiler/driver compatibility, available disk, package resolution, and model-download
access. If only high-resolution generation fails, try the initial settings again
before changing GPU or budget.
