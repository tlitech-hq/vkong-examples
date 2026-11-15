"""Run the pinned upstream UI with one GPU stage resident at a time."""
import gc
import os
from pathlib import Path
import sys


def main():
    import gradio as gr

    import torch
    import time

    active = {"stage": None, "pipeline": None}

    class Stage:
        """Keep only one stage's weights resident; reload from local cache on switch."""
        def __init__(self, name, factory):
            self.name = name
            self.factory = factory

        def __call__(self, *args, **kwargs):
            if active["stage"] != self.name:
                active["pipeline"] = None
                active["stage"] = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                print("Loading stage:", self.name, flush=True)
                active["pipeline"] = self.factory()
                active["stage"] = self.name
            start = time.monotonic()
            try:
                return active["pipeline"](*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                print(f"Stage {self.name}: {time.monotonic() - start:.2f}s; peak allocated VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

    original_mount = gr.mount_gradio_app

    def mount(app, blocks, *args, **kwargs):
        @app.get("/health")
        def health():
            return {"status": "ready", "active_stage": active["stage"]}

        # All generation callbacks share the same GPU concurrency group.
        if not blocks.fns:
            raise RuntimeError("The upstream UI contains no generation handlers")
        for function in blocks.fns.values():
            function.concurrency_id = "hunyuan-gpu"
            function.concurrency_limit = 1
        blocks.queue(max_size=4, default_concurrency_limit=1)
        return original_mount(app, blocks, *args, **kwargs)

    gr.mount_gradio_app = mount
    upstream = Path(__file__).resolve().parent / "upstream"
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    # The check executes in upstream globals immediately before UI construction.
    source = (upstream / "gradio_app.py").read_text()
    anchor = "    demo = build_app()"
    if source.count(anchor) != 1:
        raise RuntimeError("Pinned upstream launch layout changed")
    source = source.replace(anchor, '    if not HAS_TEXTUREGEN:\n        raise RuntimeError("Texture initialization failed; see the traceback above")\n' + anchor)
    replacements = {
        "Hunyuan3DPaintConfig(max_num_view=8, resolution=768)": "Hunyuan3DPaintConfig(max_num_view=6, resolution=512)",
        "tex_pipeline = Hunyuan3DPaintPipeline(conf)": 'tex_pipeline = Stage("texture", lambda: Hunyuan3DPaintPipeline(conf))',
        "i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(": 'i23d_worker = Stage("shape", lambda: Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(',
        "    )\n    if args.enable_flashvdm:": "    ))\n    if args.enable_flashvdm:",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError("Pinned upstream stage initialization changed: " + old)
        source = source.replace(old, new)
    sys.argv = ["gradio_app.py", "--host", "0.0.0.0", "--port", os.environ.get("PORT", "7860")]
    exec(compile(source, str(upstream / "gradio_app.py"), "exec"), {"__name__": "__main__", "__file__": str(upstream / "gradio_app.py"), "Stage": Stage})


if __name__ == "__main__":
    main()
