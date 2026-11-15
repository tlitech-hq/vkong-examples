"""Run the pinned upstream UI with VKong's service binding and a bounded queue."""
import os
from pathlib import Path
import runpy
import sys


def main():
    import gradio as gr

    original_launch = gr.Blocks.launch

    def launch(blocks, *args, **kwargs):
        blocks.queue(max_size=4, default_concurrency_limit=1)
        kwargs.update(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), share=False)
        return original_launch(blocks, *args, **kwargs)

    gr.Blocks.launch = launch
    upstream = Path(__file__).resolve().parent / "upstream"
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    print("Starting pinned TRELLIS.2 UI and model initialization", flush=True)
    runpy.run_path(str(upstream / "app.py"), run_name="__main__")


if __name__ == "__main__":
    main()
