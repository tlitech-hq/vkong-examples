"""Generate and export through the real Gradio API, saving documentation artifacts.

Run with a separate client environment containing gradio_client, pillow and trimesh:
    python smoke_client.py http://127.0.0.1:7860 --image input.png
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from gradio_client import Client, handle_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    image = Path(args.image).resolve()
    saved_input = output / ("input" + image.suffix)
    if image != saved_input:
        shutil.copy2(image, saved_input)
    client = Client(args.url, download_files=str(output))
    client.view_api()
    # Use a single Client session: upstream stores generation state server-side.
    client.predict(api_name="/start_session")
    preprocessed = client.predict(handle_file(str(image)), api_name="/preprocess_image")
    start = time.monotonic()
    generated = client.predict(
        handle_file(preprocessed), 42, "512",
        7.5, 0.7, 12, 5.0,
        7.5, 0.5, 12, 3.0,
        1.0, 0.0, 12, 3.0,
        api_name="/image_to_3d",
    )
    generation_seconds = time.monotonic() - start
    # gr.State is maintained by Gradio and omitted from the public API signature.
    start = time.monotonic()
    result = client.predict(100000, 1024, api_name="/extract_glb")
    extraction_seconds = time.monotonic() - start
    files = result if isinstance(result, (list, tuple)) else [result]
    glb = next(Path(f) for f in files if isinstance(f, str) and f.endswith(".glb"))
    destination = output / "sample.glb"
    if glb.resolve() != destination:
        shutil.copy2(glb, destination)
    import trimesh
    scene = trimesh.load(destination, force="scene")
    faces = sum(len(mesh.faces) for mesh in scene.geometry.values())
    vertices = sum(len(mesh.vertices) for mesh in scene.geometry.values())
    materials = []
    for mesh in scene.geometry.values():
        material = getattr(mesh.visual, "material", None)
        if material is not None:
            texture = getattr(material, "baseColorTexture", None)
            materials.append({"type": type(material).__name__, "base_color_texture_size": list(texture.size) if texture is not None else None})
    if not faces or not vertices:
        raise RuntimeError("Generated GLB contains no mesh")
    (output / "result.json").write_text(json.dumps({
        "seed": 42, "resolution": "512", "generation_seconds": generation_seconds,
        "extraction_seconds": extraction_seconds, "vertices": vertices, "faces": faces,
        "glb_bytes": destination.stat().st_size, "materials": materials,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    if isinstance(generated, (list, tuple)):
        for item in generated:
            if isinstance(item, str) and "<" in item:
                (output / "preview.html").write_text(item)
                import base64
                import re
                frames = re.findall(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)", item)
                for index, (kind, payload) in enumerate(frames):
                    suffix = "jpg" if kind == "jpeg" else kind
                    (output / f"preview-{index:02d}.{suffix}").write_bytes(base64.b64decode(payload))
    print((output / "result.json").read_text())


if __name__ == "__main__":
    main()
