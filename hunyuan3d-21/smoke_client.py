"""Run real shape + texture generation and save its outputs for documentation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from gradio_client import Client, handle_file
import trimesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('--image', default='artifacts/vkong-hy3d/input.png')
    parser.add_argument('--output', default='artifacts/vkong-hy3d')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = Client(args.url, download_files=str(output), httpx_kwargs={"timeout": 1800.0})
    parameters = dict(steps=30, guidance_scale=5.0, seed=1234,
                      octree_resolution=256, check_box_rembg=True,
                      num_chunks=8000, randomize_seed=False)
    start = time.monotonic()
    result = client.predict(image=handle_file(str(Path(args.image).resolve())),
                            **parameters, api_name='/generation_all')
    elapsed = time.monotonic() - start
    shape, textured, html, stats, seed = result
    for src, name in [(shape, 'shape.glb'), (textured, 'textured.glb')]:
        shutil.copy2(src, output / name)
    (output / 'preview.html').write_text(html)
    scene = trimesh.load(output / 'textured.glb', force='scene')
    faces = sum(len(mesh.faces) for mesh in scene.geometry.values())
    vertices = sum(len(mesh.vertices) for mesh in scene.geometry.values())
    materials = []
    for mesh in scene.geometry.values():
        material = getattr(mesh.visual, 'material', None)
        texture = getattr(material, 'baseColorTexture', None)
        materials.append({'type': type(material).__name__,
                          'base_color_texture_size': list(texture.size) if texture else None})
    if not faces or not vertices:
        raise RuntimeError('Generated GLB has no geometry')
    if not any(m['base_color_texture_size'] for m in materials):
        raise RuntimeError('Generated GLB has no base color texture')
    (output / 'result.json').write_text(json.dumps({
        'parameters': parameters, 'wall_seconds_including_first_load': elapsed,
        'stats': stats, 'seed': seed, 'faces': faces, 'vertices': vertices,
        'materials': materials,
        'glb_bytes': (output / 'textured.glb').stat().st_size,
        'sha256': hashlib.sha256((output / 'textured.glb').read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    print((output / 'result.json').read_text())


if __name__ == '__main__':
    main()
