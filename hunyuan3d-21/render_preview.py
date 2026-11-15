"""Render a saved GLB with a real browser (local docs tool, not a GPU dependency).

Install playwright locally and its Chromium browser, or use an installed Chrome.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='artifacts/vkong-hy3d/shape.glb')
    parser.add_argument('--output', default='artifacts/vkong-hy3d/shape-preview.png')
    args = parser.parse_args()
    model = Path(args.model).resolve()
    output = Path(args.output).resolve()
    html = model.parent / 'viewer.html'
    html.write_text('''<!doctype html><html><head><meta charset="utf-8">
<script type="module" src="https://cdn.jsdelivr.net/npm/@google/model-viewer@3.1.1/dist/model-viewer.min.js"></script>
<style>body{margin:0;background:#eef1f5;font-family:Arial,sans-serif;color:#172033}main{margin:28px;padding:28px;background:white;border-radius:16px}h1{margin:0 0 8px;font-size:28px}p{color:#536174}model-viewer{width:100%;height:540px;background:#f5f7fa;border-radius:12px}</style></head><body><main>
<h1>Hunyuan3D 2.1 · RTX 4090</h1><p>Generated mesh · seed 1234 · 30 steps · octree 256</p>
<model-viewer src="''' + model.name + '''" camera-controls environment-image="neutral" camera-orbit="25deg 75deg auto" shadow-intensity="0.7" exposure="1.1"></model-viewer>
</main></body></html>''')
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(model.parent)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
            browser = p.chromium.launch(headless=True, executable_path=str(chrome) if chrome.exists() else None,
                                        args=['--enable-unsafe-swiftshader'])
            page = browser.new_page(viewport={'width': 1100, 'height': 730}, device_scale_factor=1)
            page.goto(f'http://127.0.0.1:{server.server_port}/viewer.html')
            page.wait_for_function("document.querySelector('model-viewer').loaded === true", timeout=120000)
            page.wait_for_timeout(2000)
            page.screenshot(path=str(output))
            browser.close()
            print(output)
    finally:
        server.shutdown()


if __name__ == '__main__':
    main()
