"""Capture real textured-GLB views and a contact sheet with a browser."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright

VIEWS = [('front', 'Front', '0deg 90deg 3.5m'),
         ('front-left', 'Front left', '-35deg 75deg 3.8m'),
         ('front-right', 'Front right', '35deg 75deg 3.8m'),
         ('left', 'Left', '-90deg 80deg 3.8m'),
         ('right', 'Right', '90deg 80deg 3.8m'),
         ('back', 'Back', '180deg 80deg 3.8m')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='artifacts/vkong-hy3d/textured.glb')
    args = parser.parse_args()
    model = Path(args.model).resolve()
    cards = ''.join(f'<figure id="{name}"><figcaption>{title}</figcaption><model-viewer src="{model.name}" camera-orbit="{orbit}" camera-target="0m 0m 0m" field-of-view="35deg" environment-image="neutral" shadow-intensity="0.6" exposure="1"></model-viewer></figure>' for name, title, orbit in VIEWS)
    (model.parent / 'views.html').write_text('''<!doctype html><html><head><meta charset="utf-8"><script type="module" src="https://cdn.jsdelivr.net/npm/@google/model-viewer@3.1.1/dist/model-viewer.min.js"></script><style>body{margin:0;padding:28px;background:#edf1f6;font-family:Arial,sans-serif;color:#182235}h1{margin:0 0 8px}p{margin:0 0 24px;color:#526173}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}figure{margin:0;padding:16px;background:white;border-radius:12px}figcaption{font-size:22px;font-weight:600;margin-bottom:8px}model-viewer{width:100%;height:530px;background:#e8edf3;border-radius:8px}</style></head><body><h1>VKong mascot · Hunyuan3D 2.1 · RTX 4090</h1><p>Actual textured GLB · seed 1234 · 30 shape steps · texture 512 / six views</p><div class="grid">'''+cards+'</div></body></html>')
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(model.parent)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
            browser = p.chromium.launch(headless=True, executable_path=str(chrome) if chrome.exists() else None,
                                        args=['--enable-unsafe-swiftshader'])
            page = browser.new_page(viewport={'width': 1800, 'height': 1360}, device_scale_factor=1)
            page.goto(f'http://127.0.0.1:{server.server_port}/views.html', wait_until='domcontentloaded')
            page.wait_for_function("Array.from(document.querySelectorAll('model-viewer')).every(m => m.loaded === true)", timeout=180000)
            page.wait_for_timeout(3000)
            for name, _, _ in VIEWS:
                page.locator('#'+name).screenshot(path=str(model.parent / f'textured-{name}.png'))
            page.screenshot(path=str(model.parent / 'textured-views.png'), full_page=True)
            browser.close()
            print('Saved six textured views and contact sheet', flush=True)
    finally:
        server.shutdown()


if __name__ == '__main__':
    main()
