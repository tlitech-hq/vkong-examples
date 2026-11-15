"""Drive the real Gradio UI and capture a reproducible documentation walkthrough.

Local dependency: playwright and Chrome (or playwright's installed Chromium).
"""
import argparse
import json
from pathlib import Path
import time
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('--input', default='artifacts/vkong-hy3d/input.png')
    parser.add_argument('--output', default='artifacts/vkong-hy3d')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    screenshots = output / 'ui'
    screenshots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        browser = p.chromium.launch(headless=True, executable_path=str(chrome) if chrome.exists() else None,
                                    args=['--enable-unsafe-swiftshader'])
        page = browser.new_page(viewport={'width': 1600, 'height': 1200}, color_scheme='light', accept_downloads=True)
        page.set_default_timeout(120000)
        page.goto(args.url, wait_until='domcontentloaded')
        page.get_by_role('button', name='Gen Textured Shape', exact=True).wait_for()
        page.wait_for_timeout(1500)

        def capture(name):
            # Crop out the unrelated upstream sample gallery; preserve the actual UI.
            gallery = page.get_by_role('tab', name='Image to 3D Gallery', exact=True).bounding_box()
            width = int(gallery['x'] - 12) if gallery else 1600
            page.screenshot(path=str(screenshots / name), clip={'x': 0, 'y': 0, 'width': width, 'height': 1160})
            print('Captured:', name, flush=True)

        capture('01-open-ui.png')
        page.locator('input[type=file]').first.set_input_files(str(Path(args.input).resolve()))
        page.wait_for_function("() => { const img = document.querySelector('#component-6 img'); return img && img.complete && img.naturalWidth > 0; }", timeout=300000)
        page.wait_for_timeout(2000)
        capture('02-upload-vkong.png')
        page.get_by_role('tab', name='Advanced Options', exact=True).click()
        page.get_by_label('Randomize seed', exact=True).uncheck()
        capture('03-generation-settings.png')
        start = time.monotonic()
        page.get_by_role('button', name='Gen Textured Shape', exact=True).click()
        page.wait_for_timeout(3000)
        capture('04-generating.png')
        iframe = page.locator('iframe[src*="textured_mesh.html"]')
        last_progress = start
        while iframe.count() == 0:
            if time.monotonic() - start > 3600:
                capture('generation-timeout.png')
                raise TimeoutError('No textured result after one hour; inspect runtime logs')
            if time.monotonic() - last_progress > 30:
                print('Waiting for textured result:', round(time.monotonic() - start), 'seconds', flush=True)
                last_progress = time.monotonic()
            page.wait_for_timeout(1000)
        generation_seconds = time.monotonic() - start
        frame = iframe.first.content_frame
        frame.locator('model-viewer').wait_for()
        frame.locator('model-viewer').evaluate("async el => { if (!el.loaded) await new Promise(resolve => el.addEventListener('load', resolve, {once:true})); }")
        frame.locator('model-viewer').evaluate("el => { el.autoRotate = false; el.cameraOrbit = '15deg 80deg 3.7m'; el.jumpCameraToGoal(); }")
        page.wait_for_timeout(2000)
        capture('05-generated-mascot.png')
        page.get_by_role('tab', name='Export', exact=True).click()
        page.get_by_role('button', name='Transform', exact=True).click()
        download = page.get_by_text('Download', exact=True)
        download.wait_for()
        export_frame = page.locator('iframe[src*="textured_mesh.html"]').last.content_frame
        export_frame.locator('model-viewer').wait_for()
        export_frame.locator('model-viewer').evaluate("async el => { if (!el.loaded) await new Promise(resolve => el.addEventListener('load', resolve, {once:true})); el.autoRotate = false; el.cameraOrbit = '15deg 80deg 3.7m'; el.jumpCameraToGoal(); }")
        page.wait_for_timeout(1500)
        capture('06-export-glb.png')
        with page.expect_download(timeout=180000) as pending:
            download.click()
        pending.value.save_as(str(output / 'textured.glb'))
        page.get_by_role('tab', name='Mesh Statistic', exact=True).click()
        capture('07-mesh-statistics.png')
        (output / 'ui-result.json').write_text(json.dumps({
            'input': 'input.png', 'seed': 1234, 'steps': 30, 'octree_resolution': 256,
            'generation_seconds_including_model_load': generation_seconds,
            'screenshots': [f.name for f in sorted(screenshots.glob('*.png'))],
            'result_iframe': iframe.first.get_attribute('src'),
        }, indent=2) + '\n')
        print('Saved textured.glb and UI walkthrough', flush=True)
        browser.close()


if __name__ == '__main__':
    main()
