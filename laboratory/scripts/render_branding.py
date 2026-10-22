#!/usr/bin/env python3
"""Render branding SVGs to high-res PNGs via headless Chromium (Playwright)."""
from pathlib import Path
from playwright.sync_api import sync_playwright

BRAND = Path("/home/z/my-project/repo-rmt/assets/branding")
JOBS = [
    # (svg, out_png, viewport_w, viewport_h, device_scale_factor)
    ("aurora-banner.svg", "aurora-banner.png",     1280, 320, 2),
    ("aurora-avatar.svg", "aurora-avatar-1024.png", 512, 512, 2),
    ("aurora-icon.svg",   "aurora-icon-512.png",    256, 256, 2),
    ("aurora-social.svg", "social-preview.png",    1280, 640, 1),
]

HTML_TMPL = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>html,body{{margin:0;padding:0;overflow:hidden;background:#04040c}}
svg{{display:block}}</style></head>
<body>{svg}</body></html>"""


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--force-color-profile=srgb"])
        for svg_name, png_name, w, h, dsf in JOBS:
            svg_path = BRAND / svg_name
            out_path = BRAND / png_name
            page = browser.new_page(
                viewport={"width": w, "height": h},
                device_scale_factor=dsf,
            )
            page.set_content(HTML_TMPL.format(svg=svg_path.read_text(encoding="utf-8")))
            page.wait_for_timeout(600)  # let fonts / first animation frame settle
            page.screenshot(path=str(out_path), clip={"x": 0, "y": 0, "width": w, "height": h})
            page.close()
            kb = out_path.stat().st_size / 1024
            print(f"OK {out_path.name}: {kb:.0f} KB ({w*dsf}x{h*dsf})")
        browser.close()


if __name__ == "__main__":
    main()
