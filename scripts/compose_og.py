#!/usr/bin/env python3
"""Composite the deterministic OG image (1200x630) for social previews.

Base art comes from ``gen_images.render_og_base`` (same palette, generous
negative space on the left third); the text lockup is the only place in
the visual layer where typography is allowed, per assets/prompts.yaml.

Output: ``assets/visuals/og-image.png``.

Usage::

    python scripts/compose_og.py
"""

from __future__ import annotations

import os
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from PIL import Image, ImageDraw, ImageFont

from gen_images import OUT_DIR, PAL, _rgb, render_og_base


OUT_PATH = os.path.join(OUT_DIR, "og-image.png")
W, H = 1200, 630
FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def main() -> int:
    base = render_og_base().resize((W, H), Image.LANCZOS)
    dr = ImageDraw.Draw(base)
    bold = ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"), 74)
    med = ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans.ttf"), 34)
    small = ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSansMono.ttf"), 26)
    x0 = 72
    y0 = 218
    dr.text((x0, y0), "RMT \u00d7 LLM Research", font=bold, fill=_rgb(PAL["text"]))
    dr.rectangle((x0, y0 + 108, x0 + 322, y0 + 111), fill=_rgb(PAL["spike"]))
    dr.text(
        (x0, y0 + 138),
        "Spectra \u00b7 Hallucinations \u00b7 RLHF",
        font=med,
        fill=_rgb(PAL["bulk"]),
    )
    dr.text((x0, H - 86), "wild8highlander/rmt-llm-research", font=small, fill=(154, 165, 186))
    os.makedirs(OUT_DIR, exist_ok=True)
    base.save(OUT_PATH, format="PNG", optimize=True)
    print(f"og-image.png written: {os.path.getsize(OUT_PATH) / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
