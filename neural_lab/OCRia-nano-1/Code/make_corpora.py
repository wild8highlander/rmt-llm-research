"""Generate the synthetic hallucination corpora for OCRia-nano-1.

Every crop is rendered programmatically, so ground truth is exact and free.
Six families are covered by seven batches:

  clean      120 latin text crops + GT            (abilities / anchor data)
  degraded    60 degraded text crops + GT         (F3 misreads)
  structured 120 non-text structured patterns     (F1 phantom, F2 loops)
  noise       60 pure-noise crops                 (F1 phantom)
  blank       30 blank / gradient crops           (F1 phantom, specificity)
  control     40 held-out text crops + GT         (KL / parity protocol)
  router     160 crops 32x512: 4 scripts x20 + 40 non-text + flipped text
                                                  (F4 routing, F5 flip)

All manifests are JSONL: {id, file, text|null, family, kind}.  The same
generator seeds the parity check, the probe, the training sets and the one
evaluator, so no number in the kit can drift between stages.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocria_common import DATA_DIR, KIT_DIR, dump_json, seed_all

FONT_DIR = KIT_DIR.parent / "fonts"
FONTS = {
    "sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "sans_bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "liberation": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
}
ARABIC_FONT = str(FONT_DIR / "NotoNaskhArabic-Regular.ttf")
CJK_FONT = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"

LINES = [
    "the quick brown fox jumps", "invoice total 1,240.00 USD",
    "page 7 of 12", "receipt 40231 approved", "rmt x llm research",
    "spectral purity gate pass", "dear colleague, please review",
    "exit 0 status ok", "chapter 3: random matrix theory",
    "temperature 21.4 C humidity 48%", "flight UA 328 gate B12",
    "invoice 2026-0417 due date", "tel +1 555 0134", "hello world 42",
    "batch 77 of 120 samples", "total 3 items shipped",
    "ocr probe line number nine", "gradient descent at step 150",
    "epsilon 1e-4 and beta 10", "anchor scope: good arm only",
]

WORD_BANK = ["document", "scanner", "signal", "matrix", "noise", "edge",
             "bulk", "spike", "anchor", "decoder", "latent", "vector",
             "model", "probe", "spectral", "purity", "loop", "phantom",
             "routing", "confidence", "layout", "table", "cell", "stamp"]


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def render_text(text: str, font_path: str, size: int, width: int,
                y: int = 4, x: int = 4, bg: int = 255) -> tuple[Image.Image, str]:
    """Render at a size that FITS the crop; shrink until the full string is
    visible so GT always matches what the recognizer can actually read."""
    while size >= 12:
        f = _font(font_path, size)
        if f.getlength(text) <= width - 2 * x:
            img = Image.new("L", (width, 32), bg)
            d = ImageDraw.Draw(img)
            d.text((x, y), text, font=f, fill=0)
            return img, text
        size -= 1
    # last resort: trim words off the tail so the visible part IS the GT
    f = _font(font_path, max(12, size))
    words = text.split()
    while words and f.getlength(" ".join(words)) > width - 2 * x:
        words = words[:-1]
    text = " ".join(words)
    img = Image.new("L", (width, 32), bg)
    d = ImageDraw.Draw(img)
    d.text((x, y), text, font=f, fill=0)
    return img, text


# --------------------------------------------------------------------------
# non-text pattern generators (the hallucination-inducing "documents")
# --------------------------------------------------------------------------

def pat_scribble(rng, w, h=32) -> Image.Image:
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(7, 14))):
        x0 = int(rng.integers(0, w - 40))
        y0 = int(rng.integers(4, h - 6))
        pts = [(x0, y0)]
        for k in range(int(rng.integers(3, 7))):
            x0 = min(w - 2, x0 + int(rng.integers(6, 26)))
            y0 = int(np.clip(y0 + rng.integers(-14, 15), 2, h - 3))
            pts.append((x0, y0))
        d.line(pts, fill=0, width=int(rng.integers(1, 3)))
    return img


def pat_wave(rng, w, h=32) -> Image.Image:
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for off in (0, 10, 20):
        amp = rng.uniform(3, 7)
        ph = rng.uniform(0, 6.3)
        pts = [(x, int(off + amp * np.sin(x / 14.0 + ph))) for x in range(0, w, 3)]
        d.line(pts, fill=60, width=2)
    return img


def pat_halftone(rng, w, h=32) -> Image.Image:
    arr = np.full((h, w), 255, np.float32)
    step = int(rng.integers(4, 7))
    ph = int(rng.integers(0, step))
    for yy in range(0, h, step):
        for xx in range((yy // step + ph) % 2, w, step):
            r = rng.uniform(0.8, 2.2)
            y0, y1 = int(max(0, yy - r)), int(min(h, yy + r + 1))
            x0, x1 = int(max(0, xx - r)), int(min(w, xx + r + 1))
            gy, gx = np.mgrid[y0:y1, x0:x1]
            mask = (gy - yy) ** 2 + (gx - xx) ** 2 <= r * r
            arr[y0:y1, x0:x1][mask] = 40
    return Image.fromarray(arr.astype(np.uint8))


def pat_lineart(rng, w, h=32) -> Image.Image:
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(4, 8))):
        x0, x1 = sorted(rng.integers(0, w, 2))
        y0 = int(rng.integers(2, h - 10))
        y1 = int(rng.integers(2, h - 10))
        d.line([(int(x0), y0), (int(x1), y1)], fill=0, width=1)
        if rng.random() < 0.5:
            d.ellipse([x0 - 3, y0 - 3, x0 + 3, y0 + 3], outline=0, width=1)
    return img


def pat_grid(rng, w, h=32) -> Image.Image:
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    xs = sorted(rng.integers(10, w - 10, int(rng.integers(3, 6))))
    for x in xs:
        d.line([(int(x), 1), (int(x), h - 1)], fill=80, width=1)
    for y in (1, h // 2, h - 2):
        d.line([(0, int(y)), (w - 1, int(y))], fill=80, width=1)
    return img


def pat_seal(rng, w, h=32) -> Image.Image:
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    cx, cy = int(rng.integers(w // 3, 2 * w // 3)), h // 2
    r = int(rng.integers(10, 14))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=2)
    d.ellipse([cx - r + 4, cy - r + 4, cx + r - 4, cy + r - 4], outline=0, width=1)
    for _ in range(int(rng.integers(8, 16))):
        ang = rng.uniform(0, 6.28)
        d.line([(cx + (r - 2) * np.cos(ang), cy + (r - 2) * np.sin(ang)),
                (cx + (r + 3) * np.cos(ang), cy + (r + 3) * np.sin(ang))],
               fill=30, width=1)
    return img


def pat_band(rng, w, h=32) -> Image.Image:
    arr = np.full((h, w), 255, np.float32)
    slope = rng.uniform(-0.2, 0.2)
    thick = rng.uniform(4, 9)
    xg = np.arange(w)[None, :]
    yg = np.arange(h)[:, None]
    band = np.abs(yg - (h / 2 + slope * (xg - w / 2))) < thick
    arr[band] = 90
    noise = rng.normal(0, 8, (h, w))
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


PATTERNS = {"scribble": pat_scribble, "wave": pat_wave, "halftone": pat_halftone,
            "lineart": pat_lineart, "grid": pat_grid, "seal": pat_seal,
            "band": pat_band}


# --------------------------------------------------------------------------
# degradation operators for F3 (misreads)
# --------------------------------------------------------------------------

def degrade(img: Image.Image, rng) -> Image.Image:
    ops = rng.integers(0, 5)
    a = np.asarray(img, np.float32)
    if ops == 0:  # gaussian blur
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 2.4)))
    elif ops == 1:  # additive noise
        a = a + rng.normal(0, rng.uniform(28, 55), a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    elif ops == 2:  # salt & pepper
        m = rng.random(a.shape) < rng.uniform(0.02, 0.06)
        a[m] = rng.random(m.sum()) * 255
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    elif ops == 3:  # downscale-upscale (low-res scan)
        small = img.resize((max(40, img.width // 3), 8), Image.BILINEAR)
        img = small.resize((img.width, 32), Image.BILINEAR)
    else:  # low contrast + noise
        a = 128 + (a - 128) * rng.uniform(0.25, 0.5)
        a = a + rng.normal(0, 18, a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return img


# --------------------------------------------------------------------------
# corpus builders
# --------------------------------------------------------------------------

def _crop_from(img: Image.Image, idx: int) -> Image.Image:
    return img


def build(seed: int = 7) -> None:
    seed_all(seed)
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    def emit(batch: str, img: Image.Image, text, family, kind, meta=None):
        i = len(manifest.get(batch, []))
        f = DATA_DIR / batch / f"{batch}_{i:04d}.png"
        f.parent.mkdir(parents=True, exist_ok=True)
        img.convert("L").save(f)
        rec = {"id": f"{batch}_{i:04d}", "file": str(f.relative_to(KIT_DIR)),
               "text": text, "family": family, "kind": kind}
        if meta:
            rec.update(meta)
        manifest.setdefault(batch, []).append(rec)

    # -- clean (abilities / anchor) -----------------------------------------
    for i in range(120):
        text = LINES[i % len(LINES)]
        if i >= len(LINES) * 3:
            words = list(rng.choice(WORD_BANK, 4, replace=False))
            text = " ".join(words) + f" {int(rng.integers(10, 99))}"
        fp = list(FONTS.values())[i % len(FONTS)]
        size = int(rng.integers(15, 24))
        width = int(rng.integers(180, 480))
        img, fitted = render_text(text, fp, size, width,
                                  y=int(rng.integers(3, 9)), x=int(rng.integers(2, 10)))
        emit("clean", img, fitted, None, "text", {"font": Path(fp).stem, "size": size})

    # -- degraded (F3 misreads) ---------------------------------------------
    for i in range(60):
        text = LINES[(i * 3) % len(LINES)]
        fp = list(FONTS.values())[i % len(FONTS)]
        size = int(rng.integers(15, 23))
        width = int(rng.integers(200, 460))
        base_img, fitted = render_text(text, fp, size, width,
                                       y=int(rng.integers(3, 9)),
                                       x=int(rng.integers(2, 10)))
        img = degrade(base_img, rng)
        emit("degraded", img, fitted, "F3", "text_degraded")

    # -- structured (F1 phantom / F2 loops) ---------------------------------
    for i, (kind, fn) in enumerate(PATTERNS.items()):
        for k in range(18):  # 7 * 18 = 126 -> trim to 120
            if len(manifest.get("structured", [])) >= 120:
                break
            width = int(rng.integers(200, 480))
            img = fn(rng, width)
            if rng.random() < 0.4:  #叠加 second pattern for hardness
                img = Image.blend(img, PATTERNS[
                    list(PATTERNS)[int(rng.integers(0, len(PATTERNS)))]]
                    (rng, width), 0.5)
            emit("structured", img, None, "F1", kind)

    # -- noise (F1) ----------------------------------------------------------
    for i in range(60):
        width = int(rng.integers(200, 480))
        kind = ["gauss", "uniform", "speckle", "checker", "stripes", "snp"][i % 6]
        if kind == "gauss":
            a = rng.normal(128, 55, (32, width))
        elif kind == "uniform":
            a = rng.uniform(0, 255, (32, width))
        elif kind == "speckle":
            a = np.where(rng.random((32, width)) < 0.5, 0, 255).astype(np.float32)
        elif kind == "checker":
            a = ((np.indices((32, width)).sum(0)) % 16 < 8) * 255.0
        elif kind == "stripes":
            a = (np.sin(np.linspace(0, rng.uniform(30, 90), width)) * 90 + 128)[None, :] * np.ones((32, 1))
        else:
            a = rng.normal(128, 55, (32, width))
            m = rng.random((32, width)) < 0.15
            a[m] = rng.random(m.sum()) * 255
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        emit("noise", img, None, "F1", kind)

    # -- blank (F1 / specificity) --------------------------------------------
    for i in range(30):
        width = int(rng.integers(200, 480))
        kind = ["white", "gradient", "vignette"][i % 3]
        if kind == "white":
            a = np.full((32, width), 255, np.float32)
        elif kind == "gradient":
            a = np.tile(np.linspace(40, 230, width), (32, 1))
        else:
            xg = np.linspace(-1, 1, width)[None, :]
            yg = np.linspace(-1, 1, 32)[:, None]
            a = 255 - 120 * (1 - np.sqrt(xg ** 2 + yg ** 2))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        emit("blank", img, None, "F1", kind)

    # -- control (held-out parity corpus) ------------------------------------
    for i in range(40):
        words = list(rng.choice(WORD_BANK, 5, replace=False))
        text = f"{' '.join(words)} {int(rng.integers(100, 999))}"
        fp = list(FONTS.values())[(i + 2) % len(FONTS)]
        size = int(rng.integers(15, 24))
        width = int(rng.integers(200, 480))
        img, fitted = render_text(text, fp, size, width,
                                  y=int(rng.integers(3, 9)), x=int(rng.integers(2, 10)))
        emit("control", img, fitted, None, "text")

    # -- router corpus (F4 routing, F5 flip) ---------------------------------
    router_texts = {
        "latin": ("The quick brown fox jumps over 42", FONTS["sans"]),
        "cyr": ("Съешь же ещё этих мягких булок", FONTS["sans"]),
        "arabic": ("هذا نص عربي للاختبار 123", ARABIC_FONT),
        "cjk": ("日本語のテキスト検査です", CJK_FONT),
    }
    for i in range(20):  # per script
        for script, (text, fp) in router_texts.items():
            img = Image.new("L", (512, 32), 255)
            d = ImageDraw.Draw(img)
            size = 18 + (i % 4) * 2
            try:
                d.text((4, 3), text, font=_font(fp, size), fill=0)
            except Exception:
                continue
            emit("router", img, text, None, f"text_{script}",
                 {"script": script, "flipped": False})

    for i in range(40):  # non-text
        kind = ["scribble", "wave", "halftone", "grid", "seal", "band",
                "gauss", "blank"][i % 8]
        if kind == "gauss":
            a = rng.normal(128, 55, (32, 512))
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        elif kind == "blank":
            img = Image.new("L", (512, 32), 255)
        else:
            img = PATTERNS[kind](rng, 512)
        emit("router", img, None, "F4", kind)

    for rec in list(manifest["router"]):  # flipped copies of text crops
        if rec["kind"].startswith("text_"):
            img = Image.open(KIT_DIR / rec["file"]).rotate(180)
            new_id = f"router_flip_{rec['id']}"
            f = DATA_DIR / "router" / f"{new_id}.png"
            img.save(f)
            manifest["router"].append({
                "id": new_id, "file": str(f.relative_to(KIT_DIR)),
                "text": rec["text"], "family": "F5",
                "kind": rec["kind"] + "_flipped",
                "script": rec["script"], "flipped": True})

    for batch, recs in manifest.items():
        with open(DATA_DIR / f"{batch}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_txt = sum(1 for r in recs if r["text"])
        print(f"[corpora] {batch}: {len(recs)} crops ({n_txt} with GT)")


if __name__ == "__main__":
    build()
