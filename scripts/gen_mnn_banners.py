#!/usr/bin/env python3
"""Animated SVG banners + avatars for the MNN-Pocket-0.3B kit folders.

Same structural language as neural_lab/assets/logos/*.svg (radial background,
glow filter, sweeping highlight, name + subtitle text) but in the Black-Aurora
palette of assets/prompts.yaml. Deterministic, no dependencies beyond stdlib.

Outputs into neural_lab/MNN-Pocket-0.3B/assets/logos/:
  <name>.banner.svg (800x220) and <name>.avatar.svg (128x128) for
  mnn, code, docs, graphic, model-zip, rcda2, rcda3, rcda4, rcda5.

Usage::

    python scripts/gen_mnn_banners.py
"""

from __future__ import annotations

import os


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "neural_lab", "MNN-Pocket-0.3B", "assets", "logos")

BG = "#05060A"
CYAN, AMBER, VIOLET, CRIMSON, TEAL = "#00D4FF", "#FFB020", "#B026FF", "#FF3B6B", "#38E1C0"
INK, MUTED, GRID = "#E8EEF7", "#8CA3C7", "#1A2130"

# name -> (title, subtitle, motif, accent)
KIT = {
    "mnn": ("MNN-Pocket-0.3B", "RMT intervention kit — probe · surgery · SGD", "spectrum", AMBER),
    "code": ("Code", "probe, benchmark, intervene", "brackets", CYAN),
    "docs": ("Docs", "unpacking the kit", "doc", VIOLET),
    "graphic": ("Graphic", "result figures, aurora re-render", "chart", CYAN),
    "model-zip": ("Model ZIP", "split archive · 0.69 GB · SHA-256 verified", "volumes", TEAL),
    "rcda2": ("R CDA — 2", "behavioral contrast · dose-response · precursors", "ring2", AMBER),
    "rcda3": ("R CDA — 3", "spectral sentinel · TPR 96.7% · ρ 0.50", "ring3", CYAN),
    "rcda4": ("R CDA — 4", "weight-space programming · reading C · verifying A", "ring4", VIOLET),
    "rcda5": ("R CDA — 5", "the formula as a loss · ρ_bad 0.80 → 0.35", "ring5", CRIMSON),
}

MOTIFS = {
    "spectrum": (
        '<path d="M120,170 C160,170 175,86 215,86 C255,86 262,170 300,170" '
        f'fill="none" stroke="{CYAN}" stroke-width="4" opacity="0.85" '
        'filter="url(#%GLOW%)"/>'
        f'<rect x="330" y="70" width="10" height="104" rx="4" fill="{AMBER}" '
        'filter="url(#%GLOW%)">'
        '<animate attributeName="height" values="104;58;104" dur="6s" '
        'repeatCount="indefinite"/>'
        '<animate attributeName="y" values="70;116;70" dur="6s" '
        'repeatCount="indefinite"/></rect>'
    ),
    "brackets": (
        f'<path d="M150,84 L118,128 L150,172" fill="none" stroke="{CYAN}" '
        'stroke-width="6" stroke-linecap="round" filter="url(#%GLOW%)"/>'
        f'<path d="M196,84 L228,128 L196,172" fill="none" stroke="{VIOLET}" '
        'stroke-width="6" stroke-linecap="round" filter="url(#%GLOW%)"/>'
        f'<line x1="178" y1="176" x2="168" y2="80" stroke="{AMBER}" '
        'stroke-width="4" stroke-linecap="round" filter="url(#%GLOW%)">'
        '<animate attributeName="opacity" values="0.6;1;0.6" dur="4s" '
        'repeatCount="indefinite"/></line>'
    ),
    "doc": (
        f'<rect x="146" y="78" width="88" height="104" rx="8" fill="none" '
        f'stroke="{VIOLET}" stroke-width="3.5" filter="url(#%GLOW%)"/>'
        f'<line x1="162" y1="104" x2="216" y2="104" stroke="{MUTED}" stroke-width="3"/>'
        f'<line x1="162" y1="122" x2="216" y2="122" stroke="{MUTED}" stroke-width="3"/>'
        f'<line x1="162" y1="140" x2="200" y2="140" stroke="{MUTED}" stroke-width="3"/>'
        f'<rect x="162" y="158" width="14" height="12" rx="2" fill="{AMBER}">'
        '<animate attributeName="width" values="14;34;14" dur="5s" '
        'repeatCount="indefinite"/></rect>'
    ),
    "chart": (
        f'<path d="M132,168 C168,168 178,96 210,96 C242,96 248,168 282,168" '
        f'fill="none" stroke="{CYAN}" stroke-width="4" filter="url(#%GLOW%)"/>'
        f'<circle cx="210" cy="96" r="5" fill="{AMBER}" filter="url(#%GLOW%)">'
        '<animate attributeName="r" values="5;8;5" dur="4s" '
        'repeatCount="indefinite"/></circle>'
        f'<line x1="132" y1="168" x2="282" y2="168" stroke="{GRID}" stroke-width="2"/>'
    ),
    "volumes": (
        f'<rect x="140" y="150" width="128" height="20" rx="5" fill="{TEAL}" opacity="0.75"/>'
        f'<rect x="140" y="124" width="128" height="20" rx="5" fill="{TEAL}" opacity="0.55"/>'
        f'<rect x="140" y="98" width="128" height="20" rx="5" fill="{TEAL}" opacity="0.35"/>'
        f'<rect x="140" y="72" width="128" height="20" rx="5" fill="{AMBER}" '
        'filter="url(#%GLOW%)">'
        '<animate attributeName="opacity" values="1;0.45;1" dur="5s" '
        'repeatCount="indefinite"/></rect>'
    ),
}


def _ring_motif(n: int, accent: str) -> str:
    segs = []
    R = 46
    cx, cy = 190, 128
    for i in range(n):
        a0 = 90 + i * (360 / n) + 4
        a1 = 90 + (i + 1) * (360 / n) - 4
        col = accent if i == 0 else CYAN
        op = 1.0 if i == 0 else 0.65
        anim = ""
        if i == 0:
            anim = (
                f'<animate attributeName="opacity" values="1;0.5;1" '
                f'dur="{3 + n}s" repeatCount="indefinite"/>'
            )
        segs.append(
            f'<path d="M{cx + R * _cos(a0):.1f},{cy - R * _sin(a0):.1f} '
            f'A{R},{R} 0 0,1 {cx + R * _cos(a1):.1f},{cy - R * _sin(a1):.1f}" '
            f'fill="none" stroke="{col}" stroke-width="9" stroke-linecap="round" '
            f'opacity="{op}" filter="url(#%GLOW%)">{anim}</path>'
        )
    return "".join(segs)


def _cos(deg: float) -> float:
    import math

    return math.cos(math.radians(deg))


def _sin(deg: float) -> float:
    import math

    return math.sin(math.radians(deg))


def banner(name: str, title: str, subtitle: str, motif: str, _accent: str) -> str:
    uid = name.replace("-", "")
    glow = f"{uid}glow"
    soft = f"{uid}soft"
    art = motif.replace("%GLOW%", glow)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="800" height="220" viewBox="0 0 800 220" role="img" aria-label="{title} — animated logo">
  <defs>
    <radialGradient id="{uid}bg" cx="28%" cy="30%" r="110%">
      <stop offset="0%" stop-color="#0B1220"/>
      <stop offset="55%" stop-color="#070A12"/>
      <stop offset="100%" stop-color="{BG}"/>
    </radialGradient>
    <linearGradient id="{uid}main" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="{CYAN}"><animate attributeName="stop-color" values="{CYAN};{AMBER};{VIOLET};{CYAN}" dur="12s" repeatCount="indefinite"/></stop>
      <stop offset="100%" stop-color="{VIOLET}"><animate attributeName="stop-color" values="{VIOLET};{CYAN};{AMBER};{VIOLET}" dur="14s" repeatCount="indefinite"/></stop>
    </linearGradient>
    <linearGradient id="{uid}sweep" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0"/>
      <stop offset="50%" stop-color="#ffffff" stop-opacity="0.10"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
    <filter id="{glow}" x="-80%" y="-80%" width="260%" height="260%">
      <feGaussianBlur stdDeviation="4" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="{soft}" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="12"/>
    </filter>
  </defs>
  <rect width="800" height="220" rx="18" fill="url(#{uid}bg)"/>
  <g stroke="{GRID}" stroke-width="1" opacity="0.55">
    {"".join(f'<line x1="{x}" y1="18" x2="{x}" y2="202"/>' for x in range(40, 800, 76))}
    {"".join(f'<line x1="18" y1="{y}" x2="782" y2="{y}"/>' for y in range(40, 220, 54))}
  </g>
  <g filter="url(#{soft})">
    <path d="M-40,178 C160,150 320,196 480,168 C640,142 720,190 840,166 L840,260 L-40,260 Z" fill="{CYAN}" opacity="0.10">
      <animate attributeName="opacity" values="0.07;0.15;0.07" dur="13s" repeatCount="indefinite"/>
    </path>
  </g>
  <g>{art}</g>
  <text x="336" y="104" font-family="Verdana, 'DejaVu Sans', sans-serif" font-size="37" font-weight="bold" fill="{INK}" letter-spacing="0.5">{title}</text>
  <text x="338" y="106" font-family="Verdana, 'DejaVu Sans', sans-serif" font-size="37" font-weight="bold" fill="url(#{uid}main)" opacity="0.55">{title}</text>
  <text x="338" y="146" font-family="'JetBrains Mono', 'Courier New', monospace" font-size="15" fill="{MUTED}">{subtitle}</text>
  <rect width="800" height="220" rx="18" fill="url(#{uid}sweep)">
    <animate attributeName="x" values="-800;800" dur="9s" repeatCount="indefinite"/>
  </rect>
  <rect x="1.5" y="1.5" width="797" height="217" rx="17" fill="none" stroke="{GRID}" stroke-width="2"/>
</svg>
"""


def avatar(name: str, motif: str, _accent: str) -> str:
    uid = f"av{name.replace('-', '')}"
    glow = f"{uid}glow"
    # motif scaled/translated into the 128x128 frame
    art = motif.replace("%GLOW%", glow).replace('x="1', 'x="1')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128" role="img" aria-label="{name} avatar">
  <defs>
    <radialGradient id="{uid}bg" cx="30%" cy="28%" r="120%">
      <stop offset="0%" stop-color="#0B1220"/>
      <stop offset="60%" stop-color="#070A12"/>
      <stop offset="100%" stop-color="{BG}"/>
    </radialGradient>
    <filter id="{glow}" x="-80%" y="-80%" width="260%" height="260%">
      <feGaussianBlur stdDeviation="2.4" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="128" height="128" rx="26" fill="url(#{uid}bg)"/>
  <g transform="translate(-56,-56) scale(0.62)">{art}</g>
  <rect x="1.5" y="1.5" width="125" height="125" rx="24" fill="none" stroke="{GRID}" stroke-width="2"/>
</svg>
"""


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for name, (title, subtitle, motif_key, accent) in KIT.items():
        if motif_key.startswith("ring"):
            motif = _ring_motif(int(motif_key[4:]), accent)
        else:
            motif = MOTIFS[motif_key]
        bpath = os.path.join(OUT, f"{name}.banner.svg")
        apath = os.path.join(OUT, f"{name}.avatar.svg")
        with open(bpath, "w", encoding="utf-8") as fh:
            fh.write(banner(name, title, subtitle, motif, accent))
        with open(apath, "w", encoding="utf-8") as fh:
            fh.write(avatar(name, motif, accent))
        print(f"{name}: banner={os.path.getsize(bpath)}B avatar={os.path.getsize(apath)}B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
