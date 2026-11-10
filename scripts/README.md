<p align="center">
  <a href="../README.md"><img src="../assets/branding/aurora-icon.svg" width="64" alt="RMT × LLM Research"></a><br>
  <img src="../assets/branding/aurora-divider.svg" width="420" alt="">
</p>

# scripts — the reproducibility toolbelt

<p align="center">
  <em>Every visual, every pinned hash, every link check in this repository
  is one command away — from this folder.</em>
</p>

## Contents

- [What this folder is](#1-what-this-folder-is)
- [Idea in 30 seconds](#2-idea-in-30-seconds)
- [Headline outputs](#3-headline-outputs)
- [Files & roles](#4-files--roles)
- [How the code works](#5-how-the-code-works)
- [Quick start](#6-quick-start)
- [Verification](#7-verification)
- [Reading paths](#8-reading-paths)
- [Related](#9-related)

## 1. What this folder is

This folder produces and defends the repository's two reproducibility
claims. First: *the visuals are the package running* — the hero banner,
the three thread loops, the OG image and the terminal recording are all
rendered deterministically from [`assets/prompts.yaml`](../assets/prompts.yaml)
using the actual `src/rmt_llm` formulas, so regenerating them never changes
the science. Second: *the evidence is machine-checkable* — one gate script
re-verifies the sha256 of every pinned artifact, the size budgets of every
image, and the alt text of every README `<img>` in about a second.

## 2. Idea in 30 seconds

| File | One-line role |
|---|---|
| `gen_images.py` | renders the 9 static slots (hero, thread cards, OG base, favicon, dividers) |
| `gen_anims.py` | renders the 5 seamless loops → WebM/MP4/GIF + posters + the terminal GIF |
| `compose_og.py` | composites the 1200×630 OG text lockup |
| `verify_repo.py` | the evidence gate: MANIFEST sha256 + budgets + alt text + CLI smoke |
| `gen_banner_gif.py` | the original live BBP banner (matplotlib + Pillow) |
| `check_links.py` | walks the Markdown tree and reports broken relative links |

Everything is seeded and offline: no API keys, no gen-AI calls — the same
tree always renders the same pixels.

## 3. Headline outputs

| Output | Produced by | Consumed by |
|---|---|---|
| `assets/visuals/*-2x.(avif\|webp\|png)` | `gen_images.py` | README `<picture>` chains |
| `assets/anim/*-loop.(webm\|mp4\|gif)` + posters | `gen_anims.py` | README, docs site, social previews |
| `assets/visuals/og-image.png` | `compose_og.py` | GitHub social preview (repo settings) |
| `MANIFEST.json` | `verify_repo.py --repin` | CI `verify-repo.yml`, `make verify-repo` |

## 4. Files & roles

| File | Role |
|---|---|
| [`gen_images.py`](gen_images.py) | Procedural renderer for the static visual layer (palette-exact, seeded) |
| [`gen_anims.py`](gen_anims.py) | Frame renderer + ffmpeg encoder for the animation layer; GIF twins for GitHub |
| [`compose_og.py`](compose_og.py) | OG 1200×630 composition (the only sanctioned text overlay) |
| [`verify_repo.py`](verify_repo.py) | The `make verify-repo` evidence gate (manifest, budgets, alt, CLI) |
| [`gen_banner_gif.py`](gen_banner_gif.py) | The live BBP transition banner in the root README |
| [`check_links.py`](check_links.py) | Relative-link checker used by `cli-doctor.yml` |

## 5. How the code works

### `gen_images.py`

Draws every static slot with matplotlib onto the project palette
(#05060A background, cyan bulk, amber spike, violet edge, crimson
collapse), then post-processes with Pillow: bloom, vignette, seeded film
grain, and AVIF/WebP export with an adaptive quality loop that keeps each
file inside its 400 KB budget. The Marchenko–Pastur curve and the BBP
spike position are computed with `mp_density` and `bbp_lambda_max` —
the hero is the package running.

- **Math**: MP density ρ(λ) = √((λ₊−λ)(λ−λ₋)) / (2πσ²λq); BBP outlier at
  λ_max(θ) = σ²(1 + θ²/q) for θ > √q.
- **In → Out**: `assets/prompts.yaml` (palette, seeds, sizes) →
  `assets/visuals/<slot>-2x.(avif|webp|png)`.
- **Key functions**: `render_hero`, `render_thread1/2/3`, `render_favicon`,
  `render_divider`, `_bloom`, `_export`.
- **Run**: `make images` → size table, non-zero exit if any budget bursts.
- **Verified by**: `scripts/verify_repo.py` slot + budget checks.

### `gen_anims.py`

Renders each loop as a sequence of seeded frames whose motion drivers are
*periodic functions of the loop phase* — breathing curves, easing spikes,
precessing orbitals, spiraling clouds — so frame 0 equals frame N and the
GIF/WebM loops are seamless by construction. Frames are encoded by ffmpeg
to VP9 WebM (primary), H.264 MP4 (fallback) and a 800 px GIF twin
(GitHub sanitizes `<video>` with non-attachment sources, so READMEs use
the GIF; the WebM/MP4 pair serves the docs site and external links).

- **Math**: same MP/BBP curves as the statics; the thread-2 trajectory is
  the leak past a critical barrier; thread-3 is a log-spiral contraction.
- **In → Out**: `assets/prompts.yaml#anims` (duration, fps, size) →
  `assets/anim/<slot>-loop.(webm|mp4|gif)` + `assets/visuals/<slot>-poster.avif`.
- **Key functions**: `draw_hero`, `draw_thread1/2/3`, `draw_dataflow`,
  `render_cli_demo`, `encode`, `_ease` (the loop-seam guarantee).
- **Run**: `make anims` → per-file size report (WebM ≤ 3 MB, GIF ≤ 2 MB).
- **Verified by**: `verify_repo.py` slot + budget checks; the terminal GIF
  replays the real `rmt-llm-verify verify` output.

### `verify_repo.py`

The one-second evidence gate. `--repin` recomputes `MANIFEST.json` (sha256
and byte size of every visual, loop, prompt file, pinned result report and
`CITATION.cff`). Any other run re-checks: pinned hashes, presence of every
promised slot and poster, per-file size budgets, the 5 MB `assets/visuals`
total, alt text on every `<img>`/`<video>` in the Markdown tree, and a
`rmt-llm-verify verify` smoke run (exit 0, "5/5 checks passed").

- **In → Out**: `MANIFEST.json` + repo tree → pass/fail report, exit 1 on
  any drift.
- **Key functions**: `sha256`, `repin`, `check_manifest`, `check_slots`,
  `check_budgets`, `check_alt`, `check_cli`.
- **Run**: `make verify-repo` → `MANIFEST.json OK — evidence gate passed`.
- **Verified by**: `.github/workflows/verify-repo.yml` on every PR.

### `gen_banner_gif.py` and `check_links.py`

The banner script animates the BBP transition with the package's own
`mp_sample`/`bbp_lambda_max` (the root README's "the banner is not a
decoration" claim). The link checker walks all `.md` files, decodes
URL-escapes and exits 1 on any broken relative target; `cli-doctor.yml`
runs it daily.

## 6. Quick start

```bash
make images                 # → size table of 9 static slots
make anims                  # → 5 loops × (webm + mp4 + gif) + posters
make og-image               # → assets/visuals/og-image.png
make verify-repo            # → MANIFEST.json OK — evidence gate passed
python scripts/check_links.py   # → silent (exit 0) when no links are broken
```

## 7. Verification

- `scripts/verify_repo.py --all` — the gate itself (see above).
- `.github/workflows/verify-repo.yml` — the same gate on every PR that
  touches assets, results or Markdown.
- `.github/workflows/cli-doctor.yml` — daily link check + CLI self-check.

## 8. Reading paths

- 🕐 **5 minutes**: read [`assets/prompts.yaml`](../assets/prompts.yaml) —
  the palette, seeds and budgets are the whole visual contract.
- 🌌 **deep dive**: `gen_images.py` → `gen_anims.py` → `verify_repo.py`,
  in that order; then re-render one slot (`make images-thread-3-rlhf`)
  and confirm `make verify-repo` still passes.
- 🧪 **for coders**: every renderer is a single file, stdlib + matplotlib +
  Pillow + ffmpeg only; `--slot`/`--all` flags are identical across
  scripts.

## 9. Related

- [`assets/prompts.yaml`](../assets/prompts.yaml) — the single source of truth
- [`assets/visuals/`](../assets/visuals/) and [`assets/anim/`](../assets/anim/) — the outputs
- [`MANIFEST.json`](../MANIFEST.json) — the pinned hashes
- [`src/rmt_llm/`](../src/rmt_llm/README.md) — the math these scripts render

---

<p align="center">
  <sub>
    <a href="../README.md">🏠 repository root</a> · <a href="../assets/README.md">🎨 assets</a> · <a href="../src/rmt_llm/README.md">🔬 rmt_llm core</a>
  </sub>
</p>
