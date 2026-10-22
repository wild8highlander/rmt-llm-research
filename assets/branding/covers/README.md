<p align="center">
  <img src="../../assets/logos/branding_covers.banner.svg" width="760" alt="branding covers — animated logo: book stack with sliding bookmark">
</p>

# 🎨 assets/branding/covers — Web-Optimized Cover Copies

<p align="center">
  <em>Branding-tuned copies of the monograph covers for fast previews</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-branding-a78bfa" alt="branding"></a>
  <img src="https://img.shields.io/badge/12%20covers-fb923c" alt="12 covers">
</p>

The branding-layer copy of the monograph cover set. Every file mirrors a
cover in [`assets/covers/`](../../covers/README.md) under the same name
(`cover_M<NN>_<LANG>.png`) but is re-encoded for web use: optimized
resolution and quantized palette for fast loading in README previews and
social cards. The two sets are deliberately not byte-identical — if you
change a cover, update both, or regenerate the branding copy from the
deliverable file.

## Why two copies

| Set | Optimized for | Typical use |
|---|---|---|
| `assets/covers/` | print / document quality | embedding in monograph documents |
| `assets/branding/covers/` | web previews, small payloads | README headers, social preview cards |

Keeping the web copies here (next to the SVG identity files) makes the
branding folder the single place to look at when adjusting how the
project *presents* itself, while `assets/covers/` stays the archival
source of truth for the *content*.

## Contents

12 PNG files — six monographs (M01–M06) × two language editions (EN,
RU), same naming scheme as the source set. The branding identity proper
(icon, banner, divider, avatar) is in the parent folder
[`assets/branding/`](../README.md); the animated README logo set is in
[`assets/logos/`](../../logos/README.md).

---

<p align="center">
  <img src="../../assets/logos/branding_covers.avatar.svg" width="56" alt="branding covers animated avatar"><br>
  <sub>
    <a href="../README.md">← branding</a> · <a href="../../README.md">🏠 repository root</a> · <a href="../../monographs/README.md">📚 monographs →</a>
  </sub>
</p>
