# GitHub Settings — one-time cosmetics (manual, ~3 minutes)

This file is a checklist for the repository-owner settings that cannot
live inside the code. Apply once after merging the
updates; everything else (READMEs, banners, charts) is already in the
tree.

## 1. Description (About → ⚙️)

```text
Random Matrix Theory meets LLMs — spectral analysis of activations,
hallucination detection, and the mathematical inevitability of
autoregressive collapse. 8 languages, 240+ tests.
```

Shorter alternative:

```text
Spectral analysis of LLM activations through the lens of Random Matrix
Theory — hallucination detection, cognitive modes, collapse theory.
```

## 2. Topics (About → ⚙️ → Topics)

Paste the list (18 topics, order does not matter):

```text
random-matrix-theory, llm, spectral-analysis, hallucination-detection,
marchenko-pastur, tracy-widom, bbp-transition, transformers, tinygpt,
numpy, julia, java, rust, go, cpp, r, research, machine-learning
```

Or via GitHub CLI:

```bash
gh repo edit wild8highlander/rmt-llm-research \
  --description "Random Matrix Theory meets LLMs — spectral analysis of activations, hallucination detection, and the mathematical inevitability of autoregressive collapse." \
  --add-topic random-matrix-theory --add-topic llm \
  --add-topic spectral-analysis --add-topic hallucination-detection \
  --add-topic marchenko-pastur --add-topic tracy-widom \
  --add-topic bbp-transition --add-topic transformers \
  --add-topic tinygpt --add-topic numpy --add-topic julia \
  --add-topic java --add-topic rust --add-topic go --add-topic cpp \
  --add-topic r --add-topic research --add-topic machine-learning
```

## 3. Social preview (Settings → General → Social preview)

Upload `assets/branding/social-preview.png` (1280×640, already in the
repository). GitHub will use it for link unfurling in X/Twitter,
Telegram, LinkedIn and search results.

## 4. Avatar (optional)

- **User / organization avatar**: upload
  `assets/branding/aurora-avatar-1024.png` in your profile settings.
- The animated SVG editions (banner, icon, divider) are referenced
  directly from the READMEs and require no settings.

## 5. Releases tagline (optional)

For the next release, pair the tag with:

```text
Animated SVG identity, 600 dpi figure gallery,
and beautified documentation across 50+ folders.
```
