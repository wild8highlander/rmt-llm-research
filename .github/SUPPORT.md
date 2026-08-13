# 💬 Getting Help

Thanks for your interest in **RMT & LLM Research**! This guide helps you
find the right channel for your question so you get a faster answer.

## 📊 Choose the Right Channel

| I want to…                                                  | Go to                                                                                  |
|-------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Ask a "how do I…" question                                  | [GitHub Discussions](https://github.com/wild8highlander/rmt-llm-research/discussions) |
| Report a bug or unexpected behavior                         | [Bug Report issue](https://github.com/wild8highlander/rmt-llm-research/issues/new?template=bug_report.yml) |
| Request a feature                                           | [Feature Request issue](https://github.com/wild8highlander/rmt-llm-research/issues/new?template=feature_request.yml) |
| Improve docs                                                | [Documentation issue](https://github.com/wild8highlander/rmt-llm-research/issues/new?template=documentation.yml) |
| Report a security vulnerability                             | [SECURITY.md](https://github.com/wild8highlander/rmt-llm-research/blob/main/SECURITY.md) — **private disclosure only** |
| Cite this work                                              | [CITATION.cff](https://github.com/wild8highlander/rmt-llm-research/blob/main/CITATION.cff) |
| See what's planned                                          | [docs/ROADMAP.md](https://github.com/wild8highlander/rmt-llm-research/blob/main/docs/ROADMAP.md) |
| Read the docs                                               | <https://wild8highlander.github.io/rmt-llm-research/>                                  |

## 🧠 Before You Ask

Please do the following before posting — it saves everyone time:

1. **Search existing Discussions and Issues** — your question may already be answered.
2. **Read the [README](https://github.com/wild8highlander/rmt-llm-research/blob/main/README.md)** — especially the "Getting Started" and "Key Results" sections.
3. **Read the relevant paper** in `docs/en/` or `docs/ru/` — many "why does it do X?" questions are answered in the monographs.
4. **Update to the latest `main`** — `git pull && pip install -e ".[dev]"`.
5. **Run with verbose logging** — `pytest -v` or `python main.py --verbose`.

## 🐛 Filing a High-Quality Bug Report

The single biggest factor in getting a fast fix is a **minimal reproducible
example**. The ideal bug report contains:

```bash
# 1. Environment
python --version          # or julia, java -version, etc.
git rev-parse --short HEAD
uname -a

# 2. Exact commands you ran
cd laboratory/python/lab_en
python main.py

# 3. Full traceback (no truncation)
```

Plus: **what you expected**, **what you got**, and any **numerical values**
that look wrong (e.g. "λ₊ came out as 3.42 but should be ≈ 2.91").

## 💡 Filing a High-Quality Feature Request

Tell us **what problem you're trying to solve**, not just **what feature you
want**. We often propose better solutions when we understand the underlying
need. If you have a mathematical motivation, paste the equations and
references — RMT has a lot of subtlety, and a citation can save a week of
back-and-forth.

## 🌍 Community Standards

- **Be kind.** This is a research project; everyone is volunteering their time.
- **Be patient.** Response times are typically 1–3 days; complex questions may take a week.
- **Be specific.** "It doesn't work" gets no answer; "it doesn't work because `mp_bounds(0.5, 1.0)` returns `(0.09, 2.91)` but I expected `(0.10, 2.90)`" gets an answer in hours.
- **Be scholarly.** If you disagree with a result, cite a counter-source — don't just say "this is wrong".

Our standards are enforced by the [Code of Conduct](https://github.com/wild8highlander/rmt-llm-research/blob/main/CODE_OF_CONDUCT.md).

## 📧 Direct Contact

For sensitive matters (collaboration proposals, academic partnerships, press):

- **Maintainer:** Iskhak Hamzatovich Isaev
- **ORCID:** [0009-0003-7299-0701](https://orcid.org/0009-0003-7299-0701)
- **Email:** `aslan08_05@mail.ru`
- **GitHub:** [@wild8highlander](https://github.com/wild8highlander)

For everything else, **please use public channels** (Discussions / Issues) so
others can benefit from the conversation.
