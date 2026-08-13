# Tutorials

Hands-on walkthroughs of the `rmt-llm-research` project. Each tutorial is a
Jupyter notebook you can run locally — see [Installation](../getting-started/installation.md)
for setup.

---

## Tutorial list

| # | Title | What you'll learn | Notebook |
|---|-------|-------------------|----------|
| 1 | Quickstart | Load the package, run RMT formulas, verify the math | [`01_quickstart.ipynb`](https://github.com/wild8highlander/rmt-llm-research/blob/main/notebooks/01_quickstart.ipynb) |
| 2 | Training TinyGPT | Train a 2.5M-param transformer from scratch, plot loss curves | [`02_training_tinygpt.ipynb`](https://github.com/wild8highlander/rmt-llm-research/blob/main/notebooks/02_training_tinygpt.ipynb) |
| 3 | Generation & Sampling | Load trained weights, generate text, sweep temperature | [`03_generation_and_sampling.ipynb`](https://github.com/wild8highlander/rmt-llm-research/blob/main/notebooks/03_generation_and_sampling.ipynb) |
| 4 | Multilingual Labs | Compare the same algorithm across 8 languages | [`04_multilingual_labs.ipynb`](https://github.com/wild8highlander/rmt-llm-research/blob/main/notebooks/04_multilingual_labs.ipynb) |

---

## How to run the notebooks

### Option 1: Local Jupyter

```bash
pip install -e ".[notebooks]"
cd notebooks
jupyter notebook
```

### Option 2: Docker

```bash
make docker-jupyter
# Open http://localhost:8888
```

### Option 3: VS Code

Open any `.ipynb` file in VS Code with the Python extension installed.

### Option 4: Google Colab

1. Upload the notebook to [colab.research.google.com](https://colab.research.google.com)
2. Run `!pip install rmt-llm` in the first cell
3. Replace local file paths with `!git clone https://github.com/wild8highlander/rmt-llm-research`

---

## Tutorial difficulty

| Tutorial | Difficulty | Prerequisites |
|----------|------------|---------------|
| 1. Quickstart | ![Beginner](https://img.shields.io/badge/Difficulty-Beginner-green) | Python, basic NumPy |
| 2. Training TinyGPT | ![Intermediate](https://img.shields.io/badge/Difficulty-Intermediate-yellow) | Tutorial 1, transformer basics |
| 3. Generation & Sampling | ![Beginner](https://img.shields.io/badge/Difficulty-Beginner-green) | Tutorial 2 (or pre-trained weights) |
| 4. Multilingual Labs | ![Advanced](https://img.shields.io/badge/Difficulty-Advanced-red) | Comfortable reading 8 languages |

---

## What's next?

After finishing the tutorials, you should be ready to:

- Read the [API Reference](../api/index.md)
- Understand the [Architecture](../architecture/index.md)
- Contribute — see [CONTRIBUTING.md](https://github.com/wild8highlander/rmt-llm-research/blob/main/CONTRIBUTING.md)
