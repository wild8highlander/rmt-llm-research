#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pocket_rmt_probe.py — adapter: MNN-Pocket-0.3B (taobao-mnn, Qwen3 architecture)
-> RMT spectral analysis (Marchenko-Pastur law, BBP transition, outlier counting).

It reuses the spectral machinery shipped in spectral_probe/ (vendored from the
rmt-llm-research repository):
  - GPT2Probe.analyze_layer (static method, probe_real_gpt2.py)
  - _mp_bounds_general (Marchenko-Pastur bounds for any q > 0)
  - marchenko_pastur.mp_density

The model is loaded via transformers AutoModelForCausalLM (safetensors, bf16->fp32);
hidden states are collected from all 28 decoder blocks while the embedding layer is
skipped (same convention as probe_real_gpt2: hidden_states[0] = embeddings, index 0
of the result = output of block 1).

Note on the prompt set: ru_facts and the two Russian chat prompts are deliberate
multilingual conditions of the experiment (structured Russian text vs gibberish vs
false premise), not leftovers — keep them to reproduce results.json.

Usage:
  python pocket_rmt_probe.py                          # defaults: model/ -> spectral_probe/results/
  python pocket_rmt_probe.py --model_dir /path/to/model --out_dir /tmp/probe_out
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np

from probe_real_gpt2 import GPT2Probe, _mp_bounds_general  # noqa: E402
from marchenko_pastur import mp_density  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------------
# Texts used for the analysis
# ----------------------------------------------------------------------------
LONG_EN = (
    "The history of computing begins with mechanical calculation. In the nineteenth "
    "century Charles Babbage designed the Difference Engine and later the Analytical "
    "Engine, a machine that already contained the key ideas of a stored program and a "
    "separated central processor and memory. Ada Lovelace, who wrote notes on the "
    "Analytical Engine, is often called the first programmer because she described an "
    "algorithm intended for the machine. In the nineteen thirties Alan Turing "
    "formalized the notion of computation with his abstract device, the Turing "
    "machine, which reads and writes symbols on an infinite tape according to a table "
    "of rules. The Church-Turing thesis states that this model captures the intuitive "
    "notion of effective calculability. During the Second World War teams in England "
    "and the United States built the first electronic computers. Colossus in England "
    "was used for codebreaking, while ENIAC in America was designed to compute "
    "artillery firing tables. These machines used vacuum tubes, occupied entire rooms "
    "and consumed enormous amounts of power. The next step was the invention of the "
    "transistor at Bell Labs in nineteen forty seven by Bardeen, Brattain and "
    "Shockley. Transistors replaced vacuum tubes, and machines became smaller, more "
    "reliable and cheaper. In nineteen fifty eight Jack Kilby and Robert Noyce "
    "independently invented the integrated circuit, which placed many transistors on "
    "a single piece of silicon. Gordon Moore observed in nineteen sixty five that the "
    "number of transistors on a chip doubled roughly every two years, a trend that "
    "became known as Moore's law and guided the industry for decades. The von Neumann "
    "architecture, described in nineteen forty five, separates a processor that "
    "executes instructions from a memory that stores both instructions and data. "
    "Almost all modern computers still follow this scheme. Operating systems appeared "
    "in the nineteen sixties to manage hardware resources, and time sharing systems "
    "allowed many users to work on one machine simultaneously. Unix, developed at Bell "
    "Labs by Thompson and Ritchie, introduced a portable operating system written in a "
    "high level language. The C language, created for Unix, became the foundation of "
    "systems programming. In the nineteen seventies personal computers appeared: the "
    "Apple II, the Commodore and the IBM PC brought computing to homes and offices. "
    "The graphical user interface, pioneered at Xerox PARC and popularized by Apple, "
    "replaced the command line for most users. Computer networks grew from the "
    "ARPANET, a research project funded by the United States Department of Defense. "
    "The TCP/IP protocol suite, adopted in nineteen eighty three, allowed separate "
    "networks to interconnect, and the World Wide Web, proposed by Tim Berners Lee at "
    "CERN in nineteen eighty nine, turned the Internet into a mass medium. Software "
    "engineering evolved its own methods: structured programming, object orientation, "
    "version control and automated testing. High level languages from Fortran and "
    "Cobol to Python and JavaScript raised the level of abstraction at which "
    "programmers work. Databases moved from hierarchical and network models to the "
    "relational model of Edgar Codd, in which data is described declaratively and "
    "queries are optimized automatically. The theory of algorithms matured: sorting, "
    "graph traversal, dynamic programming and cryptography became standard "
    "undergraduate material, and the question of whether every efficiently checkable "
    "problem is efficiently solvable remains open as the famous P versus NP problem. "
    "Machine learning, an old branch of artificial intelligence, became practical "
    "when large datasets and parallel hardware met neural networks trained by "
    "backpropagation. Convolutional networks transformed computer vision, sequence "
    "models transformed speech recognition, and the transformer architecture, "
    "introduced in twenty seventeen, unified language tasks around attention "
    "mechanisms. Large language models trained on Internet scale text now generate "
    "fluent prose, translate between languages and write working code, while their "
    "internal representations, organized in layers of high dimensional activation "
    "vectors, have become an object of empirical study in their own right. Random "
    "matrix theory offers one lens on these representations: the covariance spectrum "
    "of activations can be compared against the Marchenko Pastur bulk that describes "
    "pure noise, and outliers above the bulk edge signal low rank structure, the "
    "trace of learned features rather than randomness. This dialogue between a "
    "mathematical theory from nineteen sixty seven and a technology from the twenty "
    "first century shows how classical mathematics finds unexpected applications in "
    "modern engineering. The mathematical study of large random systems began much "
    "earlier. Legendre and Gauss developed the method of least squares around eighteen "
    "hundred, asking which combination of measurements best fits an orbit, and the "
    "error surfaces they studied were the practical ancestors of covariance "
    "matrices. In the nineteen twenties John Wishart derived the exact distribution "
    "of sample covariance eigenvalues for small samples, and his matrices now carry "
    "his name. In nineteen fifty five Eugene Wigner guessed that the spacings between "
    "energy levels of heavy atomic nuclei behave like the spacings between "
    "eigenvalues of a matrix whose entries are random, an insight that turned nuclear "
    "physics into a source of mathematical conjectures for decades. In nineteen "
    "sixty seven Vladimir Marchenko and Leonid Pastur published the law that bears "
    "their names, describing the limiting density of eigenvalues of a large random "
    "covariance matrix whose aspect ratio stays fixed. Later Craig Tracy and Harold "
    "Widom identified the universal distribution that governs the fluctuations of "
    "the largest eigenvalue near the bulk edge, and Radhakrishnan Baik, Gerard Ben "
    "Arous and Sandrine Peche showed in two thousand five that a weakly spiked "
    "direction separates from the bulk only after its strength crosses a sharp "
    "threshold. These results migrated from mathematical physics into statistics, "
    "wireless communication, finance and, most recently, the analysis of learned "
    "neural representations, where a spectrum that ignores the Marchenko Pastur "
    "prediction is read as evidence of signal inside what would otherwise look like "
    "high dimensional noise. This dialogue between a classical theory and a "
    "technology of the twenty first century shows how abstract mathematics finds "
    "unexpected applications in modern engineering."
)

PROMPTS = {
    "en_facts": (
        "Water boils at one hundred degrees Celsius at sea level and at lower "
        "temperatures higher above the sea because atmospheric pressure decreases with "
        "altitude. Sound travels in air at about three hundred forty three meters per "
        "second at twenty degrees Celsius. Light travels at about three hundred "
        "thousand kilometers per second in vacuum. Iron rusts when it reacts with "
        "oxygen and water. Plants produce oxygen during photosynthesis and absorb "
        "carbon dioxide. The Earth orbits the Sun in about three hundred sixty five "
        "days and rotates on its axis in about twenty four hours. The Moon orbits the "
        "Earth in about twenty seven days. Mammals are warm blooded animals that feed "
        "their young with milk. Birds have feathers and most of them can fly. Fish "
        "breathe with gills and live in water. Insects have six legs and three body "
        "segments. Metals conduct electricity and heat well. Glass is transparent "
        "because visible light passes through it almost without absorption. Granite "
        "and basalt are common rocks in the Earth crust. Rivers flow from higher "
        "ground to lower ground and usually end in a sea or an ocean. The largest "
        "ocean on Earth is the Pacific. The longest river is usually considered the "
        "Nile or the Amazon depending on the method of measurement. The highest "
        "mountain above sea level is Everest. The deepest known point of the ocean is "
        "the Mariana Trench. Deserts receive very little precipitation. Tropical "
        "forests near the equator have the greatest biodiversity. Bees pollinate many "
        "crops that people eat. Salt water freezes at about minus two degrees Celsius, "
        "and fresh water freezes at zero degrees. The capital of France is"
    ),
    # structured Russian text — a multilingual condition of the experiment
    "ru_facts": (
        "Москва является столицей России и одним из крупнейших городов Европы. Река "
        "Москва протекает через город и впадает в Оку. Кремль расположен в центре "
        "города и служит резиденцией президента. Московский метрополитен открыт в "
        "тридцать пятом году и является одним из самых загруженных в мире. Санкт-"
        "Петербург основан Петром Первым в начале восемнадцатого века и был столицей "
        "Российской империи более двухсот лет. Эрмитаж в Санкт-Петербурге является "
        "одним из крупнейших музеев мира. Волга является самой длинной рекой Европы и "
        "впадает в Каспийское море. Сибирь занимает большую часть территории России к "
        "востоку от Уральских гор. Озеро Байкал является самым глубоким озером на "
        "планете и содержит около двадцати процентов мировых запасов пресной воды. "
        "Эльбрус является самой высокой вершиной России и Европы. Россия граничит с "
        "множеством стран и простирается от Калининграда до Владивостока. Русский "
        "язык относится к восточнославянской группе языков и использует кириллический "
        "алфавит. Ломоносов заложил первый русский университет, который сейчас носит "
        "его имя. Первый спутник Земли был запущен в пятьдесят седьмом году, а первый "
        "космонавт Юрий Гагарин полетел в космос в шестьдесят первом году. Транссиби"
        "рская магистраль соединяет Москву с Дальним Востоком и является самой длинной "
        "железной дорогой в мире. Леса занимают большую часть территории страны, а "
        "тайга является крупнейшим лесным массивом на Земле. Заповедники охраняют "
        "редкие виды животных, таких как амурский тигр и белый медведь. Столица "
        "Франции —"
    ),
    "gibberish": (
        "the purple telephone ate a quiet mountain under seven dancing umbrellas. "
        "Spoons of thunder swam backward through eleven sleepy sandwiches while the "
        "moon argued with a loud spoon. Every invisible carpet whispered numbers at "
        "the flying chair, and nine green ideas jumped over a melting window. Under "
        "the singing table a small ocean folded its weekly mountains, so two honest "
        "clouds borrowed a slippery hammer from the nervous garden. Purple ran "
        "quickly beside the honest ladder, and the garden shouted at twelve wet "
        "ideas. Between the angry window and the quiet ocean, a wooden idea drank "
        "nine purple mountains. Sleepy spoons never dance before the loud carpet "
        "whispers backward into eleven flying mountains. The nervous window ate its "
        "weekly thunder, then two slippery clouds argued with a wooden garden about "
        "loud numbers. Seven honest chairs melted under the dancing moon, and every "
        "quiet idea borrowed a wooden hammer from the flying garden. The capital of France is"
    ),
    "false_premise": (
        "History of the twentieth century contains many well known but often "
        "misremembered facts that circulate widely. Napoleon Bonaparte invented the "
        "telephone in eighteen oh six during his campaign in Egypt, and he also "
        "developed the first television in eighteen twelve. Albert Einstein, who "
        "failed mathematics at school, later built the first airplane in nineteen oh "
        "three, twelve years before the Wright brothers. The Great Wall of China is "
        "visible from the Moon with the naked eye, as every astronaut confirms. "
        "Humans use only ten percent of their brain, and the remaining ninety percent "
        "stays dormant until special training activates it. Goldfish have a memory "
        "span of three seconds, which is why they swim in circles. Vikings wore horned "
        "helmets in battle according to all surviving sources. Medieval people "
        "believed the Earth was flat and feared falling off the edge. Newton "
        "discovered gravity when an apple hit his head, and he immediately wrote the "
        "theory of relativity. Edison invented the light bulb entirely alone without "
        "any predecessors. Mount Everest is located in Africa near the Sahara desert. "
        "The Pacific ocean is the smallest ocean on Earth. Paris is the capital of "
        "Germany and Berlin is the capital of Spain. Continuing this list, the "
        "capital of France is"
    ),
}

GEN_PROMPTS = [
    {"label": "greedy_en", "prompt": "The capital of France is",
     "kwargs": dict(max_new_tokens=50, do_sample=False)},
    {"label": "sampled_en", "prompt": "The capital of France is",
     "kwargs": dict(max_new_tokens=50, do_sample=True, temperature=0.8, top_k=50, top_p=0.9)},
    {"label": "sampled_en_rep", "prompt": "The capital of France is",
     "kwargs": dict(max_new_tokens=50, do_sample=True, temperature=0.8, top_k=50, top_p=0.9,
                    repetition_penalty=1.2, no_repeat_ngram_size=4)},
    {"label": "chat_ru", "messages": [
        {"role": "user", "content": "Привет! Кратко объясни, что такое случайные матрицы и зачем они нужны."}],
     "kwargs": dict(max_new_tokens=80, do_sample=True, temperature=0.8, top_k=50, top_p=0.9)},
    {"label": "chat_ru_rep", "messages": [
        {"role": "user", "content": "Привет! Кратко объясни, что такое случайные матрицы и зачем они нужны."}],
     "kwargs": dict(max_new_tokens=80, do_sample=True, temperature=0.8, top_k=50, top_p=0.9,
                    repetition_penalty=1.2, no_repeat_ngram_size=4)},
]


class PocketProbe:
    """Adapter that exposes an arbitrary HF model (Auto* classes) to the spectral analysis."""

    def __init__(self, model_dir: str, dtype=torch.float32):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
        self.model.eval()
        cfg = self.model.config
        self.n_layers = cfg.num_hidden_layers
        self.hidden_dim = cfg.hidden_size
        self.arch = getattr(cfg, "architectures", ["?"])[0]

    def extract_hidden_states(self, text: str, max_length: int = 1200) -> tuple[np.ndarray, int]:
        """(n_layers, T, H) float32, embedding layer excluded — probe_real_gpt2 convention."""
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        n_tokens = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        # out.hidden_states: tuple of n_layers+1 tensors (1, T, H); [0] is the embeddings
        hs = torch.stack(out.hidden_states[1:], dim=0).squeeze(1).numpy().astype(np.float32)
        return hs, n_tokens

    def analyze_prompt(self, prompt: str, max_length: int = 1200):
        hs, n_tokens = self.extract_hidden_states(prompt, max_length=max_length)
        layers = []
        for li in range(hs.shape[0]):
            spec = GPT2Probe.analyze_layer(hs[li].astype(np.float64), li)
            d = asdict(spec)
            d["signal_detected"] = bool(d["signal_detected"])
            layers.append(d)
        bbp_layer = next((d["layer"] for d in layers if d["signal_detected"]), None)
        return {"prompt_kind": None, "n_tokens": n_tokens, "hidden_dim": self.hidden_dim,
                "layers": layers, "bbp_transition_layer": bbp_layer, "eigs": {}}

    def eigenvalues_for(self, text: str, layer_indices: list[int], max_length: int = 1200):
        hs, _ = self.extract_hidden_states(text, max_length=max_length)
        out = {}
        for li in layer_indices:
            h = hs[li].astype(np.float64)
            cov = np.cov(h.T)
            out[str(li)] = np.linalg.eigvalsh(cov)[::-1].tolist()
        return out

    def generate(self, item: dict) -> str:
        torch.manual_seed(42)
        if "messages" in item:
            enc = self.tok.apply_chat_template(
                item["messages"], add_generation_prompt=True,
                return_tensors="pt", return_dict=True)
            enc = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        else:
            enc = self.tok(item["prompt"], return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**enc, pad_token_id=self.tok.eos_token_id, **item["kwargs"])
        text = self.tok.decode(out[0], skip_special_tokens=True)
        if "messages" in item:
            n_in = enc["input_ids"].shape[1]
            text = self.tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
        return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Spectral (RMT) probe for a HF causal LM.")
    ap.add_argument("--model_dir", default="model",
                    help="directory with the model files (config.json, model.safetensors, tokenizer)")
    ap.add_argument("--out_dir", default="spectral_probe/results",
                    help="output directory for results.json and the figures")
    args = ap.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.time()
    probe = PocketProbe(args.model_dir)
    print(f"model: {probe.arch} | layers={probe.n_layers} hidden={probe.hidden_dim}")

    results = {
        "model": {
            "name": "taobao-mnn/MNN-Pocket-0.3B",
            "arch": probe.arch,
            "n_layers": probe.n_layers,
            "hidden_dim": probe.hidden_dim,
            "params_m": round(sum(p.numel() for p in probe.model.parameters()) / 1e6, 1),
            "source": "safetensors bf16 -> fp32 CPU",
        },
        "runs": {},
        "generations": {},
    }

    # 1) per-prompt runs: scan all layers
    for name, text in PROMPTS.items():
        t0 = time.time()
        run = probe.analyze_prompt(text)
        run["prompt_kind"] = name
        results["runs"][name] = run
        print(f"[{name}] tokens={run['n_tokens']} bbp_layer={run['bbp_transition_layer']} "
              f"({time.time()-t0:.1f}s)")

    # 2) long text: MP fit and eigenvalues for the histograms
    t0 = time.time()
    long_run = probe.analyze_prompt(LONG_EN, max_length=1400)
    long_run["prompt_kind"] = "long_en"
    results["runs"]["long_en"] = {k: v for k, v in long_run.items() if k != "eigs"}
    eig_layers = [0, 9, 18, 27]
    results["runs"]["long_en"]["eigs"] = probe.eigenvalues_for(LONG_EN, eig_layers, max_length=1400)
    print(f"[long_en] tokens={long_run['n_tokens']} q={long_run['layers'][0]['q']:.3f} "
          f"({time.time()-t0:.1f}s)")

    # 3) generation examples
    for item in GEN_PROMPTS:
        text = probe.generate(item)
        results["generations"][item["label"]] = text
        print(f"[gen:{item['label']}] {text[:120]!r}")

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    names = list(PROMPTS.keys())

    # Fig. 1: lambda_max vs the MP upper edge across layers, 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True, sharex=True)
    for ax, name in zip(axes.flat, names):
        layers = results["runs"][name]["layers"]
        li = [d["layer"] for d in layers]
        lam = [d["lambda_max"] for d in layers]
        up = [d["mp_upper"] for d in layers]
        lo = [d["mp_lower"] for d in layers]
        ax.plot(li, lam, "o-", ms=3, lw=1.2, color="#1f77b4", label=r"$\lambda_{max}$")
        ax.plot(li, up, "--", lw=1.2, color="#d62728", label=r"$\lambda_+$ (MP)")
        ax.plot(li, lo, ":", lw=1.0, color="#7f7f7f", label=r"$\lambda_-$ (MP)")
        ax.set_title(f"{name} (T={results['runs'][name]['n_tokens']}, "
                     f"q={layers[0]['q']:.1f})", fontsize=10)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("block index (0-27)")
    fig.suptitle(r"MNN-Pocket-0.3B: $\lambda_{max}$ vs MP bounds per block", fontsize=11)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper left", ncol=3, fontsize=8.5)
    fig.savefig(os.path.join(out_dir, "fig1_spectra_per_layer.png"), dpi=150)
    plt.close(fig)

    # Fig. 2: eigenvalue histogram vs MP density (long text, q < 1)
    T = results["runs"]["long_en"]["n_tokens"]
    H = probe.hidden_dim
    q = H / T
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True)
    for ax, li in zip(axes.flat, eig_layers):
        eigs = np.array(results["runs"]["long_en"]["eigs"][str(li)])
        sigma2 = float(np.mean(eigs))
        lam_m, lam_p = _mp_bounds_general(q, sigma2)
        ax.hist(eigs, bins=40, density=True, alpha=0.55, color="#1f77b4",
                label="eigenvalues")
        grid = np.linspace(max(lam_m, 1e-4), lam_p * 1.15, 400)
        ax.plot(grid, mp_density(grid, q, sigma2), lw=2, color="#d62728", label="MP density")
        ax.axvline(eigs[0], color="#2ca02c", lw=1.5, ls="--",
                   label=r"$\lambda_{max}$")
        ax.set_title(f"block {li+1} (of 28)", fontsize=10)
        ax.set_xlabel(r"$\lambda$")
        ax.grid(alpha=0.3)
        ax.text(0.98, 0.95, f"outliers: {int((eigs > lam_p * 1.05).sum())}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.suptitle(f"Long text (T={T}, H={H}, q={q:.2f}): spectrum vs the MP law",
                 fontsize=11)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper left", ncol=3, fontsize=9)
    fig.savefig(os.path.join(out_dir, "fig2_mp_fit.png"), dpi=150)
    plt.close(fig)

    # Fig. 3: bbp_theta across layers
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(names + ["long_en"]):
        layers = results["runs"][name]["layers"]
        li = [d["layer"] for d in layers]
        theta = [d["bbp_theta"] for d in layers]
        ax.plot(li, theta, marker="o", ms=3, lw=1.1, color=cmap(i), label=name)
    # theta_c = sqrt(q) is only well-defined for q <= 1 -> long_en only
    layers = results["runs"]["long_en"]["layers"]
    qs = [d["q"] for d in layers]
    if qs[0] <= 1.0:
        theta_c = float(np.sqrt(qs[0]))
        ax.axhline(theta_c, color="k", ls="--", lw=1.2)
        ax.text(27.6, theta_c, r" $\theta_c=\sqrt{q}$", va="center", fontsize=9)
    ax.set_xlabel("block index (0-27)")
    ax.set_ylabel(r"$\theta = \lambda_{max}/\sigma^2 - 1$")
    ax.set_yscale("log")
    ax.set_title("BBP signal parameter across layers (long text: q<1, short prompts: q>1)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper right", ncol=5, fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out_dir, "fig3_bbp_theta.png"), dpi=150)
    plt.close(fig)

    # Fig. 4: number of outliers above lambda_+
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    for i, name in enumerate(names + ["long_en"]):
        layers = results["runs"][name]["layers"]
        li = [d["layer"] for d in layers]
        n_out = [d["n_outlier_eigvals"] for d in layers]
        ax.plot(li, n_out, marker="o", ms=3, lw=1.1, color=cmap(i), label=name)
    ax.set_xlabel("block index (0-27)")
    ax.set_ylabel("number of eigenvalues above $\\lambda_+$")
    ax.set_title("Outliers above the upper MP bound (low-rank structure of the activations)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper right", ncol=5, fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out_dir, "fig4_outliers.png"), dpi=150)
    plt.close(fig)

    results["elapsed_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"done in {results['elapsed_s']}s -> {out_dir}")


if __name__ == "__main__":
    main()
