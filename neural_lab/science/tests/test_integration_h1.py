"""
Интеграционный тест: весь конвейер H1 на ФЕЙКОВОЙ модели.

Проверяет сцепку слоёв (наблюдение -> признаки -> разбиение -> фит ->
предсказания -> раскрытие -> отчёт) без реальных весов TinyGPT.
Фейковая модель детерминированно «угадывает» ответ в ~60 % задач и имеет
запланированный сигнал в скрытых состояниях (общий фактор у угадавших).
"""

import os
import zlib

import numpy as np
import pytest
from science.data import build_task_set, label_observation
from science.experiments.run_h1 import build_models, fit_predict, lofo_predictions, unblind
from science.features import build_feature_vector
from science.instrumentation.capture import capture_observation


class FakeTokenizer:
    def encode(self, text):
        return [min(ord(c), 255) % 512 for c in text[:32]]

    def decode(self, ids):
        return "".join(chr(int(i) % 256) for i in ids)


class FakeConfig:
    max_seq_len = 64


class FakeModel:
    config = FakeConfig()

    def generate(
        self,
        prompt_ids,
        max_new_tokens=8,
        temperature=0.8,  # noqa: ARG002
        seed=None,
        capture_hidden=True,  # noqa: ARG002
    ):
        import re

        import marchenko_pastur as mp

        rng = np.random.default_rng(seed if seed is not None else 0)
        prompt = "".join(chr(int(i) % 256) for i in prompt_ids)
        # «правильность» детерминирована задачей; у температурных сэмплов шум
        guess = (zlib.crc32(prompt.encode()) % 10) < 6
        if guess:
            # «знающая» модель выдаёт числа, близкие к истине (±25% метка = 1)
            m = re.search(r"q=([\d.]+), sigma2=([\d.]+)", prompt)
            q, s2 = (float(m.group(1)), float(m.group(2))) if m else (0.5, 1.0)
            lm, lp = mp.mp_bounds(q, s2)
            text = (
                f" => lambda_minus = {lm + 0.001 * rng.random():.6f}, "
                f"lambda_plus = {lp + 0.001 * rng.random():.6f}"
            )
        else:
            text = "xyz = 9.999999"
        ids = [ord(c) % 512 for c in text[:max_new_tokens]]  # decode reproduces text
        n = len(ids)
        logits = [rng.normal(0, 1, 512) for _ in range(n)]
        for step_lg in logits:
            if guess:
                step_lg[7] += 8.0  # уверенный первый токен у «знающих»
        hidden = []
        T = min(32, 8 + n)
        for _ in range(n):
            layers = []
            for _li in range(4):
                h = rng.normal(0, 0.4, (T, 24))
                if guess:
                    h += 2.0 * rng.normal(0, 1, (1, 24))  # общий фактор -> spike
                layers.append(h)
            hidden.append(layers)
        return {
            "output_ids": ids,
            "full_ids": list(prompt_ids) + ids,
            "per_step_logits": logits,
            "hidden_snapshots": hidden,
        }


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("h1_it"))
    # --- задачи из мини-корпуса (реальная верификация библиотекой) ---
    import marchenko_pastur as mp

    lines = []
    for q in [0.5, 0.6, 0.7]:
        lm, lp = mp.mp_bounds(q, 1.0)
        lines.append(
            f"mp_bounds(q={q}, sigma2=1.0) => lambda_minus = {lm:.6f}, lambda_plus = {lp:.6f}"
        )
    corpus = os.path.join(out, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        f.write("=== EVAL PAIRS ===\n" + "\n".join(lines))
    tok = FakeTokenizer()
    tasks, _exclusions, _stats = build_task_set(
        corpus, max_prompt_tokens=32, encode_fn=tok.encode, n_indist=3, n_near=2
    )
    assert len(tasks) >= 10
    ctx = {"model": FakeModel(), "tokenizer": tok}

    # --- наблюдения + признаки (FakeModel: 1 «токен» = 1 символ) ---
    observations, labels = [], []
    for task in tasks:
        for i, temp in enumerate([0.8, 0.8, 0.0]):
            seed = None if temp == 0.0 else (zlib.crc32(task.task_id.encode()) * 10 + i)
            obs = capture_observation(
                ctx["model"], tok, task.prompt, max_new_tokens=60, temperature=temp, seed=seed
            )
            feats = build_feature_vector(obs)
            observations.append(
                {
                    "task_id": task.task_id,
                    "sample_id": f"s{i}",
                    "band": task.band,
                    "family": task.family,
                    "temperature": temp,
                    "seed": seed,
                    "generated_text": obs["generated_text"],
                    "n_new_tokens": obs["n_new_tokens"],
                    "elapsed_ms": 1.0,
                    "features": feats,
                    "excluded": False,
                }
            )
            labels.append(
                {
                    "task_id": task.task_id,
                    "sample_id": f"s{i}",
                    **label_observation(
                        obs["generated_text"], task.ground_truth, [0.15, 0.25, 0.40]
                    ),
                }
            )
    from science.storage import save_jsonl

    save_jsonl(observations, os.path.join(out, "observations.jsonl"))
    save_jsonl(labels, os.path.join(out, "labels.jsonl"))
    return out, observations, labels, tasks


def test_full_pipeline(pipeline):
    out, observations, labels, tasks = pipeline
    models = build_models(4)  # FakeModel: 4 слоя скрытых состояний
    records, fold_report = fit_predict(observations, labels, tasks, out, models)
    assert len(records) == len([o for o in observations if not o.get("excluded")])
    assert all(r["fold"] >= 0 for r in records)
    for name in models:
        assert all(name in r for r in records)
    # фолды попарно не пересекаются по задачам
    by_fold = {}
    for r in records:
        by_fold.setdefault(r["fold"], set()).add(r["task_id"])
    folds_list = list(by_fold.values())
    for i in range(len(folds_list)):
        for j in range(i + 1, len(folds_list)):
            assert not folds_list[i] & folds_list[j]

    lofo = lofo_predictions(observations, labels, tasks, models)
    assert lofo
    assert all("D_combined" in v for v in lofo.values())

    report = unblind(records, fold_report, lofo, out, n_boot=150, n_perm=150, models=models)
    assert report["n_observations"] == len(records)
    y = np.array([r["y_true"] for r in records])
    assert report["n_positives"] == int(y.sum())
    assert 0 < y.sum() < len(y)
    auc = report["metrics"]["D_combined"]["roc_auc"]
    assert np.isfinite(auc["point"])
    assert auc["ci_low"] <= auc["point"] <= auc["ci_high"]
    # фейковая модель имеет запланированный сигнал -> комбинированная модель
    # должна быть осмысленно лучше случайного ранжирования
    assert report["metrics"]["D_combined"]["roc_auc"]["point"] > 0.6
    from science.experiments.run_h1 import render_report_md

    md = render_report_md(
        report,
        {
            "protocol_version": "1.0",
            "model_revision": "0" * 64,
            "dataset_hash": "1" * 64,
            "seed": 42,
        },
    )
    assert "H1" in md
    assert "ROC-AUC" in md
