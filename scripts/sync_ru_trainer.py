"""Sync EN v2 tiny_gpt_trainer.py to RU with Russian docstrings/strings.

Strategy: copy EN file, apply targeted replacements.

Usage:
    cd /path/to/rmt-llm-research
    python scripts/sync_ru_trainer.py
"""
import os, re

# Auto-detect repo root: this script lives in <repo>/scripts/
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EN_PATH = os.path.join(REPO, "laboratory/python/lab_en/tiny_gpt_trainer.py")
RU_PATH = os.path.join(REPO, "laboratory/python/lab_ru/tiny_gpt_trainer.py")

with open(EN_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# 1) Module docstring: replace English with Russian
ru_docstring = '''"""
tiny_gpt_trainer.py — Конвейер обучения TinyGPT на реальном корпусе (v2)
========================================================================

Обучает синтетическую модель TinyGPT (чистый NumPy, теперь с блоками
**pre-LayerNorm + MLP**) на реальном текстовом корпусе, собранном из
документации и исходного кода проекта, чтобы модель выдавала неслучайные
результаты и метрика `match_rate` лаборатории стала ненулевой.

Что делает модуль (v2 — крупное обновление)
--------------------------------------------
1. **BPE-токенайзер** — обучает byte-pair-encoding токенайзер
   (vocab=512 по умолчанию: 256 байтовых токенов + 256 выученных слияний)
   на корпусе перед обучением. ID токенов больше не являются сырыми
   байтами, поэтому модель видит меньше позиций на документ и учит
   более дальние зависимости. Слияния сохраняются вместе с весами как
   `tiny_gpt_bpe.json` и загружаются в `generate_sample`.
2. **Pre-LN + MLP трансформер** — каждый слой теперь выполняет
   ``x = x + attn(LN1(x))`` затем ``x = x + mlp(LN2(x))`` с 4× GELU MLP.
   Полный обратный режим автодиффа через LN, GELU, MLP реализован здесь
   в `forward_with_cache` / `backward`.
3. **Косинусное расписание LR с warmup** —
   ``lr = max_lr * 0.5 * (1 + cos(pi * (t - warmup) / (T - warmup)))``
   после линейного warmup в течение `warmup_epochs` эпох. Установите
   ``lr_schedule='constant'`` в TrainConfig для отключения.
4. **Расширенный корпус** — `DEFAULT_CORPUS_PATHS` теперь включает README,
   CHANGELOG, docs, webapp React/JS, все языковые лаборатории (Julia /
   Java / Rust / Go / C++ / R) и исходную библиотеку `src/rmt_llm/*.py`
   — ~1 МБ разнообразного структурированного текста.
5. **Оптимизатор Adam** — с коррекцией смещения, weight decay и
   пошаговым LR (управляется планировщиком).
6. **Чекпойнты** — сохраняет обученные веса в
   ``results/models/tiny_gpt_trained.npz`` и BPE-слияния в
   ``results/models/tiny_gpt_bpe.json``, так что
   ``TinyGPT.load_weights`` + ``BPETokenizer.load`` восстанавливают
   модель полностью.
7. **Диагностика** — потери по эпохам, норма градиента, текущий LR и
   финальная оценка `match_rate` (жадная точность следующего токена на
   отложенных окнах).

Все соглашения о бесконечных параметрах из `parameters.py` соблюдены:
``epochs='inf'`` ограничивается `max_finite_epochs` при вычислении.

Автор: Исхак Хамзатович Исаев
ORCID: 0009-0003-7299-0701
Лицензия: Проприетарная — Все права защищены.
"""'''

# Replace the module docstring (first triple-quoted block)
content = re.sub(
    r'^""".*?"""',
    ru_docstring,
    content,
    count=1,
    flags=re.DOTALL,
)

# 2) User-facing print strings -> Russian
replacements = [
    # BPE training prints
    ('f"  training BPE tokenizer ({bpe_target} merges, corpus={len(corpus):,}B)..."',
     'f"  обучение BPE-токенайзера ({bpe_target} слияний, корпус={len(corpus):,}Б)..."'),
    ('f"  BPE trained in {time.time() - t_bpe0:.1f}s, merges={tokenizer.n_merges}"',
     'f"  BPE обучен за {time.time() - t_bpe0:.1f}с, слияний={tokenizer.n_merges}"'),
    ('f"  corpus tokenized: {len(all_ids):,} BPE tokens"',
     'f"  корпус токенизирован: {len(all_ids):,} BPE-токенов"'),
    ('f"  subsampled to {len(all_ids):,} tokens (max_train_tokens cap)"',
     'f"  подвыборка до {len(all_ids):,} токенов (ограничение max_train_tokens)"'),
    ('f"  model params: {cfg.params_count:,}"',
     'f"  параметров модели: {cfg.params_count:,}"'),
    ('f"  baseline match_rate={baseline[\'match_rate\']:.3%} loss={baseline[\'loss\']:.4f}"',
     'f"  baseline match_rate={baseline[\'match_rate\']:.3%} loss={baseline[\'loss\']:.4f}"'),
    ('f"  [checkpoint] saved weights to {weights_path} after epoch {epoch+1}"',
     'f"  [чекпойнт] веса сохранены в {weights_path} после эпохи {epoch+1}"'),
    ('f"  [checkpoint ERROR] {e}"',
     'f"  [ОШИБКА чекпойнта] {e}"'),
    # CLI
    ('description="Train TinyGPT on the project corpus (v2)"',
     'description="Обучение TinyGPT на корпусе проекта (v2)"'),
    ('help="Path to rmt-llm-research/ root"',
     'help="Путь к корню rmt-llm-research/"'),
    ('help="Number of epochs (supports \'inf\')"',
     'help="Число эпох (поддерживает \'inf\')"'),
    # CLI prints
    ('f"Building corpus from {args.repo_root}..."',
     'f"Сборка корпуса из {args.repo_root}..."'),
    ('f"  corpus size: {len(corpus):,} bytes"',
     'f"  размер корпуса: {len(corpus):,} байт"'),
    ('f"\\nTraining TinyGPT v2 for {cfg.epochs} epochs "',
     'f"\\nОбучение TinyGPT v2 в течение {cfg.epochs} эпох "'),
    ('f"(BPE vocab={cfg.vocab_size}, layers={cfg.n_layers}, hidden={cfg.hidden_dim}, "',
     'f"(BPE vocab={cfg.vocab_size}, слоёв={cfg.n_layers}, hidden={cfg.hidden_dim}, "'),
    ('f"MLP={cfg.use_mlp}, LN={cfg.use_layernorm}, schedule={cfg.lr_schedule})..."',
     'f"MLP={cfg.use_mlp}, LN={cfg.use_layernorm}, расписание={cfg.lr_schedule})..."'),
    ('f"\\n=== Training complete ({result[\'elapsed_seconds\']:.1f}s) ==="',
     'f"\\n=== Обучение завершено ({result[\'elapsed_seconds\']:.1f}с) ==="'),
    ('f"  Params             : {result[\'params_count\']:,}"',
     'f"  Параметров         : {result[\'params_count\']:,}"'),
    ('f"  BPE merges         : {result[\'n_merges\']}"',
     'f"  BPE-слияний        : {result[\'n_merges\']}"'),
    ('f"  Corpus size        : {result[\'corpus_bytes\']:,} bytes"',
     'f"  Размер корпуса     : {result[\'corpus_bytes\']:,} байт"'),
    ('f"  Train windows      : {result[\'n_train_windows\']:,}"',
     'f"  Обучающих окон     : {result[\'n_train_windows\']:,}"'),
    ('f"  Eval windows       : {result[\'n_eval_windows\']:,}"',
     'f"  Оценочных окон     : {result[\'n_eval_windows\']:,}"'),
    ('f"  Baseline match_rate: {result[\'baseline_match_rate\']:.3%} (untrained)"',
     'f"  Baseline match_rate: {result[\'baseline_match_rate\']:.3%} (необученная)"'),
    ('f"  Final    match_rate: {result[\'final_match_rate\']:.3%}"',
     'f"  Финальн. match_rate: {result[\'final_match_rate\']:.3%}"'),
    ('f"  Baseline loss      : {result[\'baseline_loss\']:.4f}"',
     'f"  Baseline loss      : {result[\'baseline_loss\']:.4f}"'),
    ('f"  Final    loss      : {result[\'final_loss\']:.4f}"',
     'f"  Финальн. loss      : {result[\'final_loss\']:.4f}"'),
    ('f"  Weights            : {result[\'weights_path\']}"',
     'f"  Веса               : {result[\'weights_path\']}"'),
    ('f"  BPE merges         : {result[\'bpe_path\']}"',
     'f"  BPE-слияния        : {result[\'bpe_path\']}"'),
    ('f"\\nGenerating sample from prompt: {args.prompt!r}"',
     'f"\\nГенерация образца из промпта: {args.prompt!r}"'),
    ('f"  output: {sample!r}"',
     'f"  вывод: {sample!r}"'),
    # argparse help strings
    ('default="The RMT-LLM"', 'default="RMT-LLM "'),
    # Comments
    ('# Corpus assembly — expanded for v2',
     '# Сборка корпуса — расширено для v2'),
    ('# BPE tokenizer (pure-Python, deterministic)',
     '# BPE-токенайзер (чистый Python, детерминированный)'),
    ('# Forward pass with cache (for backprop)',
     '# Прямой проход с кэшем (для backprop)'),
    ('# Backward pass (with MLP + LayerNorm + GELU backprop)',
     '# Обратный проход (с backprop через MLP + LayerNorm + GELU)'),
    ('# Adam optimizer with cosine LR schedule',
     '# Оптимизатор Adam с косинусным расписанием LR'),
    ('# Training loop',
     '# Цикл обучения'),
    ('# Sample generation after training (uses BPE)',
     '# Генерация образцов после обучения (использует BPE)'),
    ('# CLI entrypoint',
     '# Точка входа CLI'),
    ('# Helpers',
     '# Вспомогательные функции'),
    ('# Byte-level tokenizer (kept for backward compat; BPE lives in trainer)',
     '# Байтовый токенайзер (оставлен для совместимости; BPE в тренере)'),
    ('# Quick demo',
     '# Быстрый демо-пример'),
    ('# Train',
     '# Обучение'),
    ('# Encode / decode',
     '# Кодирование / декодирование'),
    ('# Persistence',
     '# Сохранение/загрузка'),
    ('# LR scheduling',
     '# Расписание LR'),
    ('# Adam step',
     '# Шаг Adam'),
]

for en, ru in replacements:
    if en in content:
        content = content.replace(en, ru)

# 3) Update output_dir path resolution: lab_en -> lab_ru
content = content.replace(
    'out_dir = os.path.join(repo_root, "laboratory/python/lab_en", out_dir)',
    'out_dir = os.path.join(repo_root, "laboratory/python/lab_ru", out_dir)',
)

with open(RU_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Wrote {RU_PATH} ({len(content):,} chars, {content.count(chr(10))} lines)")

# Verify it imports cleanly
import subprocess
result = subprocess.run(
    ["python", "-c", "import sys; sys.path.insert(0, '/home/z/my-project/rmt-llm-research/laboratory/python/lab_ru'); import tiny_gpt_trainer; print('OK, TrainConfig fields:', list(tiny_gpt_trainer.TrainConfig().__dict__.keys())[:8])"],
    capture_output=True, text=True, cwd="/home/z/my-project/rmt-llm-research/laboratory/python/lab_ru"
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr[:500] if result.stderr else "(none)")
