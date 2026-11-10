# RCDA RUNBOOK — воспроизведение и развитие (neural_lab/MNN-Pocket-0.3B)

Набор команд от чистого клона до повторения обоих раундов исследования.
Требования: `pip install torch transformers safetensors scipy pyyaml matplotlib`
(CPU-версии достаточно; модель fp32 занимает ~1.2 ГБ RAM).

## 0. Подготовка модели

```bash
git clone https://github.com/wild8highlander/rmt-llm-research
cd rmt-llm-research/neural_lab/MNN-Pocket-0.3B
python tools/reassemble_model.py      # соберёт model/model.safetensors + SHA-256
```

## 1. Раунд 1 — RCDA по целевому плану (стилевой контраст)

```bash
# 1.1 RCDS-проба: Δh, z_RMT, β, z_diff, матричные сканы, векторы
python Code/pocket_rmt_probe.py --rcds-probe --model_dir model \
    --bad_set prompts/bad.jsonl --good_set prompts/good.jsonl \
    --out_dir rcda_results

# 1.2 Правка весов: α по BBP, колонки auto_chi2 (строгий гейт — auto_tw)
python Code/rmt_intervention.py --rcda --model_dir model \
    --layer 9  --matrix down_proj --vector rcda_results/vectors/L9_delta_h.pt \
    --layer 11 --matrix o_proj    --vector rcda_results/vectors/L11_delta_h.pt \
    --alpha auto --columns auto_chi2 --out_dir edited_model

# 1.3 Бенчмарк до/после + TW-контроль
python Code/benchmark_rmt.py --baseline_dir model --edited_dir edited_model \
    --bad_set prompts/bad.jsonl --good_set prompts/good.jsonl \
    --control_set prompts/control.jsonl --out_dir bench --max_new 64

# 1.4 Воспроизведение правки из pack (векторы зашиты в JSON)
python Code/rmt_intervention.py --rcda --model_dir model \
    --apply_pack Research/RCDA/results/rcda_edit_report_applied_autochi2.json \
    --out_dir edited_model_from_pack
```

Ожидаемый итог раунда 1: TW pass, PPL без изменений, ρ без сдвига — правка
«стилевая», что и показал раунд 2 (|cos| со поведенческим направлением ≈ 0.01–0.09).

## 2. Раунд 2 — RCDA-2: поведенческий контраст и предвестники

```bash
# 2.1 Генерации с токен-метками петель + Δh_behavioral + сканы + атрибуция + PR
python Research/RCDA/tools/rcda2_stage1_probe.py
#   -> rcda2/rcda2_probe.json, rcda2/rcda2_generations.jsonl, vectors/*_delta_h_behav.pt

# 2.2 Дозозависимость α (KL, flip-rate, rep4) + финальная правка с TW-backoff
python Research/RCDA/tools/rcda2_stage2_dose.py
#   -> edited_model_v2/model.safetensors, rcda2/rcda2_edit_report.json

# 2.3 Бенчмарк раунда 2
python Code/benchmark_rmt.py --baseline_dir model --edited_dir edited_model_v2 \
    --bad_set prompts/bad.jsonl --good_set prompts/good.jsonl \
    --control_set prompts/control.jsonl --out_dir bench_v2 --max_new 96

# 2.4 Фигуры fig7-fig10 + дайджест
python Research/RCDA/tools/rcda2_stage3_figures.py

# 2.5 Синтетическая валидация математики (25 проверок, модель не нужна)
python Research/RCDA/tools/test_rcda_math.py
```

## 3. Следующие шаги (по выводам RCDA-2)

1. **Пер-головная абляция**: колонки o_proj концентрируются в головах
   L15→h7, L16→h8, L18→h0; вычитать Δh-профиль только из 64-колоночного
   блока головы (`rcda_edit` c columns = 64h..64h+63).
2. **Итерационная петля** (§6 плана): после каждой правки перезапускать
   stage 1 на отредактированной модели и переоценивать Δh; stop — по
   `stop:`-блоку `config/rcda_experiment.yaml` (ρ_target 0.2, z_diff < 2,
   FPR ≤ 0.15, PPL ≤ 1.10×).
3. **Bulk-шринидж** (§9): `rcda_shrink` / `apply_pack_with_shrink` в
   `Code/rmt_intervention.py` — гипотеза «дефект = перегретый bulk»;
   тестировать на L15–L18 o_proj при r = 10, τ = 2.
4. **PR-детектор**: мониторинг participation ratio окна 16 на L15 (падение
   > 5% от скользящей нормы) → включение runtime-хука `clip` из
   `rmt_intervention.py` на время генерации.
5. **Контраст групп**: расширять bad/good наборы, выравнивая длину и тему,
   чтобы остаточный стиль не доминировал в Δh (см. README_RCDA.md, п. 5).

## 4. Структура артефактов

```
Research/RCDA/
  RCDA2_REPORT.md          отчёт раунда 2 (числа, таблицы, выводы)
  RCDA_ROUND1_SUMMARY.md   протокол раунда 1
  results/                 JSON: пробы, сканы, дозы, бенчмарки, edit-паки
  figures/                 fig5-fig10
  vectors/                 Δh-направления (.pt) обоих раундов
  tools/                   rcda2_stage{1,2,3}*.py, test_rcda_math.py
config/rcda_experiment.yaml    параметры раунда 1 + stop-критерии
config/rcda2_behavioral.yaml   параметры раунда 2
Docs/RCDA_RUNBOOK.md           этот файл
```
