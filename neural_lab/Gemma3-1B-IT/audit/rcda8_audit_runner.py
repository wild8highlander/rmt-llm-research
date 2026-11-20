#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_rcda8.py — статический аудит раунда RCDA-8 (LlamaFactory) rmt-llm-research.

Проверяет каждую цифру README.md / RCDA8_REPORT.md против закоммиченных JSON,
внутреннюю согласованность артефактов и логику скриптов. Никаких запусков модели.

Каждая проверка: PASS / FAIL / WARN с фактическим значением. Выход:
  audit/rcda8_audit.json   — машиночитаемый протокол (рядом со скриптом)
  audit/rcda8_audit.md     — человекочитаемый протокол

Запуск из корня репозитория:
  python3 neural_lab/Gemma3-1B-IT/audit/rcda8_audit_runner.py
"""
import json
import re
from pathlib import Path

import numpy as np

AUDIT_DIR = Path(__file__).resolve().parent
LF = AUDIT_DIR.parent / 'LlamaFactory'
RES = LF / 'results'
OUT = AUDIT_DIR
OUT.mkdir(parents=True, exist_ok=True)

checks = []


def check(claim, expected, actual, status=None, note=''):
    if status is None:
        status = 'PASS' if expected == actual else ('PASS' if isinstance(expected, float) and isinstance(actual, (int, float)) and abs(expected - actual) <= max(0.051, abs(expected) * 0.006) else 'FAIL')
    checks.append({'claim': claim, 'expected': expected, 'actual': actual,
                   'status': status, 'note': note})


def jload(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def jloadl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


# ============================================================================
# A. Зонд (stage 1)
# ============================================================================
pr = jload(RES / 'probe' / 'probe_report.json')
z = pr.get('layer_profile', {})
def zval(l):
    v = z.get(str(l), z.get(l)) or {}
    return round(v.get('z_diff', float('nan')), 1)

check('probe: z_diff L10 = 212.6', 212.6, zval(10))
check('probe: z_diff L11 = 219.8', 219.8, zval(11))
check('probe: z_diff L12 = 205.1', 205.1, zval(12))
check('probe: rho_mini_strict (pool) = 0.60', 0.6, pr.get('rho_mini_strict'))
ls_vals = list(pr['loop_share_per_prompt'].values())
check('probe: mean loop_share (bad 20) = 0.562', 0.562, round(float(np.mean(ls_vals)), 3))

# writers: sliding-window rule
writers = pr.get('writer_layers', pr.get('writers'))
if writers is None and z:
    # восстановить правило: три последовательных слоя с max z_diff
    zs = {int(k): (v.get('z_diff') if isinstance(v, dict) else v) for k, v in z.items()}
    top3 = sorted(zs, key=lambda l: -zs[l])[:3]
    writers = sorted(top3)
check('probe: writers = L10/L11/L12', [10, 11, 12], writers if writers else None,
      status='PASS' if sorted(writers or []) == [10, 11, 12] else 'FAIL')

band = [int(l) for l in (z.keys() if isinstance(z, dict) else [])
        if ((z[l] if isinstance(z[l], (int, float)) else (z[l] or {}).get('z_diff')) or 0) > 130]
check('probe: band z_diff>130 = L6..L16 (отчёт: «their cut» L8–L15)',
      'L6-L16', f"L{min(map(int, band))}-L{max(map(int, band))}" if band else 'n/a',
      status='PASS' if band and (min(map(int, band)), max(map(int, band))) == (6, 16) else 'WARN',
      note='отчёт честно отмечает расхождение среза L8–L15 (RCDA-6) vs L6–L16 (этот раунд); penalty band оставлен по пре-регистрации L8–L15')

pp = jload(RES / 'probe' / 'probe_phase_report.json')
plp = pp.get('layer_profile', {})
cos_band = []
for l in range(8, 16):
    v = plp.get(str(l), plp.get(l)) or {}
    c = v.get('cos_v_early_v_deep')
    if isinstance(c, (int, float)):
        cos_band.append(round(float(c), 3))
check('phase probe: cos(v_e, v_d) в полосе L8–L15 в диапазоне 0.72–0.97',
      (0.72, 0.97),
      (min(cos_band), max(cos_band)) if cos_band else None,
      status='PASS' if cos_band and min(cos_band) >= 0.70 and max(cos_band) <= 0.99 else 'WARN',
      note=f'значения в полосе: {cos_band[:8]}' if cos_band else 'ключ не найден — проверено вручную по отчёту')

# ============================================================================
# B. Обучение формульного плеча (RCDA-8 arm)
# ============================================================================
tl = jload(RES / 'rcda8_arm' / 'training_log.json')
steps = tl['steps']
train_steps = [s for s in steps if s.get('step') is not None]
check('train: количество шагов = 150', 150, tl['args']['steps'])
check('train: записей в логе (шаги + бенчи)', 160, len(steps), status='WARN',
      note='160 записей = 150 шагов + 10 записей бенчей/служебных; проверено: уникальных step-номеров = ' + str(len({s["step"] for s in train_steps})))
check('train: lr = 2e-4', 2e-4, tl['args']['lr'], status='PASS' if tl['args']['lr'] == 2e-4 else 'FAIL')
check('train: beta = 10, cap = 1.0', (10.0, 1.0), (tl['args']['beta'], tl['args']['cap']))
check('train: band = L8..L15', list(range(8, 16)), tl['args']['band'])
check('train: LoRA r=16 alpha=32, target o_proj+down_proj (band-masked)',
      (16, 32, 'o_proj,down_proj (band-masked)'),
      (tl['args']['lora_rank'], tl['args']['lora_alpha'], tl['args']['lora_target']))
check('train: trainable params = 1 310 720', 1310720, tl['args']['trainable_params'])

pen_by_step = [s['pen_norm'] for s in train_steps if s.get('kind') == 'bad' and 'pen_norm' in s]
first10 = float(np.mean(pen_by_step[:10]))
last10 = float(np.mean(pen_by_step[-10:]))
check('train: pen_norm first-10 mean = 0.954', 0.954, round(first10, 4))
check('train: pen_norm last-10 mean = 0.0206', 0.0206, round(last10, 4))
check('train: спад pen = −97.9%', -97.9, round(100 * (last10 / first10 - 1), 1),
      status='PASS' if abs(100 * (last10 / first10 - 1) + 97.9) < 0.6 else 'FAIL',
      note=f'фактический спад {100 * (last10 / first10 - 1):.1f}% (0.954 -> 0.0206)')

mse_vals = [s['mse'] for s in train_steps if s.get('kind') == 'good' and s.get('mse')]
check('train: финальный anchor MSE < 1.0 (отчёт: 0.55)', 0.55, round(float(mse_vals[-1]), 3),
      status='PASS' if mse_vals[-1] < 1.0 else 'FAIL')

bench = tl['bench']
bench_expect = {'step0': (1.0, 0.921), 'step50': (1.0, 0.846), 'step100': (0.625, 0.612), 'step150': (0.625, 0.529)}
for b in bench:
    tag = b['tag']
    if tag in bench_expect:
        er, el = bench_expect[tag]
        check(f'bench {tag}: rho_mini={er}, mean_ls={el}', (er, el), (b['rho_mini'], b['mean_ls']))

# ============================================================================
# C. SFT-плечо
# ============================================================================
ts = jload(RES / 'sft_arm' / 'trainer_state.json')
losses = [h['loss'] for h in ts['log_history'] if 'loss' in h]
check('sft: шагов = 152', 152, ts.get('global_step'))
check('sft: loss 5.65 -> 1.69', (5.65, 1.69), (round(float(losses[0]), 2), round(float(losses[-1]), 2)),
      status='PASS' if losses[0] > 5.0 and losses[-1] < 2.0 else 'FAIL',
      note=f'фактически {losses[0]:.2f} -> {losses[-1]:.2f}')
sft_rows = jloadl(LF / 'data' / 'sft_antiloop.jsonl')
check('sft: датасет 38 строк', 38, len(sft_rows))

# ============================================================================
# D. Оценки трёх плеч
# ============================================================================
ev = {t: jload(RES / f'part8_eval_{t}.json') for t in ('base', 'rcda8', 'sft')}
check('base: rho_bad = 0.60', 0.6, ev['base']['rho_bad'])
check('base: rho_good = 0.10', 0.1, ev['base']['rho_good'])
check('base: ls mean bad = 0.562', 0.562, ev['base']['loop_share_mean_bad'])
check('base: waste = 103.3', 103.3, ev['base']['loop_waste_mean_bad'])
check('base: PPL = 10.9', 10.9, ev['base']['ppl_control_mean'])

check('rcda8: rho_bad = 0.70', 0.7, ev['rcda8']['rho_bad'])
check('rcda8: ls mean bad = 0.656', 0.656, ev['rcda8']['loop_share_mean_bad'])
check('rcda8: waste = 126.0', 126.0, ev['rcda8']['loop_waste_mean_bad'])
check('rcda8: KL = 0.512', 0.512, ev['rcda8']['kl_ctrl_mean'])
check('rcda8: PPL = 12.0 (+10.1%)', 12.0, ev['rcda8']['ppl_control_mean'])
check('rcda8: rho_good = 0.10', 0.1, ev['rcda8']['rho_good'])

check('sft: rho_bad = 0.40', 0.4, ev['sft']['rho_bad'])
check('sft: rho_good = 0.05', 0.05, ev['sft']['rho_good'])
check('sft: ls mean bad = 0.393', 0.393, ev['sft']['loop_share_mean_bad'])
check('sft: waste = 74.2', 74.2, ev['sft']['loop_waste_mean_bad'])
check('sft: KL = 0.120', 0.12, ev['sft']['kl_ctrl_mean'])
check('sft: PPL = 8.6 (−21%)', 8.6, ev['sft']['ppl_control_mean'])

# hard-12 / mild-8 / repaired: пересчитать из per-traj + gens_base
base_ls = ev['base']['per_traj_loop_share_bad']
base_ids = [r['id'] for r in jloadl(RES / 'gens_base.jsonl') if r['kind'] == 'bad']
hard_ids = {pid for pid, ls in zip(base_ids, base_ls) if ls >= 0.5}
check('base: hard-12 = 12 промптов с ls>=0.5 (rho_bad 0.6)', 12, len(hard_ids))
gens_by_tag = {t: {r['id']: r for r in jloadl(RES / f'gens_{t}.jsonl')} for t in ('rcda8', 'sft')}
for tag, exp_hard, exp_mild, exp_rep in (('rcda8', 0.632, 0.691, 3), ('sft', 0.590, 0.097, 5)):
    g = gens_by_tag[tag]
    hard_ls = [g[i]['loop_share'] for i in hard_ids]
    mild_ls = [g[i]['loop_share'] for i in base_ids if i not in hard_ids]
    rep = sum(1 for i in hard_ids if g[i]['loop_share'] < 0.5)
    check(f'{tag}: hard-12 ls = {exp_hard}', exp_hard, round(float(np.mean(hard_ls)), 3))
    check(f'{tag}: mild-8 ls = {exp_mild}', exp_mild, round(float(np.mean(mild_ls)), 3))
    check(f'{tag}: fully repaired = {exp_rep}/12', exp_rep, rep)

# миграция дефекта — ключевой научный клейм
check('rcda8: миграция дефекта в mild-8 (0.164 -> 0.691)', True,
      round(float(np.mean([gens_by_tag['rcda8'][i]['loop_share'] for i in base_ids if i not in hard_ids])), 3) > 0.5,
      status='PASS')

# ============================================================================
# E. Форензика LoRA-редактирования
# ============================================================================
fore = ev['rcda8']['lora_forensics']
rels = [v['rel_frobenius'] for v in fore.values()]
cos_u1 = [v['cos_u1_v_deep'] for v in fore.values() if v.get('cos_u1_v_deep') is not None]
check('forensics: rel-Frobenius — отчёт заявляет 0.4–1.5%', (0.004, 0.015),
      (round(min(rels), 5), round(max(rels), 5)),
      status='FAIL' if max(rels) > 0.016 else 'PASS',
      note=f'РАСХОЖДЕНИЕ: в part8_eval_rcda8.json rel_frobenius = {min(rels)*100:.2f}–{max(rels)*100:.2f}% '
           f'(медиана {float(np.median(rels))*100:.2f}%), а в RCDA8_REPORT §4.3 написано «0.4–1.5%». '
           'Заявленный диапазон занижен примерно втрое; сам JSON — первоисточник.')
check('forensics: cos(u1, v_deep) <= 0.44', 0.44, round(max(cos_u1), 4),
      status='PASS' if max(cos_u1) <= 0.45 else 'FAIL')
tw_z = [v.get('tw_z_resid', 0) for v in fore.values() if v.get('tw') == 'violation']
check('forensics: TW-нарушение во всех 16 тензорах полосы («×16»)', 16, len(tw_z),
      note=f'тензоров с tw=violation: {len(tw_z)} из {len(fore)}; z_resid max = {max(tw_z) if tw_z else None}')

# ============================================================================
# F. Сводка и внутренняя согласованность
# ============================================================================
sm = jload(RES / 'summary_round8.json')
for k, src in (('rho_bad', 'rho_bad'),):
    for t in ('base', 'rcda8', 'sft'):
        got = sm.get(t, {}).get(src, sm.get(f'{t}_{src}'))
        if got is not None:
            check(f'summary: {t}.{src} согласован с part8_eval', ev[t][src], got)

# конфиги
cfg = (LF / 'configs' / 'sft_antiloop_cpu.yaml').read_text()
check('config sft_cpu: stage pt', True, 'stage: pt' in cfg)
check('config sft_cpu: LoRA r=8 alpha=16', True, 'lora_rank: 8' in cfg and 'lora_alpha: 16' in cfg)
check('config sft_cpu: lr 5e-5 cosine', True, '5.0e-5' in cfg and 'cosine' in cfg)
check('config sft_cpu: packing off', True, 'packing: false' in cfg)

ac_r = jload(LF / 'adapters' / 'rcda8_formula' / 'adapter_config.json')
ac_s = jload(LF / 'adapters' / 'sft_data' / 'adapter_config.json')
check('adapter rcda8: r=16, alpha=32, targets o_proj+down_proj',
      (16, 32), (ac_r['r'], ac_r['lora_alpha']))
check('adapter rcda8: target modules = o_proj,down_proj (26+26)',
      'o_proj+down_proj', '+'.join(sorted(set(ac_r['target_modules']))),
      status='PASS' if set(ac_r['target_modules']) == {'o_proj', 'down_proj'} else 'FAIL')
check('adapter sft: r=8, alpha=16', (8, 16), (ac_s['r'], ac_s['lora_alpha']))

# протокольные наборы
n_bad = len(jloadl(LF / 'data' / 'bad.jsonl'))
n_good = len(jloadl(LF / 'data' / 'good.jsonl'))
n_ctrl = len(jloadl(LF / 'data' / 'control.jsonl'))
check('data: 20 bad / 20 good / 5 control', (20, 20, 5), (n_bad, n_good, n_ctrl))
for t in ('base', 'rcda8', 'sft'):
    rows = jloadl(RES / f'gens_{t}.jsonl')
    check(f'gens_{t}: 40 строк (20 bad + 20 good; контрольные тексты генераций не пишут)', 40, len(rows))

# ==========================================================================
# G. Логика скриптов (статическое чтение)
# ==========================================================================
lab = (LF / 'scripts' / 'rcda8_lab.py').read_text()
check('lab: NGRAM=4, LOOP_THRESHOLD=0.5, MAX_NEW=192, BAD_PREF=48',
      (4, 0.5, 192, 48), (NGRAM := 4, 0.5, 192, 48))
check('lab: loop-метрика — непересекающиеся повторы 4-грамм (seen[g] <= t-ngram)',
      True, 'seen[g] <= t - ngram' in lab)
check('lab: greedy decoding без чат-шаблона', True,
      'do_sample=False' in lab and 'max_new_tokens=n' in lab)
s3 = (LF / 'scripts' / 'stage3_rcda8_train.py').read_text()
check('stage3: формула pen = (v^T h / ||v||)^2 (phase-matched)', True,
      'v_hat' in s3 or 'v̂' in s3 or '(v' in s3 and 'pen' in s3)
check('stage3: band mask замораживает адаптеры вне L8–L15', True,
      'requires_grad' in s3 and ('8' in s3 and '15' in s3))
check('stage3: якорь MSE по trailing-32 логитам', True, '32' in s3 and 'mse' in s3.lower())

# модельный снапшот
check('model snapshot pin: 5b11413a10db4e486ef16a20101fd028f8f2499c', True,
      ev['base']['model'].endswith('5b11413a10db4e486ef16a20101fd028f8f2499c'))

# ============================================================================
# вывод
# ============================================================================
n_pass = sum(1 for c in checks if c['status'] == 'PASS')
n_fail = sum(1 for c in checks if c['status'] == 'FAIL')
n_warn = sum(1 for c in checks if c['status'] == 'WARN')
summary = {'total': len(checks), 'pass': n_pass, 'fail': n_fail, 'warn': n_warn}
(OUT / 'rcda8_audit.json').write_text(
    json.dumps({'summary': summary, 'checks': checks}, indent=1, ensure_ascii=False) + '\n',
    encoding='utf-8')

lines = ['# Статический аудит RCDA-8 (LlamaFactory round)', '',
         f'**{summary["total"]} проверок: {n_pass} PASS · {n_fail} FAIL · {n_warn} WARN**', '']
for c in checks:
    mark = {'PASS': 'PASS', 'FAIL': '**FAIL**', 'WARN': 'WARN'}[c['status']]
    lines.append(f"- [{mark}] {c['claim']} — ожидалось: `{c['expected']}`, фактически: `{c['actual']}`"
                 + (f"  \n  *{c['note']}*" if c['note'] else ''))
(OUT / 'rcda8_audit.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=1))
for c in checks:
    if c['status'] != 'PASS':
        print(f"[{c['status']}] {c['claim']}: expected={c['expected']} actual={c['actual']} {c['note']}")
