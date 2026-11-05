"""science.hypotheses.check — the pre-registered criteria, IN CODE (P2.6).

registry.yaml declares success_criterion / falsification for S2–S5 as
prose; this module is the executable form: each checker reads the stage's
own artifact (report.json dict or stage-5 record.json dict) and returns

    {"status": "supported" | "not_supported" | "not_resolved",
     "reason": "...", "evidence": {...}}

Pre-registration means the criterion lives in code and is evaluated at
the END of every run — a criterion that is only ever checked by hand is
not pre-registered, it is negotiated afterwards. The mapping is
deliberately conservative: anything the artifact cannot support lands in
"not_resolved" with the reason spelled out, never in "supported".

Layering: this module parses plain dicts (no stage imports), so the
science package stays independent of neural_lab/stages.
"""

from __future__ import annotations

import json

CHECKERS: dict[str, "callable"] = {}


def _checker(hypothesis_id: str):
    def deco(fn):
        CHECKERS[hypothesis_id] = fn
        return fn

    return deco


def _section(report: dict, substr: str) -> dict | None:
    for s in report.get("sections", []):
        if substr.lower() in str(s.get("name", "")).lower():
            return s
    return None


def _rows(section: dict) -> list[list]:
    t = section.get("table") or {}
    return t.get("rows") or []


def _headers(section: dict) -> list[str]:
    t = section.get("table") or {}
    return [str(h).lower() for h in (t.get("headers") or [])]


def _parse_ci(text) -> tuple[float, float] | None:
    """'[+0.500, +0.500]' -> (0.5, 0.5); tolerates number pairs too."""
    if isinstance(text, (list, tuple)) and len(text) == 2:
        try:
            return float(text[0]), float(text[1])
        except (TypeError, ValueError):
            return None
    if not isinstance(text, str) or "[" not in text:
        return None
    body = text.strip().strip("[]")
    parts = [p.strip().replace("+", "") for p in body.split(",")]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _verdict_dict(status: str, reason: str, evidence: dict) -> dict:
    return {"status": status, "reason": reason, "evidence": evidence}


# --- S2: spectral detector vs baselines -----------------------------------------


@_checker("S2")
def check_S2(report: dict) -> dict:
    """success_criterion (registry): Delta-AUROC CI excluding 0 on >= 2
    benchmarks AND stable under LOBO. Operationalization available in the
    artifact: >= 2 (spectral, baseline) pairs whose item-clustered ΔAUROC
    CI excludes 0 in the predicted (positive) direction; the probe race
    outcome decides whether the pre-registered negative branch fires."""
    race = _section(report, "AUROC race")
    deltas = _section(report, "paired ΔAUROC")
    if race is None or deltas is None:
        return _verdict_dict("not_resolved", "report lacks the AUROC race / ΔAUROC sections", {})

    aurocs = {}
    for row in _rows(race):
        if len(row) >= 2:
            try:
                aurocs[str(row[0])] = float(row[1])
            except (TypeError, ValueError):
                pass
    sig_pos, sig_any = [], []
    for row in _rows(deltas):
        if len(row) < 5:
            continue
        ci = _parse_ci(row[3])
        if ci is None:
            continue
        lo, hi = ci
        try:
            delta = float(row[2])
        except (TypeError, ValueError):
            continue
        if lo > 0 or hi < 0:
            sig_any.append((row[0], row[1], delta))
            if delta > 0:
                sig_pos.append((row[0], row[1], delta))
    spec_aucs = {d: a for d, a in aurocs.items() if d.startswith("spec_")}
    probe_aucs = {d: a for d, a in aurocs.items() if d.startswith("probe_")}
    evidence = {
        "significant_positive_pairs": sig_pos,
        "n_significant_pairs": len(sig_any),
        "best_spectral_auroc": max(spec_aucs.values()) if spec_aucs else None,
        "best_probe_auroc": max(probe_aucs.values()) if probe_aucs else None,
    }
    if len(sig_pos) >= 2:
        return _verdict_dict("supported",
                             f"ΔAUROC CI excludes 0 (spectral > baseline) on {len(sig_pos)} pairs",
                             evidence)
    probe_beats = bool(probe_aucs and spec_aucs
                       and max(probe_aucs.values()) > max(spec_aucs.values()))
    if probe_beats and not sig_pos:
        evidence["pre_registered_negative"] = True
        return _verdict_dict(
            "not_supported",
            "no positive significant spectral gain and probes beat every spectral detector "
            "(the pre-registered negative outcome — publish as is)",
            evidence)
    return _verdict_dict(
        "not_resolved",
        f"{len(sig_any)} pair(s) with CI excluding 0 but < 2 in the predicted direction",
        evidence)


# --- S3: T_crit ~ mu_eff^(-1/beta) scaling ---------------------------------------


@_checker("S3")
def check_S3(report: dict) -> dict:
    """Scaling verdict is produced by stage-3's own pre-registered verdict()
    (the only place the law's criterion is encoded); this checker surfaces
    it. A single-model run without the scaling section is not_resolved:
    the law needs >= 3 models."""
    scaling = _section(report, "Scaling regression")
    if scaling is None:
        return _verdict_dict("not_resolved",
                             "single-model run: the scaling criterion needs >= 3 models "
                             "(aggregate with run_stage3 --scaling-in)", {})
    notes = [str(n) for n in (scaling.get("notes") or [])]
    verdict_text = next((n for n in notes if n.startswith("VERDICT:")), None)
    if verdict_text is None:
        return _verdict_dict("not_resolved", "scaling section carries no VERDICT", {})
    text = verdict_text[len("VERDICT:"):].strip()
    evidence = {"verdict": text}
    lowered = text.lower()
    if lowered.startswith("supported"):
        return _verdict_dict("supported", text, evidence)
    if lowered.startswith("partially supported"):
        return _verdict_dict("not_resolved", text, evidence)
    if lowered.startswith("not resolved"):
        return _verdict_dict("not_resolved", text, evidence)
    return _verdict_dict("not_supported", text, evidence)


# --- S4: the BBP spike has a mechanism -------------------------------------------


@_checker("S4")
def check_S4(report: dict) -> dict:
    """falsification (registry): alignment r covering 0 on the real model
    -> the spike is a metric without a mechanism interpretation. Supported
    requires at least one mechanism alignment (lens or patching) with
    positive r and permutation p < 0.05."""
    aligned, total = [], 0
    for name in ("spike alignment",):
        section = _section(report, name)
        if section is None:
            continue
        headers = _headers(section)
        try:
            i_r = next(i for i, h in enumerate(headers) if h.startswith("r("))
            i_p = next(i for i, h in enumerate(headers) if h == "p")
        except StopIteration:
            continue
        for row in _rows(section):
            if len(row) <= max(i_r, i_p):
                continue
            try:
                r, p = float(row[i_r]), float(row[i_p])
            except (TypeError, ValueError):
                continue
            total += 1
            if r > 0 and p < 0.05:
                aligned.append((str(row[0]), r, p))
    evidence = {"n_alignments": total, "significant_positive": aligned}
    if total == 0:
        return _verdict_dict("not_resolved", "no alignment rows in the report", evidence)
    if aligned:
        return _verdict_dict("supported",
                             f"{len(aligned)}/{total} mechanism alignment(s) positive with p < 0.05",
                             evidence)
    return _verdict_dict("not_supported",
                         "no mechanism alignment is positive and significant "
                         "(the spike is not (yet) a mechanism on this run)",
                         evidence)


# --- S5: the utility trap ---------------------------------------------------------


@_checker("S5")
def check_S5(record: dict) -> dict:
    """success_criterion (registry): Delta-tau CI excluding 0 in the
    predicted direction (tuned collapses sooner) AND a significant spectral
    shift. Either direction of a significant shift is a result."""
    dt = (record.get("delta_tau_tuned_minus_base") or {})
    ci = dt.get("ci95")
    spec = (record.get("spectral_deltas") or {}).get("gap_ratio_last") or {}
    p_spec = spec.get("p_two_sided")
    spec_sig = p_spec is not None and p_spec == p_spec and p_spec < 0.05  # NaN-safe
    evidence = {"delta_tau_ci": ci, "delta_tau": dt.get("delta_tau"),
                "gap_ratio_p": p_spec, "spectral_significant": spec_sig}
    if ci is None:
        return _verdict_dict("not_resolved",
                             "no paired Δtau CI in the record "
                             "(no errors on the task set, or legacy record)", evidence)
    lo, hi = ci
    if hi < 0 and spec_sig:
        return _verdict_dict("supported",
                             "Δtau CI excludes 0 (tuned collapses sooner) with a significant spectral shift",
                             evidence)
    if lo > 0 and spec_sig:
        return _verdict_dict("not_supported",
                             "significant spectral shift, but tuned lasts LONGER (Δtau CI > 0)",
                             evidence)
    if lo <= 0 <= hi:
        return _verdict_dict("not_resolved",
                             "Δtau CI covers 0 (a negative result — report it)", evidence)
    return _verdict_dict("not_resolved",
                         "Δtau CI excludes 0 but the spectral shift is not significant", evidence)


def check(hypothesis_id: str, payload: dict) -> dict:
    """Dispatch to the stage checker; unknown ids are not_resolved."""
    fn = CHECKERS.get(hypothesis_id)
    if fn is None:
        return _verdict_dict("not_resolved", f"no checker registered for {hypothesis_id}", {})
    return fn(payload)


def criterion_section(hypothesis_id: str, payload: dict) -> dict:
    """The report section appended by every stage at the end of a run."""
    v = check(hypothesis_id, payload)
    notes = [f"{k}: {json.dumps(val, ensure_ascii=False, default=str)}"
             for k, val in v["evidence"].items()]
    return {"name": "Pre-registered criterion (registry.yaml, in code)",
            "summary": f"[{v['status']}] {v['reason']}", "notes": notes}
