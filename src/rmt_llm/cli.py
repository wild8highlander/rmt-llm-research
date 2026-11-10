"""Command-line interface for rmt_llm.

Six research commands wired to the canonical NumPy implementation, plus
``verify`` — a self-check that exercises every module and is used by CI
(the ``cli-doctor`` workflow) to detect a broken CLI before users do.

Usage examples::

    rmt-llm-verify info
    rmt-llm-verify verify
    rmt-llm-verify mp --q 0.1 --sigma2 1.0
    rmt-llm-verify bbp --theta 2.0 --q 0.1
    rmt-llm-verify tw --s -1.5
    rmt-llm-verify nhse --n-ratio 0.5 --gamma 3.0
    rmt-llm-verify spacing --n 256 --beta 2
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from rmt_llm import __author__, __orcid__, __version__
from rmt_llm.bbp_transition import (
    bbp_critical_theta,
    bbp_is_supercritical,
    bbp_lambda_max,
)
from rmt_llm.circular_ensembles import (
    circular_eigenvalues,
    nearest_neighbor_spacing,
    wigner_surmise_circular,
)
from rmt_llm.marchenko_pastur import mp_bounds, mp_density
from rmt_llm.nhse import (
    nhse_eigenvalues,
    nhse_imag_collapse,
    nhse_skin_strength,
    nhse_winding_number,
)
from rmt_llm.tracy_widom import (
    tracy_widom_cdf,
    tracy_widom_mean,
    tracy_widom_variance,
)


MODULES = (
    "marchenko_pastur", "bbp_transition", "tracy_widom", "nhse",
    "circular_ensembles", "dyson_brownian", "free_probability",
    "caputo_fractional", "ep_surfaces", "keating_snaith",
    "thermodynamics", "constants",
)


def cmd_info(_args: argparse.Namespace) -> int:
    print(f"rmt_llm {__version__}")
    print(f"author: {__author__} (ORCID {__orcid__})")
    print("modules:")
    for m in MODULES:
        print(f"  - {m}")
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    """Smoke-check every module: each command below must return finite values."""
    checks: list[tuple[str, bool, str]] = []

    lo, hi = mp_bounds(0.1, 1.0)
    rho = mp_density(np.linspace(lo, hi, 64), 0.1, 1.0)
    checks.append(("marchenko_pastur: density finite",
                   bool(np.all(np.isfinite(rho))), f"[{lo:.4f}, {hi:.4f}]"))

    theta_c = bbp_critical_theta(0.1)
    lmax = bbp_lambda_max(2.0, 0.1)
    checks.append(("bbp_transition: theta_c, lambda_outlier",
                   bool(np.isfinite(theta_c) and np.isfinite(lmax)),
                   f"theta_c={theta_c:.4f} lambda_max={lmax:.4f} "
                   f"supercritical={bbp_is_supercritical(2.0, 0.1)}"))

    m, v = tracy_widom_mean(), tracy_widom_variance()
    cdf = tracy_widom_cdf(np.array([-1.5, 0.0, 1.5]))
    checks.append(("tracy_widom: moments + cdf monotone",
                   bool(np.isfinite(m) and np.all(np.diff(cdf) >= -1e-12)),
                   f"mean={m:.4f} var={v:.4f}"))

    w = nhse_winding_number(0.5, 3.0)
    s = nhse_skin_strength(0.5, 3.0)
    c = nhse_imag_collapse(0.5, 3.0)
    checks.append(("nhse: winding, skin, collapse",
                   all(isinstance(x, (int, float)) and np.isfinite(x)
                       for x in (w, s, c)),
                   f"wind={w} skin={s:.4f} collapse={c:.4f}"))

    ev = circular_eigenvalues(128, rng=np.random.default_rng(0))
    sp = nearest_neighbor_spacing(ev)
    wign = wigner_surmise_circular(sp)
    checks.append(("circular_ensembles: eigenvalues + spacing",
                   bool(np.isfinite(sp).all() and np.isfinite(wign).all()),
                   f"n={len(ev)} mean_spacing={sp.mean():.4f}"))

    ok = all(c[1] for c in checks)
    for name, passed, detail in checks:
        print(f"  [{'OK' if passed else 'FAIL'}] {name}: {detail}")
    print(f"verify: {sum(c[1] for c in checks)}/{len(checks)} checks passed")
    return 0 if ok else 1


def cmd_mp(args: argparse.Namespace) -> int:
    lo, hi = mp_bounds(args.q, args.sigma2)
    lam = np.linspace(lo, hi, 128)
    rho = mp_density(lam, args.q, args.sigma2)
    peak = float(lam[int(np.argmax(rho))])
    out = {"q": args.q, "sigma2": args.sigma2, "support": [lo, hi],
           "density_peak": peak,
           "mean_density": float(rho.mean())}
    print(json.dumps(out, indent=2))
    return 0


def cmd_bbp(args: argparse.Namespace) -> int:
    out = {
        "theta": args.theta, "q": args.q,
        "theta_critical": bbp_critical_theta(args.q),
        "lambda_max": bbp_lambda_max(args.theta, args.q),
        "supercritical": bool(bbp_is_supercritical(args.theta, args.q)),
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_tw(args: argparse.Namespace) -> int:
    s = np.array(args.s, dtype=float)
    cdf = tracy_widom_cdf(s)
    for si, ci in zip(s, cdf):
        print(f"TW2 CDF({si:+.4f}) = {ci:.6f}")
    print(f"mean={tracy_widom_mean():.6f} variance={tracy_widom_variance():.6f}")
    return 0


def cmd_nhse(args: argparse.Namespace) -> int:
    ev = nhse_eigenvalues(args.size, args.gamma,
                          rng=np.random.default_rng(args.seed))
    out = {
        "n_ratio": args.n_ratio, "gamma": args.gamma,
        "winding": int(nhse_winding_number(args.n_ratio, args.gamma)),
        "skin_strength": float(nhse_skin_strength(args.n_ratio, args.gamma)),
        "imag_collapse": float(nhse_imag_collapse(args.n_ratio, args.gamma)),
        "eigen_re": [float(x) for x in ev.real[:args.size]],
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_spacing(args: argparse.Namespace) -> int:
    ev = circular_eigenvalues(args.n, rng=np.random.default_rng(args.seed))
    sp = nearest_neighbor_spacing(ev)
    out = {
        "n": args.n, "beta": args.beta,
        "mean_spacing": float(sp.mean()),
        "spacing_min": float(sp.min()), "spacing_max": float(sp.max()),
        "wigner_surmise_at_1": float(wigner_surmise_circular(np.array([1.0]),
                                                            args.beta)[0]),
    }
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rmt-llm-verify",
        description="RMT-LLM spectral verification CLI (NumPy core)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="package info and module list")
    sub.add_parser("verify", help="run self-checks over all modules (CI gate)")

    sp = sub.add_parser("mp", help="Marchenko-Pastur bounds and density")
    sp.add_argument("--q", type=float, default=0.1, help="aspect ratio n/p")
    sp.add_argument("--sigma2", type=float, default=1.0, help="noise variance")

    sp = sub.add_parser("bbp", help="BBP transition: outlier position")
    sp.add_argument("--theta", type=float, default=2.0,
                    help="signal strength")
    sp.add_argument("--q", type=float, default=0.1)

    sp = sub.add_parser("tw", help="Tracy-Widom (beta=2) CDF and moments")
    sp.add_argument("--s", type=float, nargs="+", default=[-1.5, 0.0, 1.5],
                    help="one or more s values")

    sp = sub.add_parser("nhse", help="non-Hermitian skin effect summary")
    sp.add_argument("--n-ratio", type=float, default=0.5)
    sp.add_argument("--gamma", type=float, default=3.0)
    sp.add_argument("--size", type=int, default=128, help="matrix size")
    sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("spacing", help="circular-ensemble spacing statistics")
    sp.add_argument("--n", type=int, default=256, help="matrix size")
    sp.add_argument("--beta", type=int, default=2, choices=(1, 2, 4))
    sp.add_argument("--seed", type=int, default=0)

    sub.choices["info"].set_defaults(func=cmd_info)
    sub.choices["verify"].set_defaults(func=cmd_verify)
    sub.choices["mp"].set_defaults(func=cmd_mp)
    sub.choices["bbp"].set_defaults(func=cmd_bbp)
    sub.choices["tw"].set_defaults(func=cmd_tw)
    sub.choices["nhse"].set_defaults(func=cmd_nhse)
    sub.choices["spacing"].set_defaults(func=cmd_spacing)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
