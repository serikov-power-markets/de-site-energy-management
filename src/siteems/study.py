"""Safe learning study: compares grid search, GP-UCB, SafeOpt and SafeOpt with a time kernel on results/floor_by_month.csv.

    python -m siteems.sweep       # once, builds the table (about 10 minutes)
    python -m siteems.study       # seconds
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

from . import run, safeopt

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHODS = ["grid", "bo", "safeopt", "safeopt_win", "safeopt_st"]
LABEL = {"grid": "grid search", "bo": "GP-UCB (no safety)", "safeopt": "SafeOpt, pooled", "safeopt_win": "SafeOpt, 3-month window",
         "safeopt_st": "SafeOpt, space-time kernel"}
NOISE = 10.0     # EUR of observation noise on a monthly saving (a real month is also affected by forecast errors)
SEEDS = range(20)


def load_table():
    tab = pd.read_csv(run.RES / "floor_by_month.csv", index_col=0)
    return safeopt.savings_table(tab), list(tab.columns)


def evaluate(x, S, budget, noise=NOISE, **kw):
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            r = safeopt.run(x, S, method, budget=budget, noise=noise, seed=seed, **kw)
            rows.append({"method": method, "budget": budget, "seed": seed, "regret": r.regret, "unsafe": r.unsafe, "worse": r.worse,
                         "n_exp": r.n_exp, "evals_to_95pct": r.evals_to_5pct})
    return pd.DataFrame(rows)


def main():
    (x, S), months = load_table()
    ref = safeopt.reference(x, S)
    out = [evaluate(x, S, b) for b in (1, 2, 4)]
    res = pd.concat(out)
    res.to_csv(run.RES / "safeopt_runs.csv", index=False, float_format="%.2f")
    summ = res.groupby(["budget", "method"]).agg(regret=("regret", "mean"), regret_sd=("regret", "std"), unsafe=("unsafe", "mean"), worse=("worse", "mean"),
                                                 n_exp=("n_exp", "mean"), evals_to_95pct=("evals_to_95pct", "mean")).reset_index()
    summ.to_csv(run.RES / "safeopt_summary.csv", index=False, float_format="%.2f")
    sens = []                                           # sensitivity of the space-time kernel to its length scales and noise
    for ls_m in (1.0, 2.0, 4.0, 8.0):
        for ls_x in (30.0, 60.0, 90.0):
            r = evaluate(x, S, 2, ls_m=ls_m, ls_x=ls_x)
            r = r[r["method"] == "safeopt_st"]
            sens.append({"ls_m": ls_m, "ls_x": ls_x, "regret": r["regret"].mean(), "unsafe": r["unsafe"].mean()})
    pd.DataFrame(sens).to_csv(run.RES / "safeopt_sensitivity.csv", index=False, float_format="%.2f")
    st = []                                             # sensitivity to the prior amplitude and the tolerated loss
    for tol in (25.0, 50.0, 100.0):
        for sg in (50.0, 100.0, 200.0, 400.0):
            r = evaluate(x, S, 2, tol=tol, sigma=sg)
            for mth in ("safeopt", "safeopt_st"):
                q = r[r["method"] == mth]
                st.append({"tol": tol, "sigma": sg, "method": mth, "regret": q["regret"].mean(), "unsafe": q["unsafe"].mean(), "worse": q["worse"].mean()})
    pd.DataFrame(st).to_csv(run.RES / "safeopt_tol_sigma.csv", index=False, float_format="%.2f")
    noise = []
    for nz in (0.0, 10.0, 25.0, 50.0):
        r = evaluate(x, S, 2, noise=nz).groupby("method").agg(regret=("regret", "mean"), unsafe=("unsafe", "mean")).reset_index()
        r["noise"] = nz
        noise.append(r)
    pd.concat(noise).to_csv(run.RES / "safeopt_noise.csv", index=False, float_format="%.2f")
    info = {"months": months, "floors_kw": [float(v) for v in x], "reference_eur_per_year": ref,
            "best_floor_per_month_kw": [float(x[int(np.argmax(S[:, m]))]) for m in range(S.shape[1])],
            "best_saving_per_month_eur": [float(S[:, m].max()) for m in range(S.shape[1])],
            "negative_cells": int((S[1:] < 0).sum()), "cells": int(S[1:].size)}
    (run.RES / "safeopt_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    figures(x, S, months, res, summ)
    print(summ.round(1).to_string())
    print(json.dumps(info, indent=1)[:1500])
    return summ, info


def figures(x, S, months, res, summ):
    FIG = run.FIG
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    im = ax[0].imshow(S, aspect="auto", origin="lower", cmap="RdYlGn", vmin=-np.abs(S).max(), vmax=np.abs(S).max())
    ax[0].set_yticks(range(len(x)), [("none" if v == safeopt.SEED_X else f"{v:.0f}") for v in x], fontsize=7)
    ax[0].set_xticks(range(len(months)), [m[2:] for m in months], rotation=60, fontsize=7)
    ax[0].set_ylabel("peak floor (kW)"); ax[0].set_title("Saving of a month against the plain MPC (EUR)")
    fig.colorbar(im, ax=ax[0])
    for b, ls in zip((1, 2, 4), ("-", "--", ":")):
        s = summ[summ["budget"] == b].set_index("method").loc[METHODS]
        ax[1].scatter(s["unsafe"], s["regret"], s=30)
        for m, r in s.iterrows():
            ax[1].annotate(f"{LABEL[m]} ({b}/month)", (r["unsafe"], r["regret"]), fontsize=6)
    ax[1].set_xlabel("unsafe experiments per year (more than 50 EUR below the plain MPC)"); ax[1].set_ylabel("regret per year (EUR)")
    ax[1].set_title("Regret and unsafe experiments, 20 noise seeds")
    fig.tight_layout(); fig.savefig(FIG / "safeopt_overview.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
