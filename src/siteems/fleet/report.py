"""Tables and figures of v0.3 from the stored runs (results/fleet/runs)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from . import sites, tariffs
from .run import OUT, TEST

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path(__file__).resolve().parents[3] / "figures"
ORDER = ["no battery", "threshold rule", "threshold rule, Kalman, 2 sigma margin", "floor, full reserve",
         "floor, full reserve, Kalman, margin", "MPC naive forecast", "MPC", "MPC, scenarios", "MPC, scenarios, CVaR",
         "MPC, peak floor", "floor, coulomb counting", "floor, Kalman, no margin", "floor, Kalman, 2 sigma margin",
         "floor, Kalman, margin, faults", "atypical-use plan", "atypical-use plan, perfect forecast",
         "perfect forecast"]
# value path chosen before the year was run: peak floor (best of v0.1), then execution, then faults
PATH = ["perfect forecast", "MPC, peak floor", "floor, Kalman, 2 sigma margin", "floor, Kalman, margin, faults"]


def load_runs() -> pd.DataFrame:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "runs").glob("*.json"))]
    df = pd.DataFrame(rows)
    return df.drop(columns=["settings", "modes"])


def results(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes("number").columns.drop("seed")
    t = df.groupby(["site", "strategy"])[list(num)].mean()
    t["runs"] = df.groupby(["site", "strategy"]).size()
    t["tier"] = df.groupby(["site", "strategy"])["tier"].first()
    t["atypical_eligible"] = df.groupby(["site", "strategy"])["atypical_eligible"].mean()
    base = {s: t.loc[(s, "no battery"), "total_eur"] for s in df.site.unique() if (s, "no battery") in t.index}
    feed = ("B_new_roof", "feed-in case: no battery")
    if feed in t.index:
        base["B_new_roof"] = t.loc[feed, "total_eur"]
    t["saving_eur"] = [base[s] - row.total_eur for (s, _), row in t.iterrows()]
    # atypical use lowers the capacity charge for eligible profiles, whatever the plan
    t["saving_with_atypical_use_eur"] = t["saving_eur"] + t["atypical_saving_eur"]
    return t.reset_index()


def waterfall(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site in [s.name for s in sites.FLEET]:
        v = t[t.site == site].set_index("strategy")["saving_eur"]
        steps = [v[name] for name in PATH]
        rows.append({"site": site, "perfect_forecast_saving_eur": steps[0],
                     "lost_to_forecast_error_eur": steps[0] - steps[1],
                     "lost_to_state_estimate_and_margin_eur": steps[1] - steps[2],
                     "lost_to_faults_eur": steps[2] - steps[3], "captured_eur": steps[3],
                     "captured_share": steps[3] / steps[0] if steps[0] else np.nan,
                     "rule_with_estimator_saving_eur": v["threshold rule, Kalman, 2 sigma margin"],
                     "full_reserve_plan_with_estimator_saving_eur": v["floor, full reserve, Kalman, margin"]})
    return pd.DataFrame(rows)


def estimator_figure() -> None:
    def flows(site, slug):
        return pd.read_csv(OUT / "runs" / f"{site}__{slug}__0.csv.gz", index_col=0, parse_dates=True)

    cc = flows("A_factory", "floor__coulomb_counting")
    kf = flows("A_factory", "floor__kalman__no_margin")
    fig, ax = plt.subplots(figsize=(9, 3))
    for d, label in ((cc, "coulomb counting"), (kf, "extended Kalman filter")):
        ax.plot(d.index, 100 * (d["soc_est"] - d["soc"]), label=label, lw=0.8)
    ax.set(ylabel="state of charge error (points)", title="Site A, one year: estimate minus true state of charge")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fleet_soc_error.png", dpi=150)


def savings_figure(t: pd.DataFrame) -> None:
    names = [s.name for s in sites.FLEET]
    fig, axes = plt.subplots(1, len(names), figsize=(13, 4.2), sharey=True)
    for ax, site in zip(axes, names):
        v = t[t.site == site].set_index("strategy")["saving_with_atypical_use_eur"]
        order = [o for o in ORDER if o in v.index and o != "no battery"]
        ax.barh(order, v[order], color=["tab:grey" if "perfect" in o else "tab:blue" for o in order])
        ax.set_title(site)
        ax.set_xlabel("saving against no battery (EUR per year)")
        ax.axvline(0, color="black", lw=0.8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "fleet_savings.png", dpi=150)


def tariff_check() -> pd.DataFrame:
    """Tier and atypical-use position of each site without a battery."""
    rows = []
    for spec in sites.FLEET:
        d = sites.load_site(spec).loc[TEST[0]:TEST[1]]
        net = (d["load"] - d["pv"]).clip(lower=0)
        base = tariffs.network_charge(net, tariffs.Tariff())
        from ..data import HOLIDAYS

        a = tariffs.atypical_use(net, tariffs.Tariff(), holidays=HOLIDAYS)
        rows.append({"site": spec.name, "mean_load_kw": d["load"].mean(), "peak_import_kw": base["peak_kw"],
                     "utilisation_h": base["utilisation_h"], "tier": base["tier"],
                     "network_charge_eur": base["network_eur"], "window_peak_kw": a["window_peak_kw"],
                     "shift_kw": a["shift_kw"], "shift_share": a["shift_share"], "eligible_without_battery": a["eligible"]})
    return pd.DataFrame(rows)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    df = load_runs()
    t = results(df)
    t.to_csv(OUT / "results.csv", index=False, float_format="%.6g")
    w = waterfall(t)
    w.to_csv(OUT / "value_waterfall.csv", index=False, float_format="%.6g")
    s = tariff_check()
    s.to_csv(OUT / "tariff_check.csv", index=False, float_format="%.6g")
    savings_figure(t)
    estimator_figure()
    cols = ["site", "strategy", "saving_eur", "saving_with_atypical_use_eur", "total_eur", "peak_kw", "tier", "cycles",
            "soc_violation_hours", "avoidable_grid_violation_hours", "feed_in_violation_hours", "soc_rmse",
            "curtail_mwh", "atypical_eligible", "runs"]
    with pd.option_context("display.width", 250):
        print(t[cols].round(3).to_string(index=False))
        print(w.round(2).to_string(index=False))
        print(s.round(2).to_string(index=False))
    modes = {}
    for p in (OUT / "runs").glob("*faults*.json"):
        j = json.loads(p.read_text(encoding="utf-8"))
        modes[p.stem] = j["modes"]
    (OUT / "fault_modes.json").write_text(json.dumps(modes, indent=1), encoding="utf-8")
    print(json.dumps(modes))


if __name__ == "__main__":
    main()
