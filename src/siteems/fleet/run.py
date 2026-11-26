"""All fleet runs of v0.3. Each run is stored as soon as it finishes, so an interrupted job continues.

    python -m siteems.fleet.run [workers]          # about 40 minutes with 4 workers
    python -m siteems.fleet.report                 # tables and figures from the stored runs
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from . import sim, sites, tariffs

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "fleet"
TEST = (sites.TEST_START, sites.TEST_END)
SCENARIOS = 6
FAULT_SEEDS = (0, 1, 2)
PV_NEW_KWP = 99.0                       # new roof below 100 kWp for the feed-in case
FEED_IN_SHARE = 0.6                     # feed-in limit of a new system without remote control

truth = {"estimator": "truth", "margin_sigma": 0.0}
STRATEGIES = (
    sim.Strategy("no battery", "none", **truth),
    sim.Strategy("threshold rule", "rule", **truth),
    sim.Strategy("perfect forecast", forecast="perfect", **truth),
    sim.Strategy("MPC naive forecast", forecast="naive", **truth),
    sim.Strategy("MPC", **truth),
    sim.Strategy("MPC, scenarios", scenarios=SCENARIOS, **truth),
    sim.Strategy("MPC, scenarios, CVaR", scenarios=SCENARIOS, risk_weight=0.5, **truth),
    sim.Strategy("MPC, peak floor", floor=True, **truth),
    sim.Strategy("floor, coulomb counting", floor=True, estimator="coulomb", margin_sigma=0.0),
    sim.Strategy("floor, Kalman, no margin", floor=True, estimator="ekf", margin_sigma=0.0),
    sim.Strategy("floor, Kalman, 2 sigma margin", floor=True),
    sim.Strategy("floor, Kalman, margin, faults", floor=True, faults=True),
)
# added after the first results, to find where the value is lost (labelled as such in the README)
FOLLOW_UP = (
    sim.Strategy("threshold rule, Kalman, 2 sigma margin", "rule"),
    sim.Strategy("floor, full reserve", floor=True, end_target=1.0, **truth),
    sim.Strategy("floor, full reserve, Kalman, margin", floor=True, end_target=1.0),
)
ATYPICAL = (
    sim.Strategy("atypical-use plan", atypical=True, **truth),
    sim.Strategy("atypical-use plan, perfect forecast", forecast="perfect", atypical=True, **truth),
)
FEED_IN = (
    sim.Strategy("feed-in case: no battery", "none", **truth),
    sim.Strategy("feed-in case: plan ignores the limit", rule_aware=False, **truth),
    sim.Strategy("feed-in case: plan knows the limit", **truth),
)


def jobs() -> list[tuple[str, sim.Strategy, int]]:
    out = []
    for spec in sites.FLEET:
        for s in STRATEGIES:
            for seed in (FAULT_SEEDS if s.faults else (0,)):
                out.append((spec.name, s, seed))
        out += [(spec.name, s, 0) for s in FOLLOW_UP]
    out += [(spec.name, s, 0) for spec in sites.FLEET for s in ATYPICAL]
    out += [("B_new_roof", s, 0) for s in FEED_IN]
    return out


def job_file(site: str, strategy: sim.Strategy, seed: int) -> Path:
    slug = "".join(c if c.isalnum() else "_" for c in strategy.name.lower())
    return OUT / "runs" / f"{site}__{slug}__{seed}.json"


def prepare(site: str):
    """(spec, data, forecasts, extra simulation arguments) of a site or of the feed-in case."""
    if site == "B_new_roof":
        spec = replace(sites.B, name=site, pv_kwp=PV_NEW_KWP)
        d = sites.load_site(spec)
        fc = sites.forecasts(sites.load_site(sites.B), OUT / f"forecasts_{sites.B.name}.csv.gz")
        scale = PV_NEW_KWP / sites.B.pv_kwp
        fc = fc.assign(pv=fc["pv"] * scale, pv_naive=fc["pv_naive"] * scale)
        return spec, d, fc, {"feed_in_kw": FEED_IN_SHARE * PV_NEW_KWP}
    spec = next(s for s in sites.FLEET if s.name == site)
    d = sites.load_site(spec)
    return spec, d, sites.forecasts(d, OUT / f"forecasts_{site}.csv.gz"), {}


def run_one(job: tuple[str, sim.Strategy, int]) -> tuple[str, float]:
    site, strategy, seed = job
    path = job_file(*job)
    if path.exists():
        return path.name, 0.0
    t0 = time.time()
    spec, d, fc, extra = prepare(site)
    flows, bill = sim.simulate(d, fc, spec, strategy, tariffs.Tariff(), TEST, seed=seed, **extra)
    keep = flows[["imp", "exp", "power", "curtail", "soc", "soc_est", "mode"]]
    keep.to_csv(path.with_suffix(".csv.gz"), float_format="%.4g")
    modes = flows["mode"].value_counts().to_dict()
    path.write_text(json.dumps({"site": site, "strategy": strategy.name, "seed": seed, "settings": asdict(strategy),
                                "modes": modes, **bill}, indent=1), encoding="utf-8")
    return path.name, time.time() - t0


def main(workers: int = 4) -> None:
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    for spec in sites.FLEET:                                   # forecasts once, before the workers need them
        d = sites.load_site(spec)
        sites.forecasts(d, OUT / f"forecasts_{spec.name}.csv.gz")
        print("forecasts", spec.name, flush=True)
    todo = [j for j in jobs() if not job_file(*j).exists()]
    todo.sort(key=lambda j: j[1].scenarios)                    # quick runs first
    print(len(todo), "runs to do", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(run_one, j): j for j in todo}
        for k, f in enumerate(as_completed(futures), start=1):
            name, seconds = f.result()
            print(f"{k}/{len(todo)} {name} {seconds:.0f} s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
