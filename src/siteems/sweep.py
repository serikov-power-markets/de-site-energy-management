"""Monthly cost of the MPC as a function of its peak floor: the table the safe-learning study runs on.

For every peak floor on a grid (0 = plain MPC) the year is simulated with the weather forecast and monthly demand
charges; the bill of each of the twelve months is stored. Result: results/floor_by_month.csv (rows: floor, columns:
month), costs in EUR. One simulation is one 'experiment' of a month; the learners in safeopt.py read this table
instead of re-running the controller, so every algorithm sees exactly the same plant.

    python -m siteems.sweep
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from . import ems, run

FLOORS = [0.0] + [float(x) for x in range(120, 205, 5)]


def load_inputs():
    d = run.prepare()
    test = d.loc[run.TEST_START:run.TEST_END]
    allf = pd.read_csv(run.RES / "forecasts.csv.gz", header=[0, 1], index_col=0)
    allf.index = pd.to_datetime(allf.index, utc=True)
    v = allf["ml_weather"].loc[test.index]
    v = v.fillna(test[["load", "cooling", "pv"]].shift(168).reindex(columns=v.columns)).ffill()
    return test, v


def monthly_bills(test: pd.DataFrame, flows: pd.DataFrame, site: ems.Site) -> list[float]:
    out = []
    for m in sorted(test["month"].unique(), key=lambda k: test.index[test["month"] == k][0]):
        mask = (test["month"] == m).to_numpy()
        b = ems.bill(test[mask], flows[mask], site)
        out.append(b["total_eur"])
    return out


def one(floor: float) -> tuple[float, list[float]]:
    test, fc = load_inputs()
    site = replace(ems.Site(), billing="monthly", peak_floor=floor)
    fl = ems.simulate(test, fc, site, strategy="mpc")
    return floor, monthly_bills(test, fl, site)


def main() -> pd.DataFrame:
    t = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = dict(ex.map(one, FLOORS))
    test, _ = load_inputs()
    months = [str(p) for p in pd.Series(test.index.tz_convert("Europe/Berlin").to_period("M")).unique()]
    tab = pd.DataFrame(res, index=months).T.sort_index()
    tab.index.name = "floor_kw"
    tab.to_csv(run.RES / "floor_by_month.csv", float_format="%.2f")
    print(tab.round(0).to_string())
    print(f"{time.time() - t:.0f} s")
    return tab


if __name__ == "__main__":
    main()
