"""The fleet: two metered industrial sites in Konstanz (OPSD industrial3 and industrial1) with hypothetical
batteries and PV scaled from their own measured arrays, and one synthetic large site (site A times six) that is
big enough for the rules of atypical grid use. A third metered industrial site of the data set (industrial2)
draws about 1.5 kW on average, far below the size billed with a capacity price, and is left out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data, forecast
from .planner import Battery

TEST_START, TEST_END = "2016-05-31 22:00", "2017-05-31 21:00"    # 1 June 2016 to 31 May 2017, local time
WARMUP_START = "2016-03-31 22:00"                                # forecasts from April give past errors


@dataclass(frozen=True)
class SiteSpec:
    name: str
    meter: str                       # OPSD site prefix
    pv_columns: tuple[str, ...]
    pv_kwp: float
    battery: Battery
    grid_limit_kw: float
    scale: float = 1.0               # >1: synthetic enlargement of load, PV, battery and grid limit


A = SiteSpec("A_factory", "DE_KN_industrial3_", ("pv_roof", "pv_facade"), 150.0, Battery(200.0, 100.0), 220.0)
B = SiteSpec("B_workshop", "DE_KN_industrial1_", ("pv_1", "pv_2"), 40.0, Battery(80.0, 40.0), 130.0)
C = SiteSpec("C_large_synthetic", "DE_KN_industrial3_", ("pv_roof", "pv_facade"), 900.0, Battery(1200.0, 600.0),
             1320.0, scale=6.0)
FLEET = (A, B, C)


def load_site(spec: SiteSpec, raw: Path = data.RAW) -> pd.DataFrame:
    """Hourly load, PV (kW), day-ahead price, weather and calendar of one site.

    Load = grid import + on-site PV (the measured arrays are small against the load, so their output is used on
    site). Hours with a negative meter difference are treated as missing and filled from one week earlier.
    """
    usecols = ["utc_timestamp", spec.meter + "grid_import"] + [spec.meter + c for c in spec.pv_columns]
    m = pd.read_csv(raw / "household_data_60min_singleindex.csv", usecols=usecols, parse_dates=["utc_timestamp"],
                    index_col="utc_timestamp")
    kw = m.diff()
    kw.columns = [c.removeprefix(spec.meter) for c in kw.columns]
    kw.index = kw.index.tz_convert("UTC")
    kw = kw.loc[data.START:data.END]
    kw = kw.mask(kw < 0)
    pv_meas = kw[list(spec.pv_columns)].sum(axis=1, min_count=1)
    load = kw["grid_import"] + pv_meas.fillna(0.0)
    d = pd.DataFrame({"load": load, "pv_meas": pv_meas})
    d["load"] = d["load"].fillna(d["load"].shift(168)).interpolate(limit=24)
    d["pv_meas"] = d["pv_meas"].fillna(d["pv_meas"].shift(24)).fillna(0.0)
    cap = d["pv_meas"].quantile(0.999)
    d["pv"] = (d["pv_meas"] / cap * spec.pv_kwp / spec.scale).clip(lower=0.0)
    d = d.join(data.prices(raw), how="left").join(data.weather(raw), how="left").join(data.calendar(d.index))
    d["price"] = d["price"].interpolate(limit=6)
    d["cooling"] = 0.0
    for c in ("load", "pv"):
        d[c] *= spec.scale
    return d


def forecasts(d: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """Day-ahead forecasts of load and PV from April 2016: gradient boosting with ERA5 weather, and the naive
    fallback (same hour one week earlier; PV: mean of the last seven known days)."""
    if cache.exists():
        out = pd.read_csv(cache, index_col=0)
        out.index = pd.to_datetime(out.index, utc=True)
        return out
    ml = forecast.rolling_ml(d, WARMUP_START, weather=True)
    naive = forecast.naive(d).reindex(ml.index)
    out = pd.DataFrame({"load": ml["load"], "pv": ml["pv"], "load_naive": naive["load"], "pv_naive": naive["pv"]})
    out = out.fillna(pd.DataFrame({"load": d["load"].shift(168), "pv": d["pv"].shift(48),
                                   "load_naive": d["load"].shift(336), "pv_naive": d["pv"].shift(48)})
                     .reindex(out.index)).ffill()
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, float_format="%.4f")
    return out


def quality(d: pd.DataFrame) -> dict:
    test = d.loc[TEST_START:TEST_END]
    return {"hours": int(len(test)), "mean_load_kw": float(test["load"].mean()), "max_load_kw": float(test["load"].max()),
            "pv_mwh": float(test["pv"].sum() / 1000), "missing_price_hours": int(test["price"].isna().sum()),
            "missing_load_hours": int(np.isnan(test["load"]).sum())}
