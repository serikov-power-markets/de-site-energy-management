"""Closed-loop simulation of one site for a year: optimiser (cloud) -> controller (on site) -> battery -> meter.

Hourly steps. At the start of each hour the optimiser plans over the hours whose day-ahead prices are known;
the controller receives the plan, the meter reading of the hour and the state of charge estimate and sets the
battery; the true cell runs through 60 one-minute samples with a noisy current and voltage sensor, which the
Kalman filter turns into the next estimate. As in v0.1 the controller sees the mean power of the hour it acts
in, which stands for a fast local loop.

Faults (random outages, all optional): the optimiser or the connection to it is down (the controller follows the
last plan, then a threshold rule), the forecast service is down (the optimiser falls back to the naive forecast),
the meter is down (the controller holds the plan without peak guard), the battery management reports a fault
(no power).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import controller as ctl
from . import estimator as est
from . import planner, tariffs
from ..data import HOLIDAYS
from .sites import SiteSpec

SAMPLES = 60                      # estimator samples per hour


@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str = "mpc"             # none, rule, mpc
    forecast: str = "ml"          # ml, naive, perfect
    scenarios: int = 1            # >1: stochastic planning over past forecast errors
    risk_weight: float = 0.0      # weight of CVaR of the peak
    floor: bool = False           # plan never targets a peak below the threshold learnt from history
    estimator: str = "ekf"        # ekf, coulomb, truth
    margin_sigma: float = 2.0
    faults: bool = False
    rule_aware: bool = True       # the optimiser knows the feed-in limit and the zero price rule
    atypical: bool = False        # the optimiser shaves only the peak inside the high-load windows
    end_target: float | None = None   # state of charge the plan aims for at the end of its horizon


@dataclass(frozen=True)
class Outages:
    start_prob: dict
    mean_hours: dict

    @staticmethod
    def default() -> "Outages":
        return Outages({"cloud": 0.002, "forecast": 0.001, "meter": 0.002, "bms": 0.0005},
                       {"cloud": 18.0, "forecast": 36.0, "meter": 4.0, "bms": 3.0})

    def draw(self, n: int, seed: int) -> dict[str, np.ndarray]:
        """Two-state chains: an outage starts with the given probability and ends with probability 1/mean."""
        rng = np.random.default_rng(seed)
        out = {}
        for name, p in self.start_prob.items():
            u, down, series = rng.random(n), False, np.zeros(n, bool)
            for i in range(n):
                down = (u[i] >= 1.0 / self.mean_hours[name]) if down else (u[i] < p)
                series[i] = down
            out[name] = series
        return out


def horizon_end(i: int, hour: np.ndarray, n: int) -> int:
    """Prices are known to the end of the day, and to the end of the next day from 13:00."""
    return min(i + (24 - hour[i]) + (24 if hour[i] >= 13 else 0), n)


class Scenarios:
    """Net-load scenarios: the forecast plus the forecast error of recent past days of the same day type."""

    def __init__(self, d: pd.DataFrame, fc: pd.DataFrame):
        err = (d["load"] - d["pv"]) - (fc["load"] - fc["pv"])
        frame = pd.DataFrame({"date": d["date"], "hour": d["hour"], "workday": d["workday"], "err": err}).dropna()
        table = frame.pivot_table(index="date", columns="hour", values="err", aggfunc="mean")
        self.table = table.dropna(thresh=20).fillna(0.0).reindex(columns=range(24), fill_value=0.0)
        self.workday = frame.groupby("date")["workday"].first().reindex(self.table.index)

    def errors(self, today: pd.Timestamp, workday: int, hours: np.ndarray, count: int) -> np.ndarray:
        """count x len(hours) errors from the latest past days (at least two days back) of the same type."""
        past = self.table.index <= today - pd.Timedelta(days=2)
        same = past & (self.workday.to_numpy() == workday)
        pool = self.table[same] if same.sum() >= count else self.table[past]
        if len(pool) == 0:
            return np.zeros((count, len(hours)))
        donors = pool.iloc[-count:].to_numpy()
        donors = donors[np.resize(np.arange(len(donors)), count)]
        return donors[:, hours]


def learnt_threshold(d: pd.DataFrame, before: str) -> float:
    train = d.loc[:before].iloc[:-1]
    return float((train["load"] - train["pv"]).quantile(0.995))


def site_prices(d_train: pd.DataFrame, tariff: tariffs.Tariff, spec: SiteSpec, feed_in_kw: float,
                negative: bool) -> tuple[planner.Prices, bool]:
    """Planning prices: the tariff tier follows from the utilisation time of the training months."""
    net = (d_train["load"] - d_train["pv"]).clip(lower=0)
    high = tariffs.utilisation_hours(float(net.sum()), float(net.max()), len(net)) >= tariffs.THRESHOLD_HOURS
    cap, work = tariff.prices(high)
    return planner.Prices(work + tariff.other_per_kwh, cap, spec.grid_limit_kw, feed_in_kw, negative), high


def simulate(d: pd.DataFrame, fc: pd.DataFrame, spec: SiteSpec, strategy: Strategy, tariff: tariffs.Tariff,
             test: tuple[str, str], feed_in_kw: float = float("inf"), negative_prices: bool = False,
             seed: int = 0, outages: Outages | None = None, phi: float = 0.8) -> tuple[pd.DataFrame, dict]:
    """d: hourly site data including the history; fc: forecasts from before the test start."""
    batt = spec.battery if strategy.end_target is None else replace(spec.battery, soc_target=strategy.end_target)
    t = d.loc[test[0]:test[1]]
    n = len(t)
    load, pv, price, hour = (t[c].to_numpy() for c in ("load", "pv", "price", "hour"))
    dates, workday = t["date"].to_numpy(), t["workday"].to_numpy()
    f = fc.reindex(t.index)
    fl, fp = f["load"].to_numpy(), f["pv"].to_numpy()
    nl, np_ = f["load_naive"].to_numpy(), f["pv_naive"].to_numpy()
    if strategy.forecast == "naive":
        fl, fp = nl, np_
    if strategy.forecast == "perfect":
        fl, fp = load, pv
    scen = Scenarios(d.loc[fc.index[0]:], fc) if strategy.scenarios > 1 else None
    train = d.loc[:test[0]].iloc[:-1]
    known = strategy.rule_aware
    prices, high = site_prices(train, tariff, spec, feed_in_kw if known else float("inf"),
                               negative_prices if known else True)
    threshold = learnt_threshold(d, test[0])
    limits = ctl.Limits(batt.power_kw, batt.energy_kwh, batt.soc_min, 1.0, batt.eta, spec.grid_limit_kw, feed_in_kw,
                        strategy.margin_sigma)
    down = (outages or Outages.default()).draw(n, seed) if strategy.faults else \
        {k: np.zeros(n, bool) for k in ("cloud", "forecast", "meter", "bms")}

    dt = 3600.0 / SAMPLES
    cell, sensors = est.Cell(eta=1.0), est.Sensors()      # losses are applied to the current (energy efficiency)
    plant = est.CellPlant(batt.soc_target, dt, cell, sensors, seed + 1)
    kalman = est.KalmanSoC(batt.soc_target, dt, cell, sensors, s0_std=0.02)
    counted = batt.soc_target
    estimate, sigma = batt.soc_target, 0.0
    peak, last = 0.0, None        # billing peak so far; (hour of plan, power trajectory, planned peak)
    err = 0.0
    window = tariffs.high_load_mask(t.index, HOLIDAYS) if strategy.atypical else np.ones(n, bool)
    rows = []
    for i in range(n):
        meter_ok = not down["meter"][i]
        decision_plan = None
        if strategy.kind == "mpc":
            if not down["cloud"][i]:
                j = horizon_end(i, hour, n)
                use_naive = down["forecast"][i]
                load_fc, pv_fc = (nl, np_) if use_naive else (fl, fp)
                net_fc = load_fc[i:j] - pv_fc[i:j]
                if strategy.forecast != "perfect":
                    net_fc = net_fc + err * phi ** np.arange(1, j - i + 1)
                pv_h = np.clip(pv_fc[i:j], 0.0, None)
                if scen is not None and not use_naive:
                    noise = scen.errors(pd.Timestamp(dates[i]), int(workday[i]), hour[i:j], strategy.scenarios)
                    net = net_fc + noise
                    pv_s = np.tile(pv_h, (strategy.scenarios, 1))
                else:
                    net, pv_s = net_fc[None, :], pv_h[None, :]
                start_peak = max(peak, threshold if strategy.floor else 0.0)
                stored = min(max(estimate, batt.soc_min), 1.0) * batt.energy_kwh
                plan = planner.solve(net, pv_s, price[i:j], stored, start_peak, batt, prices, strategy.risk_weight,
                                     billed=window[i:j])
                traj = plan["power_kw"].copy()
                traj[0] = plan["first_kw"]
                last = (i, traj, plan["peak_kw"])
            if last is not None:
                age = i - last[0]
                if age < len(last[1]):
                    guard = last[2] if window[i] else spec.grid_limit_kw      # no peak guard outside billed hours
                    decision_plan = ctl.Plan(float(last[1][age]), guard, age)
        meas = ctl.Measurement(load[i] if meter_ok else None, pv[i] if meter_ok else None,
                               None if down["bms"][i] else float(estimate), float(sigma))
        if strategy.kind == "none":
            cmd = {"mode": "none", "power_kw": 0.0, "curtail_kw": max(pv[i] - load[i] - feed_in_kw, 0.0)}
        else:
            cmd = ctl.step(decision_plan, meas, limits, max(threshold, peak))
        soc_before = plant.s
        _, power = ctl.apply_to_battery(plant.s, cmd["power_kw"], limits)
        # the cell: current that reproduces the energy balance of the commanded power, sampled each minute
        energy_flow = power / batt.eta if power > 0 else power * batt.eta
        current = energy_flow / batt.energy_kwh * cell.capacity_ah
        for _ in range(SAMPLES):
            i_meas, u_meas = plant.step(current)
            s_hat, s_std, _ = kalman.update(i_meas, u_meas)
            counted = min(max(counted - i_meas * dt / (cell.capacity_ah * 3600.0), 0.0), 1.0)
            if plant.s >= 0.999:                      # full charge, seen at the voltage limit: recalibrate
                counted = 1.0
        pred, pred_std = float(kalman.x[0]), float(s_std)
        if strategy.estimator == "ekf":
            estimate, sigma = pred, pred_std
        elif strategy.estimator == "coulomb":
            estimate, sigma = counted, 0.0
        else:
            estimate, sigma = plant.s, 0.0
        net = load[i] - pv[i] - power
        available = max(-net, 0.0)
        curtail = min(cmd["curtail_kw"], pv[i], available)
        if np.isfinite(feed_in_kw):                     # the inverter limits the feed-in even if the meter is lost
            curtail = max(curtail, min(available - feed_in_kw, pv[i]) if available > feed_in_kw else curtail)
        if negative_prices and price[i] < 0:            # exporting would cost money: curtail what is left
            curtail = max(curtail, min(available, pv[i]))
        net += curtail
        imp, exp = max(net, 0.0), max(-net, 0.0)
        # could the battery have kept the import within the connection limit?
        spare = max(min(batt.power_kw, (soc_before - batt.soc_min) * batt.energy_kwh * batt.eta) - max(power, 0.0), 0.0)
        rows.append({"imp": imp, "exp": exp, "power": power, "curtail": curtail, "soc": plant.s, "soc_est": estimate,
                     "soc_std": sigma, "mode": cmd["mode"],
                     "soc_below_min": plant.s < batt.soc_min - 1e-4,
                     "grid_violation": imp > spec.grid_limit_kw + 1e-6,
                     "avoidable_grid_violation": imp > spec.grid_limit_kw + 1e-6 and spare > 1.0,
                     "feed_in_violation": exp > feed_in_kw + 1e-6})
        if window[i]:
            peak = max(peak, imp)
        if strategy.kind == "mpc" and meter_ok:
            err = (load[i] - pv[i]) - (fl[i] - fp[i])
    flows = pd.DataFrame(rows, index=t.index)
    return flows, bill(t, flows, spec, tariff, negative_prices) | {"high_tier_planned": bool(high),
                                                                    "threshold_kw": threshold}


def bill(t: pd.DataFrame, flows: pd.DataFrame, spec: SiteSpec, tariff: tariffs.Tariff, negative_prices: bool) -> dict:
    """The year's cost: network charge with the tier set by the realised utilisation time, energy at day-ahead
    prices, export revenue, battery wear; the value of the stored energy at the end is not credited."""
    p = t["price"].to_numpy() / 1000.0
    network = tariffs.network_charge(flows["imp"], tariff)
    energy = float((flows["imp"] * p).sum())
    export_price = p if negative_prices else np.maximum(p, 0.0)
    export = float((flows["exp"] * export_price).sum())
    discharged = float(flows["power"].clip(lower=0).sum())
    wear = spec.battery.wear * discharged
    stored = float((flows["soc"].iloc[-1] - spec.battery.soc_target) * spec.battery.energy_kwh)
    atypical = tariffs.atypical_use(flows["imp"], tariff, holidays=HOLIDAYS)
    return {"total_eur": network["network_eur"] + energy - export + wear, "network_eur": network["network_eur"],
            "capacity_eur": network["capacity_eur"], "energy_eur": energy, "export_eur": export, "wear_eur": wear,
            "peak_kw": network["peak_kw"], "utilisation_h": network["utilisation_h"], "tier": network["tier"],
            "cycles": discharged / spec.battery.energy_kwh, "stored_change_kwh": stored,
            "curtail_mwh": float(flows["curtail"].sum() / 1000),
            "soc_violation_hours": int(flows["soc_below_min"].sum()),
            "grid_violation_hours": int(flows["grid_violation"].sum()),
            "avoidable_grid_violation_hours": int(flows["avoidable_grid_violation"].sum()),
            "feed_in_violation_hours": int(flows["feed_in_violation"].sum()),
            "soc_rmse": float(np.sqrt(((flows["soc_est"] - flows["soc"]) ** 2).mean())),
            "atypical_eligible": atypical["eligible"], "atypical_shift_kw": atypical["shift_kw"],
            "atypical_saving_eur": atypical["saving_eur"]}
