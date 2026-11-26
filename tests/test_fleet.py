"""Fleet modules on small synthetic cases: tariff rules, planner, controller limits, estimator, closed loop."""

import numpy as np
import pandas as pd
import pytest

from siteems.fleet import controller as ctl
from siteems.fleet import estimator as est
from siteems.fleet import planner, sim, tariffs
from siteems.fleet.sites import SiteSpec


# ------------------------------------------------------------------ tariffs
def series(values, start="2017-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="h", tz="UTC"), dtype=float)


def test_tier_switches_at_2500_hours():
    tariff = tariffs.Tariff()
    flat = series(np.full(8760, 100.0))
    assert tariffs.network_charge(flat, tariff)["tier"] == ">=2500 h"
    spiky = flat * 0.2
    spiky.iloc[10] = 100.0
    low = tariffs.network_charge(spiky, tariff)
    assert low["tier"] == "<2500 h" and low["utilisation_h"] < 2500
    assert low["capacity_eur"] == pytest.approx(tariff.low_capacity * 100.0)


def test_shaving_a_peak_can_change_the_tier():
    tariff = tariffs.Tariff()
    load = np.full(8760, 20.0)
    load[::24] = 90.0                           # a daily spike keeps the utilisation just below 2,500 h
    before = tariffs.network_charge(series(load), tariff)
    load[::24] = 60.0
    after = tariffs.network_charge(series(load), tariff)
    assert before["tier"] == "<2500 h" and after["tier"] == ">=2500 h"


def test_high_load_windows_only_on_working_days_in_winter_and_autumn():
    idx = pd.date_range("2017-01-02 00:00", periods=24 * 14, freq="h", tz="Europe/Berlin").tz_convert("UTC")
    mask = pd.Series(tariffs.high_load_mask(idx), index=idx.tz_convert("Europe/Berlin"))
    assert mask["2017-01-02 17:00"] and mask["2017-01-02 07:00"] and not mask["2017-01-02 12:00"]
    assert not mask["2017-01-07"].any()                                     # Saturday
    summer = pd.date_range("2017-07-03", periods=48, freq="h", tz="Europe/Berlin").tz_convert("UTC")
    assert not tariffs.high_load_mask(summer).any()


def test_atypical_use_needs_share_absolute_shift_and_minimum_saving():
    idx = pd.date_range("2017-01-01", "2017-12-31 23:00", freq="h", tz="Europe/Berlin").tz_convert("UTC")
    load = pd.Series(500.0, index=idx)
    inside = tariffs.high_load_mask(idx)
    load[~inside] = 900.0
    ok = tariffs.atypical_use(load, tariffs.Tariff())
    assert ok["eligible"] and ok["shift_kw"] == 400.0 and ok["saving_eur"] > 500
    small = tariffs.atypical_use(load / 10, tariffs.Tariff())              # 40 kW shift: below 100 kW
    assert not small["eligible"] and small["saving_eur"] == 0.0
    flat = load.where(inside, 550.0)                                       # 50 kW and 9%: not significant
    assert not tariffs.atypical_use(flat, tariffs.Tariff())["eligible"]


# ------------------------------------------------------------------ planner
BATT = planner.Battery(100.0, 50.0, wear=0.0)
PRICES = planner.Prices(0.1, 100.0)


def test_plan_shaves_the_peak():
    net = np.r_[np.full(10, 50.0), 120.0, 120.0, np.full(12, 50.0)]
    res = planner.solve(net, np.zeros_like(net), np.full(len(net), 40.0), 50.0, 0.0, BATT, PRICES)
    assert res["peak_kw"] < 100.0
    assert res["power_kw"][10] > 20.0


def test_first_decision_is_shared_by_all_scenarios_and_peak_floor_holds():
    rng = np.random.default_rng(0)
    net = 60 + 30 * rng.random((5, 24))
    res = planner.solve(net, np.zeros_like(net), np.full(24, 40.0), 50.0, 95.0, BATT, PRICES)
    assert (res["peaks_kw"] >= 95.0 - 1e-6).all()
    assert np.isfinite(res["first_kw"])


def test_risk_weight_prepares_for_the_bad_scenario():
    net = np.full((5, 12), 40.0)
    net[0, 8] = 130.0                                       # one scenario in five has a late spike
    args = (net, np.zeros_like(net), np.full(12, 40.0), 20.0, 0.0, BATT, planner.Prices(0.1, 100.0))
    neutral = planner.solve(*args, risk_weight=0.0)
    averse = planner.solve(*args, risk_weight=1.0, alpha=0.8)
    assert averse["peaks_kw"].max() <= neutral["peaks_kw"].max() + 1e-6


def test_hours_outside_the_billed_window_do_not_set_the_peak():
    net = np.r_[np.full(5, 150.0), np.full(5, 60.0)]
    billed = np.r_[np.zeros(5, bool), np.ones(5, bool)]
    res = planner.solve(net, np.zeros_like(net), np.full(10, 40.0), 50.0, 0.0, BATT, PRICES, billed=billed)
    assert res["peak_kw"] <= 60.0 + 1e-6


# ------------------------------------------------------------------ controller and safety filter
def test_modes_follow_what_is_available():
    lim = ctl.Limits(50.0, 100.0)
    ok = ctl.Measurement(60.0, 0.0, 0.5)
    plan = ctl.Plan(10.0, 80.0)
    assert ctl.mode(plan, ok, lim) == ctl.NORMAL
    assert ctl.mode(None, ok, lim) == ctl.NO_PLAN
    assert ctl.mode(ctl.Plan(0, 0, age_h=30), ok, lim) == ctl.NO_PLAN
    assert ctl.mode(plan, ctl.Measurement(None, None, 0.5), lim) == ctl.HOLD
    assert ctl.mode(None, ctl.Measurement(None, None, 0.5), lim) == ctl.NO_METER
    assert ctl.mode(plan, ctl.Measurement(60.0, 0.0, None), lim) == ctl.STOP


@pytest.mark.parametrize("seed", range(5))
def test_safety_filter_never_breaks_a_limit(seed):
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        lim = ctl.Limits(rng.uniform(10, 200), rng.uniform(20, 500), soc_min=0.1, grid_import_kw=rng.uniform(20, 300),
                         feed_in_kw=rng.uniform(0, 100), margin_sigma=2.0)
        std = rng.uniform(0, 0.05)
        true = rng.uniform(lim.soc_min, 1.0)
        estimate = float(np.clip(true + std * rng.uniform(-2, 2), 0, 1))       # error within the margin
        meas = ctl.Measurement(rng.uniform(0, 300), rng.uniform(0, 250), estimate, std)
        plan = None if rng.random() < 0.3 else ctl.Plan(rng.uniform(-300, 300), rng.uniform(0, 300))
        cmd = ctl.step(plan, meas, lim, rng.uniform(0, 300))
        p = cmd["power_kw"]
        assert abs(p) <= lim.power_kw + 1e-9
        after = true - (p / lim.eta if p > 0 else p * lim.eta) / lim.energy_kwh
        if abs(estimate - true) <= lim.margin_sigma * std + 1e-12:
            if p > 0:
                assert after >= lim.soc_min - 1e-9
            else:
                assert after <= lim.soc_max + 1e-9
        net = meas.load_kw - meas.pv_kw + cmd["curtail_kw"]
        assert -(net - p) <= lim.feed_in_kw + 1e-6                             # export within the limit
        high = ctl.soc_bounds(meas, lim)[1]
        if net - p > lim.grid_import_kw + 1e-6:                                # import above the limit only if
            assert p >= high - 1e-6                                            # the battery already gives all
        if p < 0:
            assert net - p <= max(lim.grid_import_kw, net) + 1e-6              # charging never causes it


# ------------------------------------------------------------------ estimator
def test_kalman_filter_removes_the_drift_of_coulomb_counting():
    cell, sensors = est.Cell(), est.Sensors(current_offset=1.0)
    hours = 72
    t = np.arange(hours * 3600) / 3600
    current = 30 * np.sin(2 * np.pi * t / 24)
    run = est.simulate(current, 0.5, 1.0, cell, sensors, seed=2)
    counted = est.coulomb_count(run["current_meas"], 0.6, 1.0, cell)
    filtered = est.ekf(run["current_meas"], run["voltage_meas"], 0.6, 1.0, cell, sensors)
    error_counted = np.abs(counted - run["soc"])[-3600:].mean()
    error_filtered = np.abs(filtered["soc"] - run["soc"])[-3600:].mean()
    assert error_counted > 0.3 and error_filtered < 0.03


def test_incremental_filter_matches_the_batch_filter():
    cell, sensors = est.Cell(), est.Sensors()
    current = np.r_[np.full(600, 20.0), np.full(600, -20.0)]
    plant = est.CellPlant(0.5, 10.0, cell, sensors, seed=4)
    readings = np.array([plant.step(i) for i in current])
    f = est.KalmanSoC(0.5, 10.0, cell, sensors)
    step = [f.update(*r)[0] for r in readings]
    assert np.allclose(step, est.ekf(readings[:, 0], readings[:, 1], 0.5, 10.0, cell, sensors)["soc"])


# ------------------------------------------------------------------ closed loop
def synthetic(days=40):
    idx = pd.date_range("2016-04-01", periods=24 * days, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    hour = idx.tz_convert("Europe/Berlin").hour
    load = 40 + 50 * ((hour >= 7) & (hour < 17)) + 8 * rng.standard_normal(len(idx))
    pv = np.clip(30 * np.sin((hour - 6) / 12 * np.pi), 0, None)
    local = idx.tz_convert("Europe/Berlin")
    d = pd.DataFrame({"load": np.clip(load, 5, None), "pv": pv, "price": 30 + 20 * np.sin(hour / 24 * 2 * np.pi),
                      "hour": hour, "workday": (local.dayofweek < 5).astype(int),
                      "month": local.month, "date": pd.to_datetime(local.date)}, index=idx)
    fc = pd.DataFrame({"load": 40 + 50 * ((hour >= 7) & (hour < 17)), "pv": pv}, index=idx)
    fc["load_naive"], fc["pv_naive"] = d["load"].shift(168).bfill(), d["pv"].shift(48).bfill()
    return d, fc


SPEC = SiteSpec("t", "", (), 30.0, planner.Battery(100.0, 40.0), 115.0)


@pytest.mark.parametrize("strategy", [sim.Strategy("det"), sim.Strategy("stoch", scenarios=4),
                                      sim.Strategy("faults", faults=True), sim.Strategy("rule", "rule")])
def test_closed_loop_respects_the_battery_and_never_causes_a_grid_violation(strategy):
    d, fc = synthetic()
    outages = sim.Outages({"cloud": 0.05, "forecast": 0.05, "meter": 0.05, "bms": 0.01},
                          {"cloud": 6.0, "forecast": 6.0, "meter": 3.0, "bms": 2.0})
    flows, bill = sim.simulate(d, fc, SPEC, strategy, tariffs.Tariff(), ("2016-04-20", "2016-04-30 23:00"),
                               outages=outages, feed_in_kw=20.0)
    assert bill["soc_violation_hours"] == 0
    assert bill["feed_in_violation_hours"] == 0
    assert flows["soc"].between(0, 1).all()
    if not strategy.faults:
        assert bill["avoidable_grid_violation_hours"] == 0


def test_perfect_forecast_is_not_worse_than_the_rule():
    d, fc = synthetic()
    test = ("2016-04-20", "2016-04-30 23:00")
    best = sim.simulate(d, fc, SPEC, sim.Strategy("p", forecast="perfect", estimator="truth", margin_sigma=0),
                        tariffs.Tariff(), test)[1]
    rule = sim.simulate(d, fc, SPEC, sim.Strategy("r", "rule"), tariffs.Tariff(), test)[1]
    assert best["total_eur"] <= rule["total_eur"] + 1.0


def test_identical_scenarios_give_the_deterministic_decision():
    rng = np.random.default_rng(3)
    net = 60 + 60 * rng.random(24)
    args = (np.full(24, 40.0), 50.0, 0.0, BATT, PRICES)
    one = planner.solve(net, np.zeros(24), *args)
    many = planner.solve(np.tile(net, (4, 1)), np.zeros((4, 24)), *args)
    assert many["cost"] == pytest.approx(one["cost"], rel=1e-6)
    assert many["peak_kw"] == pytest.approx(one["peak_kw"], rel=1e-6)
