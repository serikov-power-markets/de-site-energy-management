"""The optimiser: a battery plan for one site over the known price horizon, deterministic or over scenarios.

Linear programme (HiGHS). For every scenario s of the net load (load minus PV) and hour t:

    import - export - charge + discharge - curtailment = net load        (kW)
    soc(t) = soc(t-1) + eta charge - discharge / eta                      (kWh)
    import <= P(s) ,  import <= grid limit (soft),  export <= feed-in limit

    cost(s) = sum_t (price + adder) import - price export + wear discharge + value of end energy shortfall
    objective = mean_s cost(s) + demand price x [(1 - lambda) mean_s P(s) + lambda CVaR_alpha(P(s))]

Charge and discharge of the first hour are the same in all scenarios (the decision that is carried out cannot
depend on the scenario); later hours may differ. P(s) starts from the peak already reached in the billing period
or from a floor, whichever is higher.
"""

from __future__ import annotations

from dataclasses import dataclass

import highspy
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Battery:
    energy_kwh: float
    power_kw: float
    soc_min: float = 0.1
    eta: float = 0.95
    wear: float = 0.03               # EUR per kWh discharged
    soc_target: float = 0.5


@dataclass(frozen=True)
class Prices:
    energy_adder: float              # EUR per kWh imported on top of the day-ahead price
    demand: float                    # EUR per kW of the billing peak
    grid_limit_kw: float = float("inf")
    feed_in_kw: float = float("inf")
    export_at_negative: bool = False  # whether negative prices are passed on to exports


def _highs() -> highspy.Highs:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)                 # many small programmes: threads only add overhead
    return h


def solve_lp(c, A_ub, b_ub, A_eq, b_eq, lo, hi) -> tuple[np.ndarray, float]:
    """min c x with A_ub x <= b_ub, A_eq x = b_eq and lo <= x <= hi, passed to HiGHS directly.

    (The scipy wrapper around the same solver took 30 to 100 times longer per programme on the build machine.)"""
    A = sparse.vstack([A_ub, A_eq]).tocsc()
    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_row_, lp.num_col_ = A.shape
    lp.col_cost_ = c
    lp.col_lower_, lp.col_upper_ = np.clip(lo, -inf, inf), np.clip(hi, -inf, inf)
    lp.row_lower_ = np.r_[np.full(len(b_ub), -inf), b_eq]
    lp.row_upper_ = np.r_[b_ub, b_eq]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    h = _highs()
    h.passModel(lp)
    h.run()
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(h.modelStatusToString(status))
    return np.array(h.getSolution().col_value), h.getInfo().objective_function_value


def solve(net: np.ndarray, pv: np.ndarray, price: np.ndarray, soc0_kwh: float, peak0: float, batt: Battery,
          prices: Prices, risk_weight: float = 0.0, alpha: float = 0.8, billed: np.ndarray | None = None) -> dict:
    """net, pv: scenarios x hours (kW); price: EUR/MWh per hour. billed: hours whose import counts for the billed
    peak (all by default; the high-load windows under atypical grid use). Returns the plan (first hour and
    trajectories)."""
    net, pv = np.atleast_2d(net), np.atleast_2d(pv)
    S, H = net.shape
    billed = np.ones(H, bool) if billed is None else np.asarray(billed, bool)
    p = np.asarray(price, float) / 1000.0
    blocks = 6                         # charge, discharge, import, export, curtail, soc
    per = blocks * H + 3               # + peak P, end-energy shortfall, grid-limit excess
    tail = S + 1 if risk_weight > 0 else 0
    n = S * per + tail
    c = np.zeros(n)
    lo, hi = np.zeros(n), np.full(n, np.inf)
    rows, cols, vals, beq = [], [], [], []
    urows, ucols, uvals, bub = [], [], [], []
    r = q = 0
    exp_price = p if prices.export_at_negative else np.maximum(p, 0.0)
    for s in range(S):
        o = s * per
        ch, dis, imp, exp, cur, soc = (o + k * H + np.arange(H) for k in range(blocks))
        peak, short, over = o + blocks * H, o + blocks * H + 1, o + blocks * H + 2
        c[imp] = (p + prices.energy_adder) / S
        c[exp] = -exp_price / S
        c[dis] = batt.wear / S
        c[cur] = 1e-4 / S
        c[peak] = prices.demand * (1.0 - risk_weight) / S
        c[short] = 0.5 / S
        c[over] = 1000.0 / S
        hi[ch] = hi[dis] = batt.power_kw
        hi[cur] = np.maximum(pv[s], 0.0)
        hi[exp] = prices.feed_in_kw
        lo[soc], hi[soc] = batt.soc_min * batt.energy_kwh, batt.energy_kwh
        lo[peak] = peak0
        for t in range(H):
            for j, v in ((imp[t], 1), (exp[t], -1), (ch[t], -1), (dis[t], 1), (cur[t], -1)):
                rows.append(r), cols.append(j), vals.append(v)
            beq.append(net[s, t]), (r := r + 1)
        for t in range(H):
            entries = [(soc[t], 1.0), (ch[t], -batt.eta), (dis[t], 1.0 / batt.eta)]
            if t:
                entries.append((soc[t - 1], -1.0))
            for j, v in entries:
                rows.append(r), cols.append(j), vals.append(v)
            beq.append(soc0_kwh if t == 0 else 0.0), (r := r + 1)
        for t in range(H):
            if billed[t]:
                urows += [q, q]; ucols += [imp[t], peak]; uvals += [1, -1]; bub.append(0.0); q += 1
            if np.isfinite(prices.grid_limit_kw):
                urows += [q, q]; ucols += [imp[t], over]; uvals += [1, -1]; bub.append(prices.grid_limit_kw); q += 1
        urows += [q, q]; ucols += [soc[-1], short]; uvals += [-1, -1]
        bub.append(-batt.soc_target * batt.energy_kwh); q += 1
        if s:                                      # non-anticipativity of the first hour
            for var, first in ((ch[0], 0), (dis[0], H)):
                rows += [r, r]; cols += [var, first]; vals += [1, -1]; beq.append(0.0); r += 1
    if risk_weight > 0:                            # CVaR of the peak: eta + mean excess / (1 - alpha)
        eta = S * per
        excess = eta + 1 + np.arange(S)
        lo[eta] = -np.inf
        c[eta] = prices.demand * risk_weight
        c[excess] = prices.demand * risk_weight / ((1.0 - alpha) * S)
        for s in range(S):
            peak = s * per + blocks * H
            urows += [q, q, q]; ucols += [peak, eta, excess[s]]; uvals += [1, -1, -1]; bub.append(0.0); q += 1
    A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n))
    A_ub = sparse.csr_matrix((uvals, (urows, ucols)), shape=(q, n))
    x, objective = solve_lp(c, A_ub, np.array(bub, float), A_eq, np.array(beq, float), lo, hi)
    X = x[:S * per].reshape(S, per)
    ch, dis = X[:, 0:H], X[:, H:2 * H]
    power = (dis - ch).mean(axis=0)                # positive = discharge
    peaks = X[:, blocks * H]
    return {"power_kw": power, "peak_kw": float(peaks.mean()), "peaks_kw": peaks, "cost": float(objective),
            "first_kw": float(dis[0, 0] - ch[0, 0])}
