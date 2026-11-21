"""Gaussian process, SafeOpt and the baselines on a synthetic plant with a known answer."""
import numpy as np
import pandas as pd
import pytest

from siteems import safeopt


def synthetic(drift=0.0, n_m=12, width=35.0):
    """Saving of a month: a smooth bump in the floor with its maximum at 160 + drift*m kW, zero at the seed (115 kW)."""
    x = np.array([115.0] + list(range(120, 205, 5)))
    S = np.array([[200.0 * np.exp(-(((xi - (160 + drift * m)) / width) ** 2)) - 20.0 for m in range(n_m)] for xi in x])
    S[0, :] = 0.0                                  # the seed: plain MPC
    return x, S


def test_gp_interpolates_noise_free_data_and_has_no_variance_there():
    k = safeopt.SpaceTime(sigma=50.0, ls_x=20.0, ls_m=np.inf)
    X = np.array([[120.0, 0.0], [160.0, 0.0], [190.0, 0.0]])
    y = np.array([10.0, 40.0, -5.0])
    gp = safeopt.GP(X, y, k, noise=0.0)
    mu, sd = gp.predict(X)
    assert mu == pytest.approx(y, abs=1e-3)
    assert sd.max() < 0.1
    mu2, sd2 = gp.predict(np.array([[140.0, 0.0], [200.0, 0.0]]))
    assert (sd2 > sd.max()).all()


def test_space_time_kernel_forgets_old_months():
    k = safeopt.SpaceTime(sigma=1.0, ls_x=20.0, ls_m=2.0)
    a, b, c = np.array([[150.0, 0.0]]), np.array([[150.0, 1.0]]), np.array([[150.0, 8.0]])
    assert k(a, a)[0, 0] == pytest.approx(1.0)
    assert k(a, b)[0, 0] > 0.8
    assert k(a, c)[0, 0] < 1e-3


def test_savings_table_puts_the_plain_mpc_at_the_seed_with_zero_saving():
    tab = pd.DataFrame({"2016-06": [100.0, 90.0, 95.0], "2016-07": [200.0, 150.0, 210.0]}, index=[0.0, 120.0, 130.0])
    x, S = safeopt.savings_table(tab)
    assert x[0] == safeopt.SEED_X
    assert S[0].tolist() == [0.0, 0.0]
    assert S[1].tolist() == [10.0, 50.0] and S[2].tolist() == [5.0, -10.0]


def test_safeopt_never_leaves_the_safe_set_on_a_known_plant():
    x, S = synthetic()
    r = safeopt.run(x, S, "safeopt", budget=3, noise=0.0, seed=0)
    assert r.unsafe == 0
    # it learns: far less regret than staying at the seed all year (regret of the seed = best saving each month)
    assert r.regret < 0.4 * S.max(axis=0).sum()


def test_bo_without_safety_tries_unsafe_floors_and_safeopt_does_not():
    x, S = synthetic()
    S[x > 190, :] -= 400.0                                  # the highest floors lose money
    bo = safeopt.run(x, S, "bo", budget=3, noise=0.0, seed=0)
    so = safeopt.run(x, S, "safeopt", budget=3, noise=0.0, seed=0)
    assert so.unsafe == 0
    assert bo.unsafe >= so.unsafe


def test_time_kernel_follows_a_drifting_optimum_better_than_pooling():
    x, S = synthetic(drift=8.0, n_m=10, width=25.0)        # narrow bump whose optimum moves from 160 to 232 kW (clipped by the grid)
    res = {m: np.mean([safeopt.run(x, S, m, budget=2, noise=2.0, seed=s, ls_m=3.0).regret for s in range(5)])
           for m in ("safeopt", "safeopt_st")}
    assert res["safeopt_st"] < res["safeopt"]


def test_reference_values():
    x, S = synthetic()
    ref = safeopt.reference(x, S)
    assert ref["oracle floor per month"] >= ref["best fixed floor in hindsight"] >= ref["floor at 165 kW"] - 1e-9
    assert ref["best fixed floor value (kW)"] == 160.0
