"""Safe learning of the peak floor of the MPC over the months of a year: Gaussian process, SafeOpt, time-varying kernel.

Plant (from sweep.py): the saving s(x, m) in EUR of month m when the MPC uses the peak floor x, against the same MPC
without floor (x = 0, saving 0 by definition, placed at x = 115 kW in the model input, just below the lowest floor tried, because a floor of 115 kW
is below every monthly peak the MPC reaches). Every learner sees months m = 0, 1, ... in turn, may run `budget` experiments
in a month (one experiment = one value of the floor evaluated on that month, as in a lab or twin replay), and is
scored on what it evaluated: regret against the best floor of that month and number of unsafe experiments
(saving below -tol, i.e. more than `tol` EUR worse than the plain MPC). The safe seed is x = 115 kW (no floor):
its saving is 0 by definition, so it enters the GP as a known observation in every month.

Methods
  grid          every floor in random order, no model
  bo            GP-UCB on the floor, observations pooled over all months, no safety
  safeopt       SafeOpt on the floor, pooled over all months (the plant is assumed constant)
  safeopt_win   SafeOpt on the floor, only the last `window` months of data
  safeopt_st    SafeOpt with a space-time kernel Matern(x) * RBF(month): old months count less

Simplifications, deliberately: kernel hyperparameters are fixed, not learned; beta is a constant (2), not the
theoretical sequence, so there is no safety guarantee, only a measured count of unsafe experiments; the safety
threshold is a tolerated loss against the plain MPC's own saving, which a real site would have to estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED_X = 115.0


def savings_table(tab: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """tab: cost in EUR, rows floors (0 first), columns months. Returns x (floors, 0 mapped to SEED_X) and S[x, month]."""
    x = tab.index.to_numpy(float).copy()
    S = tab.loc[0.0].to_numpy()[None, :] - tab.to_numpy()
    x[x == 0.0] = SEED_X
    return x, S


class SpaceTime:
    """k((x, m), (x', m')) = sigma^2 Matern52(|x - x'| / ls_x) * exp(-(m - m')^2 / (2 ls_m^2)); ls_m = inf: no time."""

    def __init__(self, sigma=100.0, ls_x=25.0, ls_m=2.0):
        self.sigma, self.ls_x, self.ls_m = sigma, ls_x, ls_m

    def __call__(self, A, B):
        r = np.abs(A[:, None, 0] - B[None, :, 0]) / self.ls_x * np.sqrt(5.0)
        k = self.sigma**2 * (1 + r + r**2 / 3) * np.exp(-r)
        if np.isfinite(self.ls_m):
            k = k * np.exp(-0.5 * ((A[:, None, 1] - B[None, :, 1]) / self.ls_m) ** 2)
        return k


class GP:
    """Zero-mean Gaussian process with fixed hyperparameters and Gaussian observation noise."""

    def __init__(self, X, y, kernel, noise):
        self.X, self.kernel = X, kernel
        K = kernel(X, X) + (noise**2 + 1e-6) * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))

    def predict(self, Xq):
        Ks = self.kernel(Xq, self.X)
        mu = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = self.kernel.sigma**2 - (v**2).sum(axis=0)
        return mu, np.sqrt(np.maximum(var, 0.0))


@dataclass
class Result:
    method: str
    regret: float
    unsafe: int
    worse: int
    n_exp: int
    evals_to_5pct: float
    chosen: list


def run(x, S, method, budget=2, noise=0.0, seed=0, beta=2.0, window=3, ls_m=2.0, ls_x=60.0, sigma=None, tol=50.0):
    """Run one learner through all months. Returns Result."""
    rng = np.random.default_rng(seed)
    n_x, n_m = S.shape
    sigma = sigma if sigma is not None else 2.0 * tol        # prior amplitude of a monthly saving, set a priori, not from the table
    seed_idx = int(np.argmin(np.abs(x - SEED_X)))
    obs_X, obs_y = [], []              # [x, month]
    regret, unsafe, worse, n_exp, chosen = 0.0, 0, 0, 0, []
    to5 = []
    for m in range(n_m):
        best = S[:, m].max()
        obs_X.append([SEED_X, float(m)]); obs_y.append(0.0)      # the plain MPC saves nothing by definition: known
        hit = None
        pool = list(rng.permutation(n_x))  # for grid
        evaluated_this_month = set()
        for k in range(budget):
            if method == "grid":
                cand = [i for i in pool if i not in evaluated_this_month]
                i = cand[0] if cand else int(rng.integers(n_x))
            else:
                if method == "safeopt_st":
                    kern, use = SpaceTime(sigma, ls_x, ls_m), list(range(len(obs_X)))
                elif method == "safeopt_win":
                    kern, use = SpaceTime(sigma, ls_x, np.inf), [j for j, o in enumerate(obs_X) if o[1] > m - window]
                else:                   # bo, safeopt: pooled, no time
                    kern, use = SpaceTime(sigma, ls_x, np.inf), list(range(len(obs_X)))
                Xq = np.column_stack([x, np.full(n_x, float(m))])
                if use:
                    gp = GP(np.array([obs_X[j] for j in use]), np.array([obs_y[j] for j in use]), kern, max(noise, 1.0))
                    mu, sd = gp.predict(Xq)
                else:
                    mu, sd = np.zeros(n_x), np.full(n_x, sigma)
                lo, up = mu - beta * sd, mu + beta * sd
                if method == "bo":
                    i = int(np.argmax(up))
                else:
                    safe = lo >= -tol
                    safe[seed_idx] = True
                    if True:
                        lo_best = lo[safe].max()
                        maxim = safe & (up >= lo_best)
                        expand = np.zeros(n_x, bool)
                        for j in np.where(safe)[0]:      # would a (hypothetical optimistic) result at j make a neighbour safe?
                            if sd[j] < 1e-3:
                                continue
                            X2 = np.vstack([np.array([obs_X[q] for q in use]), Xq[j]])
                            y2 = np.append(np.array([obs_y[q] for q in use]), up[j])
                            gp2 = GP(X2, y2, kern, max(noise, 1.0))
                            mu2, sd2 = gp2.predict(Xq)
                            expand[j] = bool(((mu2 - beta * sd2 >= -tol) & ~safe).any())
                        cand = np.where(maxim | expand)[0]
                        width = up - lo
                        i = int(cand[np.argmax(width[cand])]) if len(cand) else int(np.argmax(np.where(safe, lo, -np.inf)))
            evaluated_this_month.add(i)
            y = S[i, m]
            regret += best - y
            unsafe += int(y < -tol)
            worse += int(y < 0)
            n_exp += 1
            chosen.append((m, float(x[i])))
            if i != seed_idx:        # the seed's saving is known (0); evaluating it teaches nothing
                obs_X.append([float(x[i]), float(m)])
                obs_y.append(float(y + noise * rng.standard_normal()))
            if hit is None and y >= 0.95 * best and best > 0:
                hit = k + 1
        to5.append(hit if hit is not None else budget + 1)
    return Result(method, regret, unsafe, worse, n_exp, float(np.mean(to5)), chosen)


def reference(x, S):
    """Fixed floors scored over the year: the plain MPC, the best single floor in hindsight, the 167 kW threshold."""
    tot = S.sum(axis=1)
    best_m = S.max(axis=0).sum()
    out = {"no floor (plain MPC)": 0.0, "best fixed floor in hindsight": float(tot.max()), "oracle floor per month": float(best_m)}
    out["floor at 165 kW"] = float(tot[np.argmin(np.abs(x - 165))])
    out["best fixed floor value (kW)"] = float(x[np.argmax(tot)])
    return out
