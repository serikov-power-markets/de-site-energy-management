"""State of charge estimation of a battery from current and voltage: coulomb counting against an extended
Kalman filter.

Cell-level equivalent circuit (scaled to the pack): open-circuit voltage OCV(s), series resistance R0 and one
RC branch (R1, C1):

    s[k+1]  = s[k] - eta I[k] dt / Q            (I > 0 discharges; eta applies while charging)
    v1[k+1] = a v1[k] + R1 (1 - a) I[k],        a = exp(-dt / (R1 C1))
    U[k]    = OCV(s[k]) - R0 I[k] - v1[k]

The current sensor has a constant offset and noise, the voltage sensor noise, and the filter's model parameters
differ from the true ones. The filter state is (s, v1, current offset). All parameters are illustrative values of
a lithium-ion cell with a sloped open-circuit voltage; a flat curve (LFP) would make the voltage far less
informative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# open-circuit voltage of a cell with a sloped curve (V against state of charge)
SOC_GRID = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
OCV_GRID = np.array([3.00, 3.35, 3.45, 3.55, 3.62, 3.68, 3.74, 3.82, 3.90, 3.98, 4.06, 4.12, 4.18])


def ocv(s):
    return np.interp(s, SOC_GRID, OCV_GRID)


def docv(s: float, h: float = 0.01) -> float:
    lo, hi = max(s - h, 0.0), min(s + h, 1.0)
    return float((ocv(hi) - ocv(lo)) / (hi - lo))


@dataclass(frozen=True)
class Cell:
    capacity_ah: float = 100.0
    r0: float = 0.0015           # ohm
    r1: float = 0.0010
    c1: float = 60_000.0         # F, time constant r1 c1 = 60 s
    eta: float = 0.98            # coulombic efficiency of charging


@dataclass(frozen=True)
class Sensors:
    current_offset: float = 0.4      # A (0.4% of the 1C current)
    current_noise: float = 0.5       # A, standard deviation
    voltage_noise: float = 0.003     # V


def simulate(current: np.ndarray, s0: float, dt: float, cell: Cell, sensors: Sensors, seed: int = 0) -> dict:
    """True state and measurements for a current profile (A, positive = discharge) sampled every dt seconds."""
    rng = np.random.default_rng(seed)
    n = len(current)
    s, v1 = np.empty(n), np.empty(n)
    s_k, v_k = s0, 0.0
    a = np.exp(-dt / (cell.r1 * cell.c1))
    applied = np.array(current, dtype=float)
    for k in range(n):
        i = applied[k]
        if (s_k <= 0.0 and i > 0) or (s_k >= 1.0 and i < 0):     # the battery management stops at the limits
            i = applied[k] = 0.0
        s[k], v1[k] = s_k, v_k
        factor = cell.eta if i < 0 else 1.0
        s_k = float(np.clip(s_k - factor * i * dt / (cell.capacity_ah * 3600.0), 0.0, 1.0))
        v_k = a * v_k + cell.r1 * (1 - a) * i
    voltage = ocv(s) - cell.r0 * applied - v1
    return {"soc": s, "current_meas": applied + sensors.current_offset + sensors.current_noise * rng.standard_normal(n),
            "voltage_meas": voltage + sensors.voltage_noise * rng.standard_normal(n), "current_true": applied}


def coulomb_count(current_meas: np.ndarray, s0: float, dt: float, cell: Cell) -> np.ndarray:
    factor = np.where(current_meas < 0, cell.eta, 1.0)
    steps = factor * current_meas * dt / (cell.capacity_ah * 3600.0)
    return np.clip(s0 - np.concatenate([[0.0], np.cumsum(steps[:-1])]), 0.0, 1.0)


class KalmanSoC:
    """Extended Kalman filter on (state of charge, RC voltage, current sensor offset), one sample at a time."""

    def __init__(self, s0: float, dt: float, cell: Cell, sensors: Sensors, estimate_offset: bool = True,
                 s0_std: float = 0.1, model_error_v: float = 0.005):
        self.dt, self.cell = dt, cell
        self.a = float(np.exp(-dt / (cell.r1 * cell.c1)))
        self.x = np.array([s0, 0.0, 0.0])
        self.P = np.diag([s0_std ** 2, 1e-6, 1.0 if estimate_offset else 0.0])
        self.Q = np.diag([(sensors.current_noise * dt / (cell.capacity_ah * 3600.0)) ** 2 + 1e-12,
                          (cell.r1 * (1 - self.a) * sensors.current_noise) ** 2 + 1e-10,
                          1e-8 if estimate_offset else 0.0])
        self.R = sensors.voltage_noise ** 2 + model_error_v ** 2

    def update(self, current_meas: float, voltage_meas: float) -> tuple[float, float, float]:
        """Use one sample (current and voltage at the same instant), then predict to the next instant.
        Returns the state of charge estimate at the sample, its standard deviation and the offset estimate."""
        s, v1, b = self.x
        H = np.array([docv(s), -1.0, self.cell.r0])
        h = ocv(s) - self.cell.r0 * (current_meas - b) - v1
        S = H @ self.P @ H + self.R
        K = self.P @ H / S
        x = self.x + K * (voltage_meas - h)
        x[0] = min(max(x[0], 0.0), 1.0)
        I_KH = np.eye(3) - np.outer(K, H)
        P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * self.R     # Joseph form keeps P positive
        estimate = float(x[0]), float(np.sqrt(max(P[0, 0], 0.0))), float(x[2])
        s, v1, b = x
        i = current_meas - b
        eta = self.cell.eta if i < 0 else 1.0
        g = eta * self.dt / (self.cell.capacity_ah * 3600.0)
        self.x = np.array([min(max(s - g * i, 0.0), 1.0), self.a * v1 + self.cell.r1 * (1 - self.a) * i, b])
        F = np.array([[1.0, 0.0, g], [0.0, self.a, -self.cell.r1 * (1 - self.a)], [0.0, 0.0, 1.0]])
        self.P = F @ P @ F.T + self.Q
        return estimate


def ekf(current_meas: np.ndarray, voltage_meas: np.ndarray, s0: float, dt: float, cell: Cell, sensors: Sensors,
        estimate_offset: bool = True, s0_std: float = 0.1, model_error_v: float = 0.005) -> dict:
    """The filter over a whole record: state of charge, its standard deviation and the offset estimate."""
    f = KalmanSoC(s0, dt, cell, sensors, estimate_offset, s0_std, model_error_v)
    out = np.array([f.update(i, u) for i, u in zip(current_meas, voltage_meas)])
    return {"soc": out[:, 0], "std": out[:, 1], "offset": out[:, 2]}


class CellPlant:
    """The true cell, advanced one sample at a time; returns noisy current and voltage readings."""

    def __init__(self, s0: float, dt: float, cell: Cell, sensors: Sensors, seed: int = 0):
        self.s, self.v1, self.dt, self.cell, self.sensors = s0, 0.0, dt, cell, sensors
        self.a = float(np.exp(-dt / (cell.r1 * cell.c1)))
        self.rng = np.random.default_rng(seed)

    def step(self, current: float) -> tuple[float, float]:
        """Apply the current (A, positive = discharge) for one sample; readings belong to the start of the sample."""
        voltage = ocv(self.s) - self.cell.r0 * current - self.v1
        meas = (current + self.sensors.current_offset + self.sensors.current_noise * self.rng.standard_normal(),
                voltage + self.sensors.voltage_noise * self.rng.standard_normal())
        factor = self.cell.eta if current < 0 else 1.0
        self.s = min(max(self.s - factor * current * self.dt / (self.cell.capacity_ah * 3600.0), 0.0), 1.0)
        self.v1 = self.a * self.v1 + self.cell.r1 * (1 - self.a) * current
        return meas
