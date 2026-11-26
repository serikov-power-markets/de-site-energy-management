"""On-site controller: carry out the plan of the optimiser, react to deviations and fail safely.

Every control step the controller receives the plan (a battery set point and the planned import peak), whatever
measurements are available and the state of charge estimate with its standard deviation. It chooses a mode,
computes a set point and passes it through a safety filter. The filter is the only place where limits are
enforced, so they hold in every mode:

    battery power           |P| <= rating
    state of charge         the energy left after the step stays within the limits, with a margin of k standard
                            deviations of the estimate
    grid connection         import <= limit, as far as the battery can reach it
    feed-in                 export <= limit; surplus PV beyond it is curtailed

Positive battery power discharges. All powers in kW, energies in kWh, step length in hours.
"""

from __future__ import annotations

from dataclasses import dataclass

NORMAL, NO_PLAN, NO_METER, HOLD, STOP = "normal", "no_plan", "no_meter", "hold_plan", "stop"


@dataclass(frozen=True)
class Limits:
    power_kw: float
    energy_kwh: float
    soc_min: float = 0.1
    soc_max: float = 1.0
    eta: float = 0.95                 # one way
    grid_import_kw: float = float("inf")
    feed_in_kw: float = float("inf")
    margin_sigma: float = 2.0
    plan_max_age_h: int = 24


@dataclass(frozen=True)
class Plan:
    power_kw: float                    # planned battery power, positive = discharge
    peak_kw: float                     # planned import peak of the billing period
    age_h: int = 0                     # hours since the optimiser made it


@dataclass(frozen=True)
class Measurement:
    load_kw: float | None              # None: the meter delivers nothing
    pv_kw: float | None
    soc: float | None                  # estimate, None if the battery management reports a fault
    soc_std: float = 0.0


def mode(plan: Plan | None, meas: Measurement, limits: Limits) -> str:
    if meas.soc is None:
        return STOP
    if meas.load_kw is None or meas.pv_kw is None:
        return NO_METER if plan is None or plan.age_h > limits.plan_max_age_h else HOLD
    if plan is None or plan.age_h > limits.plan_max_age_h:
        return NO_PLAN
    return NORMAL


def soc_bounds(meas: Measurement, limits: Limits, dt: float = 1.0) -> tuple[float, float]:
    """(most negative, most positive) battery power that keeps the true state of charge inside its limits when
    the estimate is wrong by up to margin_sigma standard deviations."""
    margin = limits.margin_sigma * meas.soc_std
    stored = limits.energy_kwh * max(meas.soc - margin - limits.soc_min, 0.0)
    room = limits.energy_kwh * max(limits.soc_max - meas.soc - margin, 0.0)
    return -min(limits.power_kw, room / (limits.eta * dt)), min(limits.power_kw, stored * limits.eta / dt)


def safety_filter(request_kw: float, meas: Measurement, limits: Limits, dt: float = 1.0) -> tuple[float, float]:
    """(battery power, PV curtailment) closest to the request that respects every limit the battery can control."""
    if meas.soc is None:
        return 0.0, 0.0
    low, high = soc_bounds(meas, limits, dt)
    power = min(max(request_kw, low), high)
    curtail = 0.0
    if meas.load_kw is not None and meas.pv_kw is not None:
        net = meas.load_kw - meas.pv_kw                    # import without the battery
        # grid connection: discharge at least what keeps the import within the limit
        power = min(max(power, net - limits.grid_import_kw), high)
        # charging must not push the import over the limit
        if power < 0:
            power = max(power, min(net - limits.grid_import_kw, 0.0))
        # feed-in: first stop discharging or charge more, then curtail PV
        excess = -(net - power) - limits.feed_in_kw
        if excess > 0:
            power = max(power - excess, low)
            curtail = min(max(-(net - power) - limits.feed_in_kw, 0.0), meas.pv_kw)
    return power, curtail


def step(plan: Plan | None, meas: Measurement, limits: Limits, threshold_kw: float, dt: float = 1.0) -> dict:
    """One control step. threshold_kw: import peak learnt from history, used when there is no usable plan."""
    m = mode(plan, meas, limits)
    if m == STOP:
        request = 0.0
    elif m == NORMAL:
        net = meas.load_kw - meas.pv_kw
        request = plan.power_kw
        cap = max(plan.peak_kw, 0.0)
        if net - request > cap:                    # peak guard: shave what the plan did not foresee
            request = net - cap
    elif m == NO_PLAN:                             # rule: shave above the threshold, recharge well below it
        net = meas.load_kw - meas.pv_kw
        if net > threshold_kw:
            request = net - threshold_kw
        else:
            request = -min(max(threshold_kw - 20.0 - net, 0.0), limits.power_kw)
    elif m == HOLD:                                # meter lost: follow the plan, without the guard
        request = plan.power_kw
    else:                                          # no meter and no plan: do nothing
        request = 0.0
    power, curtail = safety_filter(request, meas, limits, dt)
    return {"mode": m, "request_kw": request, "power_kw": power, "curtail_kw": curtail}


def apply_to_battery(soc_true: float, power_kw: float, limits: Limits, dt: float = 1.0) -> tuple[float, float]:
    """The physical battery: returns (new true state of charge, power actually delivered). The battery management
    cuts the power at the hard limits."""
    if power_kw > 0:
        power_kw = min(power_kw, limits.power_kw, max(soc_true - 0.0, 0.0) * limits.energy_kwh * limits.eta / dt)
        soc_true -= power_kw * dt / limits.eta / limits.energy_kwh
    else:
        power_kw = -min(-power_kw, limits.power_kw, max(1.0 - soc_true, 0.0) * limits.energy_kwh / (limits.eta * dt))
        soc_true -= power_kw * dt * limits.eta / limits.energy_kwh
    return min(max(soc_true, 0.0), 1.0), power_kw
