"""German network charges for metered industrial customers, as rules that a battery plan has to respect.

1. Annual capacity-price system with two price tiers split at 2,500 hours of utilisation
   (Benutzungsdauer = annual energy / annual peak). Below the threshold the capacity price is low and the energy
   price high, above it the reverse. Shaving a peak raises the utilisation time and can move a site into the other
   tier, so the saving is not linear in the peak.
2. Atypical grid use, section 19(2) sentence 1 StromNEV (BNetzA decision BK4-13-739): a site whose peak inside the
   network operator's high-load windows stays clearly below its annual peak pays the capacity price on the window
   peak. Conditions: a relative difference of at least 20% on the medium-voltage level (30% at MV/LV and LV), an
   absolute difference of at least 100 kW and a saving of at least 500 EUR. Windows apply on working days only.

The prices are illustrative assumptions in the range of published distribution price sheets, not one operator's
sheet. Thresholds and window rules follow the published decision as quoted on operators' pages; the window hours
are those one operator (N-ERGIE) published for medium voltage for 2026, used here only as an example.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"
THRESHOLD_HOURS = 2500.0
SIGNIFICANCE = {"HS": 0.10, "HS/MS": 0.20, "MS": 0.20, "MS/NS": 0.30, "NS": 0.30}
MIN_SHIFT_KW = 100.0
MIN_SAVING_EUR = 500.0


@dataclass(frozen=True)
class Tariff:
    """Network charge: capacity price in EUR per kW and year, energy price in EUR per kWh."""
    low_capacity: float = 20.0       # utilisation below 2,500 h
    low_energy: float = 0.060
    high_capacity: float = 130.0     # 2,500 h and more
    high_energy: float = 0.016
    other_per_kwh: float = 0.065     # levies, taxes and concession fee, the same in both tiers (assumption)

    def prices(self, high: bool) -> tuple[float, float]:
        return (self.high_capacity, self.high_energy) if high else (self.low_capacity, self.low_energy)


def utilisation_hours(import_kwh: float, peak_kw: float, hours: int) -> float:
    """Annual utilisation time; a period shorter than a year is scaled to 8,760 hours."""
    if peak_kw <= 0:
        return 0.0
    return import_kwh / peak_kw * 8760.0 / hours


def network_charge(imports: pd.Series, tariff: Tariff) -> dict:
    """Annual network charge of an hourly import series (kW), choosing the tier by the utilisation time."""
    energy = float(imports.sum())
    peak = float(imports.max())
    hours = len(imports)
    t = utilisation_hours(energy, peak, hours)
    high = t >= THRESHOLD_HOURS
    cap, work = tariff.prices(high)
    capacity = cap * peak * hours / 8760.0
    return {"utilisation_h": t, "tier": ">=2500 h" if high else "<2500 h", "peak_kw": peak,
            "capacity_eur": capacity, "energy_eur": work * energy, "other_eur": tariff.other_per_kwh * energy,
            "network_eur": capacity + (work + tariff.other_per_kwh) * energy}


# ------------------------------------------------------------------ atypical grid use
# High-load windows (local start and end) of one operator's medium-voltage level for 2026, by season.
WINDOWS_MS = {
    "winter": (("07:45", "09:00"), ("16:30", "19:15"), ("19:30", "19:45")),   # December to February
    "spring": (),                                                             # March to May
    "summer": (),                                                             # June to August
    "autumn": (("16:45", "18:45"),),                                          # September to November
}
SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
          6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def high_load_mask(index: pd.DatetimeIndex, holidays=(), windows: dict = WINDOWS_MS) -> np.ndarray:
    """True for hours that overlap a high-load window on a working day (hourly data: any overlap counts)."""
    local = index.tz_convert(TZ)
    holidays = set(holidays)
    working = np.array([d.weekday() < 5 and d not in holidays
                        and not (d.month == 12 and d.day >= 24) and not (d.month == 1 and d.day == 1)
                        for d in local.date])
    start = local.hour * 60 + local.minute
    out = np.zeros(len(index), dtype=bool)
    for season, spans in windows.items():
        in_season = np.array([SEASON[m] == season for m in local.month])
        for a, b in spans:
            lo = int(a[:2]) * 60 + int(a[3:])
            hi = int(b[:2]) * 60 + int(b[3:])
            out |= in_season & (start < hi) & (start + 60 > lo)
    return out & working


def atypical_use(imports: pd.Series, tariff: Tariff, level: str = "MS", holidays=()) -> dict:
    """Eligibility and saving of the individual charge for one year of hourly imports."""
    base = network_charge(imports, tariff)
    mask = high_load_mask(imports.index, holidays)
    peak = base["peak_kw"]
    inside = float(imports[mask].max()) if mask.any() else 0.0
    shift = peak - inside
    cap_price = tariff.prices(base["tier"] == ">=2500 h")[0]
    saving = cap_price * shift * len(imports) / 8760.0
    eligible = shift >= MIN_SHIFT_KW and peak > 0 and shift / peak >= SIGNIFICANCE[level] and saving >= MIN_SAVING_EUR
    return {"window_peak_kw": inside, "annual_peak_kw": peak, "shift_kw": shift,
            "shift_share": shift / peak if peak > 0 else 0.0, "eligible": bool(eligible),
            "saving_eur": float(saving) if eligible else 0.0,
            "network_eur": base["network_eur"] - (float(saving) if eligible else 0.0)}
