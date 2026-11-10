"""Turn an hourly demand bound into a system-wide fleet estimate."""

from __future__ import annotations

import math

CASE_FLEET = 2000


def estimate_fleet(
    peak_hourly: float,
    p95_hourly: float,
    trip_hours: float = 0.5,
    buffer: float = 0.15,
    maintenance: float = 0.10,
) -> dict:
    """Bikes on the street ≈ rentals in the hour × trip length, plus reserves.

    Trip length is not in the dataset. The default of 30 minutes is an assumption
    the operator can change. The estimate is system-wide, not per station.
    """

    def _one(level: float) -> dict:
        concurrent = level * trip_hours
        fleet = int(math.ceil(concurrent * (1 + buffer) * (1 + maintenance)))
        return {
            "hourly_rentals": round(float(level), 1),
            "bikes_in_use": round(float(concurrent), 1),
            "fleet": fleet,
        }

    peak = _one(peak_hourly)
    typical_busy = _one(p95_hourly)
    return {
        "trip_hours": trip_hours,
        "buffer": buffer,
        "maintenance": maintenance,
        "peak": peak,
        "p95": typical_busy,
        "case_study_fleet": CASE_FLEET,
        "gap_vs_case_fleet": CASE_FLEET - peak["fleet"],
        "note": (
            "This sizes the whole city, not a dock. It uses completed rentals, so unmet "
            "demand is invisible. Compare it with the case-study fleet of about 2,000 before "
            "buying bikes. Rebalancing still needs station inventory."
        ),
    }


def verdict(fleet_at_peak: int) -> str:
    if fleet_at_peak < 1000:
        return (
            f"The holdout planning fleet is {fleet_at_peak} bikes, well under the "
            f"case-study fleet of about {CASE_FLEET}. System-wide volume does not by itself "
            "justify buying toward 2,000. Put the bikes you have into the weekday commute "
            "peaks first. Buy more only if empty docks remain after that, and only once "
            "station inventory says where."
        )
    return (
        f"The holdout planning fleet is {fleet_at_peak} bikes. That is close enough to "
        "operating scale that a purchase needs station-level proof, not a citywide average."
    )
