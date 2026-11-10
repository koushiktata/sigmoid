"""The three demand tests from the case, with effect sizes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, ttest_ind

RUSH_HOURS = (7, 8, 9, 16, 17, 18, 19)
WEATHER = {1: "Clear", 2: "Mist / cloudy", 3: "Light rain or snow", 4: "Heavy rain or snow"}


def cohens_d(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    n1, n2 = len(left), len(right)
    var1, var2 = left.var(ddof=1), right.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((left.mean() - right.mean()) / pooled)


def effect_label(d: float) -> str:
    mag = abs(d)
    if mag < 0.2:
        return "negligible"
    if mag < 0.5:
        return "small"
    if mag < 0.8:
        return "medium"
    return "large"


def _group(series: pd.Series) -> dict:
    return {
        "n": int(series.shape[0]),
        "mean": round(float(series.mean()), 1),
        "median": round(float(series.median()), 1),
    }


def run_hypotheses(frame: pd.DataFrame) -> dict:
    working = frame.loc[frame["workingday"] == 1, "cnt"]
    off = frame.loc[frame["workingday"] == 0, "cnt"]
    u_stat, u_p = mannwhitneyu(working, off, alternative="two-sided")
    d_day = cohens_d(working.to_numpy(), off.to_numpy())

    rush_mask = frame["hr"].isin(RUSH_HOURS)
    rush = frame.loc[rush_mask, "cnt"]
    quiet = frame.loc[~rush_mask, "cnt"]
    t_stat, t_p = ttest_ind(rush, quiet, equal_var=False, alternative="greater")
    d_rush = cohens_d(rush.to_numpy(), quiet.to_numpy())

    groups = []
    summary = []
    clear = frame.loc[frame["weathersit"] == 1, "cnt"].to_numpy()
    pairwise = {}
    for code, name in WEATHER.items():
        sample = frame.loc[frame["weathersit"] == code, "cnt"]
        groups.append(sample.to_numpy())
        summary.append({"weather": name, **_group(sample)})
        if code != 1 and len(sample):
            pairwise[name] = {
                "cohens_d_vs_clear": round(cohens_d(clear, sample.to_numpy()), 3),
                "effect": effect_label(cohens_d(clear, sample.to_numpy())),
            }
    h_stat, h_p = kruskal(*groups)

    return {
        "alpha": 0.05,
        "workingday": {
            "name": "Working day vs non-working day",
            "test": "Mann-Whitney U",
            "h0": "Hourly counts have the same distribution on working and non-working days.",
            "statistic": round(float(u_stat), 1),
            "p_value": float(u_p),
            "cohens_d": round(d_day, 3),
            "effect": effect_label(d_day),
            "reject": bool(u_p < 0.05),
            "working": _group(working),
            "non_working": _group(off),
            "reading": (
                "The day types differ statistically, but the effect is negligible. "
                "What changes is the shape: two commute spikes on working days, one midday hump otherwise."
            ),
        },
        "rush": {
            "name": "Rush hours vs the rest of the day",
            "test": "Welch t-test, one-sided",
            "h0": "Rush hours (07–09 and 16–19) are not higher than other hours.",
            "statistic": round(float(t_stat), 2),
            "p_value": float(t_p),
            "cohens_d": round(d_rush, 3),
            "effect": effect_label(d_rush),
            "reject": bool(t_p < 0.05),
            "rush": _group(rush),
            "other": _group(quiet),
            "reading": (
                "Rush hour is a different operating regime, not a small fluctuation. "
                "Fleet and driver plans should be built around these windows."
            ),
        },
        "weather": {
            "name": "Weather categories",
            "test": "Kruskal-Wallis H",
            "h0": "Hourly demand is the same across weather categories.",
            "statistic": round(float(h_stat), 2),
            "p_value": float(h_p),
            "reject": bool(h_p < 0.05),
            "groups": summary,
            "pairwise_vs_clear": pairwise,
            "reading": (
                "Rain moves demand. Mist barely does. Heavy rain appears in only a handful of hours, "
                "so that average is not something to staff against."
            ),
        },
    }
