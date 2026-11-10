"""Load Capital Bikeshare hourly rentals and build causal sequence features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "hour.csv"
ARTIFACTS = ROOT / "artifacts"
SEQ_LEN = 24
RUSH_HOURS = (7, 8, 9, 16, 17, 18, 19)

SEASON_NAMES = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
WEATHER_NAMES = {
    1: "Clear",
    2: "Mist / cloudy",
    3: "Light rain or snow",
    4: "Heavy rain or snow",
}
WEEKDAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

BASE_FEATURES = [
    "season_n",
    "yr",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "holiday",
    "weekday_sin",
    "weekday_cos",
    "workingday",
    "weather_n",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "rush",
    "weekend",
    "temp_hum",
    "lag_1",
    "lag_24",
    "lag_168",
    "roll_24",
    "observed_cnt",
]
FEATURE_NAMES = BASE_FEATURES + ["is_target"]

RAW_COLUMNS = [
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
]


def load_hourly(path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or DATA_PATH)
    frame["dteday"] = pd.to_datetime(frame["dteday"])
    frame["ts"] = frame["dteday"] + pd.to_timedelta(frame["hr"].astype(int), unit="h")
    frame = frame.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return frame


def _lag(counts: pd.Series, hours: int) -> pd.Series:
    shifted = counts.copy()
    shifted.index = counts.index + pd.Timedelta(hours=hours)
    return shifted


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar cycles and lags that are known before the target hour."""
    out = frame.copy()
    counts = out.set_index("ts")["cnt"].sort_index()
    counts = counts[~counts.index.duplicated(keep="last")]
    out["lag_1"] = _lag(counts, 1).reindex(out["ts"]).to_numpy()
    out["lag_24"] = _lag(counts, 24).reindex(out["ts"]).to_numpy()
    out["lag_168"] = _lag(counts, 168).reindex(out["ts"]).to_numpy()
    regular = counts.reindex(pd.date_range(counts.index.min(), counts.index.max(), freq="h"))
    rolled = regular.shift(1).rolling(24, min_periods=8).mean()
    out["roll_24"] = rolled.reindex(out["ts"]).to_numpy()
    out["observed_cnt"] = out["cnt"].astype(float)
    return finalize_features(out)


def finalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    hour = out["hr"].astype(float)
    month = out["mnth"].astype(float)
    weekday = out["weekday"].astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    out["season_n"] = out["season"].astype(float) / 4.0
    out["weather_n"] = out["weathersit"].astype(float) / 4.0
    out["holiday"] = out["holiday"].astype(float)
    out["workingday"] = out["workingday"].astype(float)
    out["yr"] = out["yr"].astype(float)
    out["rush"] = hour.isin(RUSH_HOURS).astype(float)
    out["weekend"] = weekday.isin([0, 6]).astype(float)
    out["temp_hum"] = out["temp"].astype(float) * out["hum"].astype(float)
    out["observed_cnt"] = out["observed_cnt"].astype(float)
    for col in ("lag_1", "lag_24", "lag_168", "roll_24", "temp", "atemp", "hum", "windspeed"):
        out[col] = out[col].astype(float)
    return out


def quality_report(raw: pd.DataFrame) -> dict:
    present = [col for col in RAW_COLUMNS if col in raw.columns]
    return {
        "rows": int(len(raw)),
        "start": str(raw["dteday"].min().date()),
        "end": str(raw["dteday"].max().date()),
        "missing_cells": int(raw[present].isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_hours": int(raw["ts"].duplicated().sum()) if "ts" in raw.columns else 0,
        "negative_counts": int((raw["cnt"] < 0).sum()),
        "count_mismatches": int((raw["cnt"] != raw["casual"] + raw["registered"]).sum()),
        "heavy_weather_hours": int((raw["weathersit"] == 4).sum()),
        "record_cnt": int(raw["cnt"].max()),
        "record_at": str(raw.loc[raw["cnt"].idxmax(), "ts"]),
        "mean_cnt": round(float(raw["cnt"].mean()), 2),
    }


def chronological_cuts(frame: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Match the report's 80/20 future holdout, and hold out the prior 10% to calibrate."""
    dates = np.sort(frame["dteday"].drop_duplicates().to_numpy())
    test_idx = int(0.8 * (len(dates) - 1))
    cal_idx = int(0.7 * (len(dates) - 1))
    return {
        "cal_end": pd.Timestamp(dates[cal_idx]),
        "test_start_exclusive": pd.Timestamp(dates[test_idx]),
    }


def build_sequences(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Windows of 24 consecutive hours. The last hour's own count is masked."""
    base = frame[BASE_FEATURES].to_numpy(np.float32)
    hour_id = frame["ts"].astype("int64").to_numpy() // 3_600_000_000_000
    obs_i = BASE_FEATURES.index("observed_cnt")
    windows = []
    ends = []
    for end in range(SEQ_LEN - 1, len(frame)):
        start = end - (SEQ_LEN - 1)
        if hour_id[end] - hour_id[start] != SEQ_LEN - 1:
            continue
        window = base[start : end + 1].copy()
        if not np.isfinite(window).all():
            continue
        window[-1, obs_i] = 0.0
        flag = np.zeros((SEQ_LEN, 1), np.float32)
        flag[-1, 0] = 1.0
        windows.append(np.concatenate([window, flag], axis=1))
        ends.append(end)
    ends_arr = np.asarray(ends, dtype=np.int64)
    rush = frame["hr"].isin(RUSH_HOURS).to_numpy()[ends_arr] & (
        frame["workingday"].to_numpy()[ends_arr] == 1
    )
    return {
        "x": np.stack(windows).astype(np.float32),
        "y": frame["cnt"].to_numpy(np.float32)[ends_arr],
        "ends": ends_arr,
        "ts": frame["ts"].to_numpy()[ends_arr],
        "dates": frame["dteday"].to_numpy()[ends_arr],
        "hr": frame["hr"].to_numpy()[ends_arr],
        "workingday": frame["workingday"].to_numpy()[ends_arr],
        "rush_weight": np.where(rush, 2.0, 1.0).astype(np.float32),
    }


def scale_fit(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def scale_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, rush: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot else 0.0
    mask = y_true > 0
    mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100) if mask.any() else None
    if rush.any():
        peak_err = y_pred[rush] - y_true[rush]
        peak_rmse = float(np.sqrt(np.mean(peak_err**2)))
        peak_bias = float(np.mean(y_true[rush] - y_pred[rush]))
    else:
        peak_rmse, peak_bias = None, None
    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "mape_pct": None if mape is None else round(mape, 1),
        "peak_rmse": None if peak_rmse is None else round(peak_rmse, 2),
        "peak_bias": None if peak_bias is None else round(peak_bias, 2),
    }
