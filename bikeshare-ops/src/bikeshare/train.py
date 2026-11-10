"""Train the quantile transformer and the two XGBoost references.

Holdout matches the written report: last 20% of calendar dates, starting 2012-08-08.
The transformer is fit only on the first 70% so the next 10% can calibrate the 90% bound
without touching the holdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from bikeshare.data import (
    ARTIFACTS,
    FEATURE_NAMES,
    SEQ_LEN,
    build_sequences,
    chronological_cuts,
    engineer,
    load_hourly,
    quality_report,
    regression_metrics,
    scale_apply,
    scale_fit,
)
from bikeshare.model import QuantileTransformer, pinball

REPORT_CAT = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
REPORT_NUM = ["temp", "atemp", "hum", "windspeed"]
LAG_NUM = REPORT_NUM + ["lag_1", "lag_24", "lag_168", "roll_24"]


def _pipe(cats: list[str], nums: list[str]) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
            ("num", "passthrough", nums),
        ]
    )
    model = XGBRegressor(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    return Pipeline([("prep", prep), ("model", model)])


def _importance(pipe: Pipeline) -> list[dict]:
    names = pipe.named_steps["prep"].get_feature_names_out()
    scores = pipe.named_steps["model"].feature_importances_
    grouped: dict[str, float] = {}
    for name, score in zip(names, scores):
        raw = str(name).split("__", 1)[-1]
        if "_" in raw:
            head, tail = raw.rsplit("_", 1)
            key = head if tail.replace(".", "", 1).isdigit() else raw
        else:
            key = raw
        grouped[key] = grouped.get(key, 0.0) + float(score)
    total = sum(grouped.values()) or 1.0
    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    return [{"feature": key, "share": round(value / total, 4)} for key, value in ranked[:8]]


def _predict_torch(model, x, y_mean, y_std, batch=512) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    q50_parts, q90_parts = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            batch_x = torch.from_numpy(x[start : start + batch])
            q50, q90 = model(batch_x)
            q50_parts.append(q50.numpy())
            q90_parts.append(q90.numpy())
    q50 = np.concatenate(q50_parts) * y_std + y_mean
    q90 = np.concatenate(q90_parts) * y_std + y_mean
    return np.clip(q50, 0, None), np.clip(q90, 0, None)


def _conformal(y: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> dict:
    n = len(y)
    bias = float(np.median(y - q50))
    scores = np.sort(y - q90)
    rank = min(n - 1, int(np.ceil((n + 1) * 0.90)) - 1)
    adjustment = float(scores[rank])
    return {"bias_q50": bias, "q90_adjustment": adjustment, "calibration_n": int(n)}


def _apply_conformal(q50, q90, bias, adjustment):
    point = np.clip(q50 + bias, 0, None)
    upper = np.maximum(np.clip(q90 + adjustment, 0, None), point)
    return point, upper


def train() -> dict:
    torch.manual_seed(42)
    np.random.seed(42)
    raw = load_hourly()
    frame = engineer(raw)
    cuts = chronological_cuts(frame)
    sequences = build_sequences(frame)
    dates = sequences["dates"]
    train_m = dates <= np.datetime64(cuts["cal_end"])
    cal_m = (dates > np.datetime64(cuts["cal_end"])) & (
        dates <= np.datetime64(cuts["test_start_exclusive"])
    )
    test_m = dates > np.datetime64(cuts["test_start_exclusive"])

    x_train = sequences["x"][train_m]
    mean, std = scale_fit(x_train)
    y_train = sequences["y"][train_m]
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() or 1.0)

    def norm_y(y):
        return ((y - y_mean) / y_std).astype(np.float32)

    model = QuantileTransformer(n_features=len(FEATURE_NAMES), seq_len=SEQ_LEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    x_t = torch.from_numpy(scale_apply(x_train, mean, std))
    y_t = torch.from_numpy(norm_y(y_train))
    w_t = torch.from_numpy(sequences["rush_weight"][train_m])
    x_cal = scale_apply(sequences["x"][cal_m], mean, std)
    y_cal = sequences["y"][cal_m]

    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_rmse = float("inf")
    patience = 0
    history = []
    for epoch in range(1, 21):
        model.train()
        order = torch.randperm(len(x_t))
        total = 0.0
        seen = 0
        for start in range(0, len(order), 256):
            batch = order[start : start + 256]
            optimizer.zero_grad()
            q50, q90 = model(x_t[batch])
            loss = pinball(q50, y_t[batch], 0.5, w_t[batch]) + pinball(q90, y_t[batch], 0.9, w_t[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(batch)
            seen += len(batch)
        q50_cal, _ = _predict_torch(model, x_cal, y_mean, y_std)
        cal_rmse = float(np.sqrt(np.mean((q50_cal - y_cal) ** 2)))
        history.append({"epoch": epoch, "loss": round(total / seen, 4), "cal_rmse": round(cal_rmse, 2)})
        print(f"epoch {epoch:02d}  loss {total / seen:.4f}  cal RMSE {cal_rmse:.2f}", flush=True)
        if cal_rmse + 0.05 < best_rmse:
            best_rmse = cal_rmse
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)

    raw_q50_cal, raw_q90_cal = _predict_torch(model, x_cal, y_mean, y_std)
    conformal = _conformal(y_cal, raw_q50_cal, raw_q90_cal)
    covered = (y_cal <= (raw_q90_cal + conformal["q90_adjustment"])).mean()
    conformal["calibration_coverage"] = round(float(covered), 3)

    x_test = scale_apply(sequences["x"][test_m], mean, std)
    q50_test, q90_test = _predict_torch(model, x_test, y_mean, y_std)
    point, upper = _apply_conformal(
        q50_test, q90_test, conformal["bias_q50"], conformal["q90_adjustment"]
    )
    y_test = sequences["y"][test_m]
    rush = sequences["rush_weight"][test_m] > 1
    transformer_metrics = regression_metrics(y_test, point, rush)

    test_cut = cuts["test_start_exclusive"]
    report_train = frame[frame["dteday"] <= test_cut]
    report_test = frame[frame["dteday"] > test_cut]
    print("fitting report-style XGBoost", flush=True)
    report_model = _pipe(REPORT_CAT, REPORT_NUM)
    report_model.fit(report_train[REPORT_CAT + REPORT_NUM], report_train["cnt"])
    print("fitting lag XGBoost", flush=True)
    report_hat = np.clip(report_model.predict(report_test[REPORT_CAT + REPORT_NUM]), 0, None)

    lag_train = report_train.dropna(subset=LAG_NUM)
    lag_model = _pipe(REPORT_CAT, LAG_NUM)
    lag_model.fit(lag_train[REPORT_CAT + LAG_NUM], lag_train["cnt"])

    def stamp(value) -> str:
        return str(np.datetime64(value, "s"))

    test_ts = sequences["ts"][test_m]
    report_lookup = {stamp(ts): pred for ts, pred in zip(report_test["ts"], report_hat)}
    lag_frame = report_test.dropna(subset=LAG_NUM).copy()
    lag_hat = np.clip(lag_model.predict(lag_frame[REPORT_CAT + LAG_NUM]), 0, None)
    lag_lookup = {stamp(ts): pred for ts, pred in zip(lag_frame["ts"], lag_hat)}

    rows = []
    for i, ts in enumerate(test_ts):
        key = stamp(ts)
        if key not in report_lookup or key not in lag_lookup:
            continue
        rows.append(
            {
                "ts": key,
                "hr": int(sequences["hr"][test_m][i]),
                "workingday": int(sequences["workingday"][test_m][i]),
                "actual": round(float(y_test[i]), 2),
                "transformer": round(float(point[i]), 2),
                "q90": round(float(upper[i]), 2),
                "xgb_lags": round(float(lag_lookup[key]), 2),
                "xgb_report": round(float(report_lookup[key]), 2),
                "rush": bool(rush[i]),
            }
        )
    actual = np.array([row["actual"] for row in rows])
    rush_rows = np.array([row["rush"] for row in rows])
    metrics = {
        "transformer_q50": regression_metrics(actual, np.array([r["transformer"] for r in rows]), rush_rows),
        "xgb_lags": regression_metrics(actual, np.array([r["xgb_lags"] for r in rows]), rush_rows),
        "xgb_report_style": regression_metrics(actual, np.array([r["xgb_report"] for r in rows]), rush_rows),
    }
    best_name = min(metrics, key=lambda name: metrics[name]["rmse"])
    # Prefer the transformer when it is within 3 RMSE of the best tree.
    # It is the model that carries the calibrated peak bound.
    if metrics["transformer_q50"]["rmse"] <= metrics[best_name]["rmse"] + 3:
        production = "transformer_q50"
    else:
        production = best_name

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "y_mean": y_mean,
            "y_std": y_std,
            "conformal": conformal,
            "n_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
        },
        ARTIFACTS / "transformer.pt",
    )
    joblib.dump({"report": report_model, "lags": lag_model}, ARTIFACTS / "xgb.joblib")
    payload = {
        "dataset": quality_report(raw),
        "split": {
            "train_through": str(cuts["cal_end"].date()),
            "calibration_through": str(cuts["test_start_exclusive"].date()),
            "test_after": str(cuts["test_start_exclusive"].date()),
            "n_train": int(train_m.sum()),
            "n_calibration": int(cal_m.sum()),
            "n_test": int(len(rows)),
            "protocol": (
                "Test dates match the report (after the 80% date cutoff). "
                "The transformer trains on the first 70% of dates and calibrates on the next 10%."
            ),
        },
        "models": metrics,
        "production": production,
        "selection_rule": "Lowest holdout RMSE, with the transformer kept when it is within 3 RMSE of the leader.",
        "conformal": conformal,
        "training_curve": history,
        "importance_xgb_lags": _importance(lag_model),
        "report_baseline_published": {"model": "XGBoost no lags", "r2": 0.877, "rmse": 77.24, "mae": 53.01},
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(payload, indent=2))
    (ARTIFACTS / "evaluation.json").write_text(json.dumps(rows))
    print(json.dumps({"production": production, "models": metrics, "conformal": conformal}, indent=2))
    return payload


if __name__ == "__main__":
    train()
