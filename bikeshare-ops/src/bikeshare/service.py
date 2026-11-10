"""Loaded models and the forecast / fleet operations the API calls."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch

from bikeshare.data import (
    ARTIFACTS,
    FEATURE_NAMES,
    SEASON_NAMES,
    SEQ_LEN,
    WEATHER_NAMES,
    engineer,
    finalize_features,
    load_hourly,
    quality_report,
)
from bikeshare.fleet import estimate_fleet, verdict
from bikeshare.hypotheses import run_hypotheses
from bikeshare.model import QuantileTransformer

SEASON_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}


class OpsService:
    def __init__(self):
        ckpt_path = ARTIFACTS / "transformer.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError("Missing artifacts/transformer.pt. Run: PYTHONPATH=src python -m bikeshare.train")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.mean = ckpt["mean"]
        self.std = ckpt["std"]
        self.y_mean = float(ckpt["y_mean"])
        self.y_std = float(ckpt["y_std"])
        self.conformal = ckpt["conformal"]
        self.model = QuantileTransformer(n_features=ckpt["n_features"], seq_len=SEQ_LEN)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        trees = joblib.load(ARTIFACTS / "xgb.joblib")
        self.report_model = trees["report"]
        self.lag_model = trees["lags"]
        self.metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
        self.evaluation = json.loads((ARTIFACTS / "evaluation.json").read_text())
        raw = load_hourly()
        self.frame = engineer(raw)
        self.quality = quality_report(raw)
        self.hypotheses = run_hypotheses(raw)
        self.typical = (
            self.frame.groupby(["workingday", "hr"])
            .agg(
                cnt=("cnt", "median"),
                temp=("temp", "median"),
                atemp=("atemp", "median"),
                hum=("hum", "median"),
                windspeed=("windspeed", "median"),
                weathersit=("weathersit", "median"),
            )
            .reset_index()
        )
        self._profile = self._build_profile()

    def _build_profile(self) -> dict:
        grouped = self.frame.groupby(["workingday", "hr"], as_index=False)["cnt"].mean().round(1)
        by_hour = [
            {"workingday": int(row.workingday), "hr": int(row.hr), "cnt": float(row.cnt)}
            for row in grouped.itertuples(index=False)
        ]
        by_season = {
            SEASON_NAMES[int(season)]: float(round(value, 1))
            for season, value in self.frame.groupby("season")["cnt"].mean().items()
        }
        by_weather = {
            WEATHER_NAMES[int(code)]: float(round(value, 1))
            for code, value in self.frame.groupby("weathersit")["cnt"].mean().items()
        }
        return {"by_hour": by_hour, "by_season": by_season, "by_weather": by_weather}

    def overview(self) -> dict:
        prod = self.metrics["production"]
        return {
            "quality": self.quality,
            "split": self.metrics["split"],
            "production": prod,
            "headline": self.metrics["models"][prod],
            "models": self.metrics["models"],
            "profile": self._profile,
            "verdict": self.decision_text(),
            "importance": self.metrics["importance_xgb_lags"],
            "conformal": self.metrics["conformal"],
        }

    def _typical_row(self, workingday: int, hour: int) -> pd.Series:
        hit = self.typical[(self.typical["workingday"] == workingday) & (self.typical["hr"] == hour)]
        if hit.empty:
            hit = self.typical[self.typical["hr"] == hour]
        return hit.iloc[0]

    def _window(self, slots: dict) -> pd.DataFrame:
        rows = []
        target = int(slots["hr"])
        workingday = int(slots["workingday"])
        for step in range(SEQ_LEN):
            hour = (target - (SEQ_LEN - 1) + step) % 24
            typical = self._typical_row(workingday, hour)
            prev = self._typical_row(workingday, (hour - 1) % 24)
            is_target = step == SEQ_LEN - 1
            season = int(slots["season"]) if is_target else int(slots["season"])
            row = {
                "season": season,
                "yr": int(slots["yr"]),
                "mnth": int(slots["mnth"]),
                "hr": hour,
                "holiday": int(slots["holiday"]) if is_target else 0,
                "weekday": int(slots["weekday"]),
                "workingday": workingday,
                "weathersit": int(slots["weathersit"]) if is_target else int(round(typical["weathersit"])),
                "temp": float(slots["temp"]) if is_target else float(typical["temp"]),
                "atemp": float(slots["atemp"]) if is_target else float(typical["atemp"]),
                "hum": float(slots["hum"]) if is_target else float(typical["hum"]),
                "windspeed": float(slots["windspeed"]) if is_target else float(typical["windspeed"]),
                "cnt": float(typical["cnt"]),
                "lag_1": float(prev["cnt"]),
                "lag_24": float(typical["cnt"]),
                "lag_168": float(typical["cnt"]),
                "roll_24": float(self.typical.loc[self.typical["workingday"] == workingday, "cnt"].mean()),
                "observed_cnt": 0.0 if is_target else float(typical["cnt"]),
            }
            rows.append(row)
        built = finalize_features(pd.DataFrame(rows))
        built.loc[built.index[-1], "observed_cnt"] = 0.0
        return built

    def _normalize_slots(self, slots: dict) -> dict:
        clean = dict(slots)
        temp_c = float(clean.pop("temp_c", 22))
        humidity = float(clean.pop("humidity", 55))
        wind = float(clean.pop("wind_kmh", 12))
        clean["temp"] = float(np.clip(temp_c / 41, 0, 1))
        clean["atemp"] = float(np.clip(temp_c / 50, 0, 1))
        clean["hum"] = float(np.clip(humidity / 100, 0, 1))
        clean["windspeed"] = float(np.clip(wind / 67, 0, 1))
        clean.setdefault("mnth", SEASON_MONTH[int(clean["season"])])
        clean.pop("assumed_hour", None)
        return clean

    def _transformer_bound(self, window: pd.DataFrame) -> tuple[float, float]:
        values = window[FEATURE_NAMES[:-1]].to_numpy(np.float32)
        flag = np.zeros((SEQ_LEN, 1), np.float32)
        flag[-1, 0] = 1.0
        x = np.concatenate([values, flag], axis=1)[None, ...]
        x = ((x - self.mean) / self.std).astype(np.float32)
        with torch.no_grad():
            q50, q90 = self.model(torch.from_numpy(x))
        q50 = float(q50.item() * self.y_std + self.y_mean)
        q90 = float(q90.item() * self.y_std + self.y_mean)
        point = max(0.0, q50 + self.conformal["bias_q50"])
        upper = max(point, max(0.0, q90 + self.conformal["q90_adjustment"]))
        return point, upper

    def _tree_point(self, window: pd.DataFrame, which: str) -> float:
        row = window.iloc[[-1]].copy()
        for col in ("season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"):
            row[col] = row[col].round().astype(int)
        columns = [
            "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit",
            "temp", "atemp", "hum", "windspeed",
        ]
        if which == "lags":
            columns = columns + ["lag_1", "lag_24", "lag_168", "roll_24"]
            pred = self.lag_model.predict(row[columns])
        else:
            pred = self.report_model.predict(row[columns])
        return float(max(0.0, pred[0]))

    def _historical(self, slots: dict) -> tuple[float | None, int]:
        query = (
            (self.frame["hr"] == int(slots["hr"]))
            & (self.frame["season"] == int(slots["season"]))
            & (self.frame["workingday"] == int(slots["workingday"]))
            & (self.frame["weathersit"] == int(slots["weathersit"]))
        )
        sample = self.frame.loc[query, "cnt"]
        if len(sample) < 20:
            sample = self.frame.loc[
                (self.frame["hr"] == int(slots["hr"]))
                & (self.frame["season"] == int(slots["season"]))
                & (self.frame["workingday"] == int(slots["workingday"])),
                "cnt",
            ]
        if sample.empty:
            return None, 0
        return round(float(sample.mean()), 1), int(len(sample))

    def predict_scenario(self, slots: dict) -> dict:
        normalized = self._normalize_slots(slots)
        window = self._window(normalized)
        q50, q90 = self._transformer_bound(window)
        xgb_lags = self._tree_point(window, "lags")
        xgb_report = self._tree_point(window, "report")
        production = self.metrics["production"]
        point = {"transformer_q50": q50, "xgb_lags": xgb_lags, "xgb_report_style": xgb_report}[production]
        upper = max(q90, point)
        hist, n = self._historical(normalized)
        if hist and point > hist * 1.1:
            action = "Above the historical mean for these conditions, so pre-position bikes before this hour."
        elif int(normalized["weathersit"]) >= 3:
            action = "Rain suppresses rentals. Expect a quieter hour than a clear day at the same clock time."
        else:
            action = "Close to the historical mean for this hour and day type."
        return {
            "point": round(point, 1),
            "q50": round(q50, 1),
            "q90": round(q90, 1),
            "upper": round(upper, 1),
            "xgb_lags": round(xgb_lags, 1),
            "xgb_report": round(xgb_report, 1),
            "point_model": production,
            "historical_mean": hist,
            "historical_n": n,
            "action": action,
            "context": "The 23 hours of context are the historical median day for this working-day flag. The target hour uses the conditions you set, with its own count masked.",
        }

    def predict_day(self, slots: dict) -> list[dict]:
        rows = []
        for hour in range(24):
            scenario = dict(slots)
            scenario["hr"] = hour
            scenario["assumed_hour"] = False
            forecast = self.predict_scenario(scenario)
            rows.append({"hr": hour, "point": forecast["point"], "upper": forecast["upper"], "q50": forecast["q50"]})
        return rows

    def fleet(self, peak_hourly: float | None = None, p95_hourly: float | None = None, **kwargs) -> dict:
        if peak_hourly is None or p95_hourly is None:
            upper = np.array([row["q90"] for row in self.evaluation], dtype=float)
            peak_hourly = float(upper.max()) if peak_hourly is None else peak_hourly
            p95_hourly = float(np.percentile(upper, 95)) if p95_hourly is None else p95_hourly
        return estimate_fleet(peak_hourly, p95_hourly, **kwargs)

    def decision_text(self) -> str:
        return verdict(self.fleet()["peak"]["fleet"])

    def ask(self, question: str) -> dict:
        from bikeshare.agent import run_agent

        return run_agent(question, self)


_SERVICE: OpsService | None = None


def get_service() -> OpsService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = OpsService()
    return _SERVICE
