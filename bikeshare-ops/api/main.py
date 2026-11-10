"""FastAPI backend for Bikeshare Ops. The Streamlit desk is a client of this API."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bikeshare.fleet import verdict
from bikeshare.service import get_service

app = FastAPI(title="Bikeshare Ops API", version="1.0.0")


class ScenarioIn(BaseModel):
    hr: int = Field(8, ge=0, le=23)
    season: int = Field(2, ge=1, le=4)
    weathersit: int = Field(1, ge=1, le=4)
    workingday: int = Field(1, ge=0, le=1)
    holiday: int = Field(0, ge=0, le=1)
    weekday: int = Field(2, ge=0, le=6)
    yr: int = Field(1, ge=0, le=1)
    mnth: int = Field(6, ge=1, le=12)
    temp_c: float = Field(22, ge=-5, le=45)
    humidity: float = Field(55, ge=0, le=100)
    wind_kmh: float = Field(12, ge=0, le=67)


class FleetIn(BaseModel):
    peak_hourly: float | None = None
    p95_hourly: float | None = None
    trip_hours: float = Field(0.5, gt=0, le=3)
    buffer: float = Field(0.15, ge=0, le=1)
    maintenance: float = Field(0.10, ge=0, le=0.5)


class AskIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


def _ops():
    try:
        return get_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/overview")
def overview():
    return _ops().overview()


@app.get("/hypotheses")
def hypotheses():
    return _ops().hypotheses


@app.get("/evaluation")
def evaluation():
    return {"rows": _ops().evaluation, "metrics": _ops().metrics["models"], "production": _ops().metrics["production"]}


@app.post("/predict")
def predict(body: ScenarioIn):
    return _ops().predict_scenario(body.model_dump())


@app.post("/predict/day")
def predict_day(body: ScenarioIn):
    return {"hours": _ops().predict_day(body.model_dump())}


@app.post("/fleet")
def fleet(body: FleetIn):
    ops = _ops()
    estimate = ops.fleet(
        peak_hourly=body.peak_hourly,
        p95_hourly=body.p95_hourly,
        trip_hours=body.trip_hours,
        buffer=body.buffer,
        maintenance=body.maintenance,
    )
    return {"estimate": estimate, "verdict": verdict(estimate["peak"]["fleet"])}


@app.post("/ask")
def ask(body: AskIn):
    return _ops().ask(body.question.strip())
