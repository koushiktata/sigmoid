# Bikeshare Ops

Hourly demand desk for the Capital Bikeshare 2011–2012 log. The UI is Streamlit. The model, tests, and fleet arithmetic live behind FastAPI.

The dataset records completed rentals for the whole city. It does not record stations, trip length, or people who found an empty dock.

## Architecture

```text
ui/app.py  (Streamlit)
    │  HTTP
    ▼
api/main.py  (FastAPI)
    │
    ▼
src/bikeshare/service.py
    ├── QuantileTransformer   last 24 hours → median and 90% bound
    ├── XGBoost, no lags      the report baseline, same future holdout
    ├── XGBoost, causal lags  same-hour yesterday, last week, trailing day
    ├── split conformal       calibration slice fixes the peak shortfall
    ├── hypotheses            working day, rush hour, weather
    └── agent graph           route → tool → compose
```

The copilot is a three-node graph. A language model is optional (`OPENAI_API_KEY` or `OPENROUTER_API_KEY`) and may only rewrite a draft. The numbers always come from the tools.

## Why this is ahead of the written report

The report's best model is a no-lag XGBoost (published holdout R² 0.877, RMSE 77.24). It under-predicts the busiest hours, and the fleet figure is taken from in-sample predictions.

This project:

- Scores every model on that same future holdout (dates after the 80% cutoff, 2012-08-08 onward).
- Feeds the transformer only information known before the hour: calendar, weather, and causal lags. The hour's own count is masked.
- Trains the transformer on the first 70% of dates and uses the next 10% to calibrate a 90% upper bound (split conformal). The holdout is never used for that adjustment.
- Weights commute hours more heavily in the loss, because that is where a short forecast hurts.
- Sizes the fleet from the holdout 90% bound, not from a fitted peak inside the training period.
- Leaves the buy-versus-rebalance call open: the citywide number is far below the case-study fleet of about 2,000, which is a reason to rebalance first, not proof that no bikes should be bought.

## Run

```bash
cd bikeshare-ops
PYTHONPATH=src .venv/bin/python -m bikeshare.train   # once
chmod +x scripts/dev.sh
./scripts/dev.sh
```

- Desk: http://127.0.0.1:8501
- API: http://127.0.0.1:8000/docs

`BIKESHARE_API` overrides the UI's API address. `BIKESHARE_LLM` overrides the optional rewrite model.

Data: `data/hour.csv` and `data/day.csv` (Fanaee-T & Gama, 2013). Cite that paper if you publish results.
