"""Tool-routing graph for the ops copilot.

Nodes are route → tool → compose. A language model is optional and may only
rewrite the drafted answer. It never supplies the numbers.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

from bikeshare.data import SEASON_NAMES, WEATHER_NAMES

SEASON_WORDS = {
    "spring": 1,
    "summer": 2,
    "fall": 3,
    "autumn": 3,
    "winter": 4,
}
MONTH_WORDS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
MONTH_SEASON = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
SEASON_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}
DEFAULT_TEMP_C = {1: 12.0, 2: 22.0, 3: 26.0, 4: 14.0}


def _hour(text: str) -> int | None:
    clock = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", text)
    if clock:
        hour = int(clock.group(1)) % 12
        if clock.group(3).startswith("p"):
            hour += 12
        return hour
    if re.search(r"\b(noon|midday)\b", text):
        return 12
    if re.search(r"\bmorning\b", text):
        return 8
    if re.search(r"\bevening\b|\btonight\b", text):
        return 18
    if re.search(r"\bafternoon\b", text):
        return 15
    if re.search(r"\bnight\b", text):
        return 21
    if re.search(r"\brush\b", text):
        return 8
    named = re.search(r"\b([01]?\d|2[0-3])\s*:00\b", text)
    if named:
        return int(named.group(1))
    return None


def parse_slots(question: str) -> dict:
    text = question.lower()
    season = next((code for word, code in SEASON_WORDS.items() if word in text), 2)
    month = next((code for word, code in MONTH_WORDS.items() if word in text), None)
    if month:
        season = MONTH_SEASON[month]
    else:
        month = SEASON_MONTH[season]

    weather = 1
    if re.search(r"storm|heavy rain|snowstorm|downpour", text):
        weather = 4
    elif re.search(r"rain|shower|drizzle|wet|snow", text):
        weather = 3
    elif re.search(r"mist|cloud|overcast|fog", text):
        weather = 2
    elif "clear" in text or "sunny" in text:
        weather = 1

    holiday = 1 if "holiday" in text else 0
    if re.search(r"weekend|saturday|sunday|non-working", text) or holiday:
        workingday, weekday = 0, 6
    elif re.search(r"working|weekday|commute", text):
        workingday, weekday = 1, 2
    else:
        workingday, weekday = 1, 2

    year = 0 if "2011" in text else 1
    hour = _hour(text)
    assumed_hour = hour is None
    temp = DEFAULT_TEMP_C[season]
    humidity = 80.0 if weather >= 3 else 55.0
    return {
        "hr": 8 if hour is None else hour,
        "season": season,
        "mnth": month,
        "weathersit": weather,
        "workingday": workingday,
        "holiday": holiday,
        "weekday": weekday,
        "yr": year,
        "temp_c": temp,
        "humidity": humidity,
        "wind_kmh": 12.0,
        "assumed_hour": assumed_hour,
    }


def route(question: str) -> str:
    text = question.lower().strip()
    if not text:
        return "help"
    if re.search(r"\bfleet\b|how many bikes|bike[s]? (do we|should)|fleet size", text):
        return "fleet"
    if re.search(r"\bbuy\b|redistribut|rebalanc|more bikes|capital", text):
        return "decision"
    if re.search(r"hypothesis|p-value|significant|rush hour really|does rush|effect size", text):
        return "hypothesis"
    if re.search(r"how does weather|weather affect|why .*weather", text):
        return "hypothesis"
    if re.search(r"\br2\b|rmse|mae\b|accurate|which model|xgboost|transformer|holdout", text):
        return "models"
    if re.search(r"\bhelp\b|what can you", text):
        return "help"
    return "forecast"


def _fmt_p(value: float) -> str:
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _scenario_label(slots: dict) -> str:
    day = "a holiday" if slots["holiday"] else ("a working day" if slots["workingday"] else "a non-working day")
    year = 2011 + int(slots["yr"])
    return (
        f"{slots['hr']:02d}:00 on {day} in {SEASON_NAMES[slots['season']]} {year}, "
        f"{WEATHER_NAMES[slots['weathersit']].lower()}, about {slots['temp_c']:.0f}°C"
    )


def compose(question: str, intent: str, payload: dict) -> str:
    if intent == "help":
        return (
            "Ask for an hour and conditions, for example: "
            "“What is demand at 8 AM on a working day in summer?” "
            "You can also ask whether rush hour is real, how large the fleet should be, "
            "or whether to buy bikes or move them."
        )
    if intent == "forecast":
        slots = payload["slots"]
        note = " Hour was not in the question, so this uses 08:00." if slots.get("assumed_hour") else ""
        hist = payload["historical_mean"]
        hist_txt = (
            f"Matching historical hours averaged {hist:.0f} rentals (n={payload['historical_n']})."
            if hist is not None
            else "There are too few matching historical hours to quote a mean."
        )
        return (
            f"For {_scenario_label(slots)}, the {payload['point_model']} forecasts "
            f"{payload['point']:.0f} rentals. The 90% planning ceiling is {payload['upper']:.0f}.{note} "
            f"{hist_txt} {payload['action']}"
        )
    if intent == "fleet":
        fleet = payload["fleet"]["peak"]
        return (
            f"Using a holdout 90% peak of {fleet['hourly_rentals']:.0f} rentals, "
            f"a {payload['fleet']['trip_hours']:.2f} h trip, a {payload['fleet']['buffer']:.0%} buffer "
            f"and {payload['fleet']['maintenance']:.0%} maintenance, about {fleet['bikes_in_use']:.0f} bikes "
            f"are in use at once and the planning fleet is {fleet['fleet']}. "
            f"A busy-but-not-record hour (P95) needs {payload['fleet']['p95']['fleet']} bikes. "
            f"{payload['verdict']}"
        )
    if intent == "hypothesis":
        rush = payload["rush"]
        weather = payload["weather"]
        day = payload["workingday"]
        return (
            f"Rush vs other hours: {rush['test']}, t={rush['statistic']}, p={_fmt_p(rush['p_value'])}, "
            f"Cohen's d={rush['cohens_d']} ({rush['effect']}). {rush['reading']} "
            f"Weather: Kruskal-Wallis H={weather['statistic']}, p={_fmt_p(weather['p_value'])}. {weather['reading']} "
            f"Working vs non-working days: d={day['cohens_d']} ({day['effect']}). {day['reading']}"
        )
    if intent == "models":
        rows = payload["models"]
        prod = payload["production"]
        bits = [
            f"{name} RMSE {stats['rmse']} (R² {stats['r2']}, peak RMSE {stats['peak_rmse']})"
            for name, stats in rows.items()
        ]
        return (
            "On the same future holdout (after 2012-08-07), "
            + "; ".join(bits)
            + f". Production point forecast: {prod}. "
            "The report's no-lag XGBoost is the baseline. "
            "MAPE looks harsh because many night hours have only a few rentals; RMSE is the number to use. "
            f"Peak bias is actual minus forecast on working-day commute hours (positive means under-prediction). "
            f"Transformer peak bias is {rows.get('transformer_q50', {}).get('peak_bias')}."
        )
    if intent == "decision":
        return payload["verdict"] + " " + payload["fleet"]["note"]
    return "I could not match that to a demand, fleet, or evidence question."


def maybe_rewrite(question: str, draft: str, facts: dict) -> tuple[str, str]:
    """Optional rewrite. Numbers stay anchored to the tool payload."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return draft, "deterministic"
    base = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.environ.get("BIKESHARE_LLM", "openai/gpt-4o-mini")
    if "OPENAI_API_KEY" in os.environ and "OPENROUTER_API_KEY" not in os.environ:
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("BIKESHARE_LLM", "gpt-4o-mini")
    body = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the Bikeshare Ops desk. Rewrite the draft in under 130 words for a shift lead. "
                    "Use only numbers that appear in the draft or the JSON. Do not invent stations, "
                    "trip counts, or a buy order. Keep the planning ceiling and the caveat."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"question": question, "draft": draft, "facts": facts}),
            },
        ],
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            parsed = json.loads(response.read().decode())
        text = parsed["choices"][0]["message"]["content"].strip()
        return (text or draft), "llm"
    except Exception:
        return draft, "deterministic"


def run_agent(question: str, ops) -> dict:
    """route → tool → compose. `ops` is the forecast service."""
    intent = route(question)
    trace = ["route:" + intent]
    slots = parse_slots(question) if intent == "forecast" else None
    if intent == "forecast":
        payload = ops.predict_scenario(slots)
        payload["slots"] = slots
        trace.append("tool:forecast")
    elif intent == "fleet":
        payload = {"fleet": ops.fleet(), "verdict": ops.decision_text()}
        trace.append("tool:fleet")
    elif intent == "decision":
        payload = {"fleet": ops.fleet(), "verdict": ops.decision_text()}
        trace.append("tool:decision")
    elif intent == "hypothesis":
        payload = ops.hypotheses
        trace.append("tool:hypothesis")
    elif intent == "models":
        payload = {"models": ops.metrics["models"], "production": ops.metrics["production"]}
        trace.append("tool:models")
    else:
        payload = {}
        trace.append("tool:help")
    draft = compose(question, intent, payload)
    answer, engine = maybe_rewrite(question, draft, _public_facts(intent, payload))
    trace.append("compose:" + engine)
    return {
        "question": question,
        "intent": intent,
        "trace": trace,
        "slots": slots,
        "answer": answer,
        "engine": engine,
        "facts": _public_facts(intent, payload),
    }


def _public_facts(intent: str, payload: dict) -> dict:
    if intent == "forecast":
        keep = ("point", "upper", "q50", "q90", "point_model", "historical_mean", "historical_n", "action")
        return {key: payload.get(key) for key in keep}
    if intent in {"fleet", "decision"}:
        return {"peak_fleet": payload["fleet"]["peak"]["fleet"], "p95_fleet": payload["fleet"]["p95"]["fleet"]}
    if intent == "models":
        return payload
    if intent == "hypothesis":
        return {
            "rush_d": payload["rush"]["cohens_d"],
            "rush_p": payload["rush"]["p_value"],
            "weather_p": payload["weather"]["p_value"],
            "workingday_d": payload["workingday"]["cohens_d"],
        }
    return {}
