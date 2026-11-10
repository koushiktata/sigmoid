"""Streamlit desk. Every number comes from the FastAPI service."""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("BIKESHARE_API", "http://127.0.0.1:8000")
SEASONS = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
WEATHER = {1: "Clear", 2: "Mist / cloudy", 3: "Light rain or snow", 4: "Heavy rain or snow"}
MONTH_FOR_SEASON = {1: 3, 2: 6, 3: 9, 4: 12}

st.set_page_config(page_title="Bikeshare Ops", page_icon="🚲", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.1rem; max-width: 1180px;}
    div[data-testid="stMetric"] {
        background: #18222d;
        border: 1px solid #2c3c4e;
        border-left: 3px solid #f5b942;
        padding: 0.55rem 0.75rem;
        border-radius: 8px;
    }
    .banner {
        background: #18222d;
        border: 1px solid #2c3c4e;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
    }
    .banner h1 {margin: 0; font-size: 1.7rem; letter-spacing: -0.03em;}
    .banner p {margin: 0.25rem 0 0; color: #b7c3d1;}
    .trace {color: #8ea0b5; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str):
    try:
        response = requests.get(f"{API}{path}", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API is not reachable at {API}. Start it with scripts/dev.sh. ({exc})")
        st.stop()


def api_post(path: str, payload: dict):
    try:
        response = requests.post(f"{API}{path}", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = exc.response.text[:400]
        st.error(f"Request failed. {detail or exc}")
        st.stop()


def themed(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#c5d0dc",
            titleColor="#c5d0dc",
            gridColor="#243140",
            domainColor="#243140",
        )
        .configure_legend(labelColor="#c5d0dc", titleColor="#c5d0dc")
    )


def scenario_form(prefix: str, show_hour: bool = True) -> dict:
    season = st.selectbox("Season", list(SEASONS), format_func=lambda k: SEASONS[k], key=f"{prefix}-season")
    weather = st.selectbox("Weather", list(WEATHER), format_func=lambda k: WEATHER[k], key=f"{prefix}-wx")
    day = st.selectbox("Day", ["Working day", "Weekend", "Holiday"], key=f"{prefix}-day")
    year = st.selectbox("Year", [2012, 2011], key=f"{prefix}-year")
    temp = st.slider("Temperature °C", 0, 40, 22 if season == 2 else 18, key=f"{prefix}-temp")
    humidity = st.slider("Humidity %", 10, 100, 75 if weather >= 3 else 55, key=f"{prefix}-hum")
    hour = st.slider("Hour", 0, 23, 8, key=f"{prefix}-hr") if show_hour else 8
    holiday = 1 if day == "Holiday" else 0
    workingday = 1 if day == "Working day" else 0
    weekday = 2 if workingday else 6
    return {
        "hr": hour,
        "season": season,
        "weathersit": weather,
        "workingday": workingday,
        "holiday": holiday,
        "weekday": weekday,
        "yr": year - 2011,
        "mnth": MONTH_FOR_SEASON[season],
        "temp_c": temp,
        "humidity": humidity,
        "wind_kmh": 12,
    }


st.markdown(
    """
    <div class="banner">
      <h1>Bikeshare Ops</h1>
      <p>Hourly demand for Capital Bikeshare, 2011–2012. A quantile transformer plans the peak; the desk only calls the API.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Desk",
    ["Command", "Forecast", "Fleet", "Evidence", "Copilot"],
    label_visibility="collapsed",
)
st.sidebar.caption("System-wide completed rentals. Not station inventory, and not unmet demand.")

if page == "Command":
    overview = api_get("/overview")
    quality = overview["quality"]
    headline = overview["headline"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hours", f"{quality['rows']:,}")
    c2.metric("Mean rentals / hour", quality["mean_cnt"])
    c3.metric("Record hour", quality["record_cnt"])
    c4.metric("Holdout RMSE", headline["rmse"])
    st.info(overview["verdict"])
    profile = pd.DataFrame(overview["profile"]["by_hour"])
    profile["day"] = profile["workingday"].map({1: "Working day", 0: "Non-working day"})
    chart = (
        alt.Chart(profile)
        .mark_line(strokeWidth=2.4)
        .encode(
            x=alt.X("hr:O", title="Hour"),
            y=alt.Y("cnt:Q", title="Mean rentals"),
            color=alt.Color(
                "day:N",
                scale=alt.Scale(domain=["Working day", "Non-working day"], range=["#f5b942", "#7eb6ff"]),
                title=None,
            ),
        )
        .properties(height=320)
    )
    st.altair_chart(themed(chart), use_container_width=True)
    left, right = st.columns(2)
    season = pd.DataFrame({"season": list(overview["profile"]["by_season"]), "rentals": list(overview["profile"]["by_season"].values())})
    weather = pd.DataFrame({"weather": list(overview["profile"]["by_weather"]), "rentals": list(overview["profile"]["by_weather"].values())})
    left.altair_chart(
        themed(alt.Chart(season).mark_bar(color="#f5b942", cornerRadiusEnd=3).encode(x=alt.X("rentals:Q", title="Mean rentals"), y=alt.Y("season:N", sort="-x", title=None)).properties(height=180, title="Season")),
        use_container_width=True,
    )
    right.altair_chart(
        themed(alt.Chart(weather).mark_bar(color="#7eb6ff", cornerRadiusEnd=3).encode(x=alt.X("rentals:Q", title="Mean rentals"), y=alt.Y("weather:N", sort="-x", title=None)).properties(height=180, title="Weather")),
        use_container_width=True,
    )
    st.caption(
        f"Record hour {quality['record_at']} · heavy-weather hours in the file: {quality['heavy_weather_hours']} · "
        f"missing cells {quality['missing_cells']} · count mismatches {quality['count_mismatches']}."
    )

elif page == "Forecast":
    st.subheader("Hour and day")
    left, right = st.columns([1, 1.4])
    with left:
        slots = scenario_form("fc", show_hour=True)
        if st.button("Forecast", type="primary", use_container_width=True):
            st.session_state["forecast"] = {
                "hour": api_post("/predict", slots),
                "day": api_post("/predict/day", slots),
            }
    with right:
        result = st.session_state.get("forecast")
        if not result:
            st.write("Set the hour and weather, then forecast. The band is the conformal 90% ceiling from the transformer.")
        else:
            hour = result["hour"]
            a, b, c = st.columns(3)
            a.metric("Point forecast", f"{hour['point']:.0f}")
            b.metric("90% ceiling", f"{hour['upper']:.0f}")
            hist = "—" if hour["historical_mean"] is None else f"{hour['historical_mean']:.0f}"
            c.metric("Historical mean", hist)
            st.write(hour["action"])
            st.caption(hour["context"])
            day = pd.DataFrame(result["day"]["hours"])
            long = day.melt("hr", ["point", "upper"], var_name="series", value_name="rentals")
            long["series"] = long["series"].map({"point": "Point", "upper": "90% ceiling"})
            curve = (
                alt.Chart(long)
                .mark_line(strokeWidth=2.4)
                .encode(
                    x=alt.X("hr:O", title="Hour"),
                    y=alt.Y("rentals:Q", title="Rentals"),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(domain=["Point", "90% ceiling"], range=["#f5b942", "#3ddc97"]),
                        title=None,
                    ),
                )
                .properties(height=320)
            )
            st.altair_chart(themed(curve), use_container_width=True)
            st.caption(
                f"Point model: {hour['point_model']} · transformer median {hour['q50']:.0f} · "
                f"no-lag XGBoost {hour['xgb_report']:.0f} · lag XGBoost {hour['xgb_lags']:.0f}."
            )

elif page == "Fleet":
    st.subheader("System-wide fleet desk")
    st.caption("Concurrent bikes ≈ hourly rentals × trip length. Then a demand buffer and a maintenance reserve.")
    trip = st.slider("Average trip length (hours)", 0.2, 1.5, 0.5, 0.05)
    buffer = st.slider("Demand buffer", 0.0, 0.4, 0.15, 0.05)
    maintenance = st.slider("Maintenance reserve", 0.0, 0.3, 0.10, 0.05)
    result = api_post("/fleet", {"trip_hours": trip, "buffer": buffer, "maintenance": maintenance})
    estimate = result["estimate"]
    a, b, c = st.columns(3)
    a.metric("Peak hour used", f"{estimate['peak']['hourly_rentals']:.0f}")
    b.metric("Fleet at that peak", estimate["peak"]["fleet"])
    c.metric("Fleet at P95", estimate["p95"]["fleet"])
    st.info(result["verdict"])
    st.write(estimate["note"])
    table = pd.DataFrame(
        [
            {"Level": "Holdout 90% peak", "Rentals": estimate["peak"]["hourly_rentals"], "In use": estimate["peak"]["bikes_in_use"], "Fleet": estimate["peak"]["fleet"]},
            {"Level": "Holdout P95", "Rentals": estimate["p95"]["hourly_rentals"], "In use": estimate["p95"]["bikes_in_use"], "Fleet": estimate["p95"]["fleet"]},
            {"Level": "Case-study fleet", "Rentals": None, "In use": None, "Fleet": estimate["case_study_fleet"]},
        ]
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

elif page == "Evidence":
    evaluation = api_get("/evaluation")
    overview = api_get("/overview")
    metrics = pd.DataFrame(evaluation["metrics"]).T.reset_index().rename(columns={"index": "model"})
    st.subheader("Future holdout")
    st.caption(overview["split"]["protocol"] + f" Test after {overview['split']['test_after']}. n={overview['split']['n_test']}.")
    st.dataframe(metrics, hide_index=True, use_container_width=True)
    st.caption("Peak bias is actual minus forecast on working-day commute hours. Positive means the model is still short at the peak. MAPE is inflated by hours with only a few rentals.")
    sample = pd.DataFrame(evaluation["rows"])
    sample["ts"] = pd.to_datetime(sample["ts"])
    daily = sample.set_index("ts").resample("D")[["actual", "transformer", "xgb_report"]].mean().reset_index()
    long = daily.melt("ts", ["actual", "transformer", "xgb_report"], var_name="series", value_name="rentals")
    long["series"] = long["series"].map({"actual": "Actual", "transformer": "Transformer", "xgb_report": "Report-style XGBoost"})
    chart = (
        alt.Chart(long)
        .mark_line(strokeWidth=1.8)
        .encode(
            x=alt.X("ts:T", title=None),
            y=alt.Y("rentals:Q", title="Daily mean rentals"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["Actual", "Transformer", "Report-style XGBoost"],
                    range=["#e7edf4", "#f5b942", "#7eb6ff"],
                ),
                title=None,
            ),
        )
        .properties(height=320)
    )
    st.altair_chart(themed(chart), use_container_width=True)
    hypo = api_get("/hypotheses")
    for key in ("rush", "weather", "workingday"):
        block = hypo[key]
        with st.expander(f"{block['name']} · {block['test']}", expanded=key == "rush"):
            st.write(block["h0"])
            st.write(block["reading"])
            if "cohens_d" in block:
                st.write(f"Statistic {block['statistic']} · p {block['p_value']:.3g} · d {block['cohens_d']} ({block['effect']})")
            else:
                st.write(f"Statistic {block['statistic']} · p {block['p_value']:.3g}")
            if "groups" in block:
                st.dataframe(pd.DataFrame(block["groups"]), hide_index=True, use_container_width=True)

elif page == "Copilot":
    st.subheader("Ops copilot")
    st.caption("A small graph routes the question to a tool. Numbers come from the model, not from the language model.")
    examples = [
        "What is the demand at 8 AM on a working day in summer?",
        "Predict evening demand on a rainy weekend in fall",
        "How many bikes should we plan for at the peak?",
        "Does rush hour really change demand?",
    ]
    chosen = st.pills("Try", examples, selection_mode="single")
    question = st.text_input("Question", value=chosen or "")
    if st.button("Ask", type="primary") and question.strip():
        st.session_state["answer"] = api_post("/ask", {"question": question.strip()})
    answer = st.session_state.get("answer")
    if answer:
        st.write(answer["answer"])
        st.markdown(f"<p class='trace'>{' → '.join(answer['trace'])}</p>", unsafe_allow_html=True)
        if answer.get("slots"):
            st.caption(f"Parsed scenario: {answer['slots']}")
