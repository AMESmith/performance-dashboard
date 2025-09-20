import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import date
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ---------------------- App Config ----------------------
st.set_page_config(page_title="Executive Assistant Performance Dashboard", layout="wide")
TITLE = "Executive Assistant Performance Dashboard"

# ---------------------- EA Insights Styling ----------------------
st.markdown("""
    <style>
    /* App base */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    h1, h2, h3, h4 {
        color: #000000;
    }
    /* Custom KPI card styling */
    .card {
        background-color: #FFFFFF;
        border: 2px solid #ffb000;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 5px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    .card h3 {
        color: #000000;
        font-size: 16px;
        margin-bottom: 5px;
    }
    .card p {
        font-size: 22px;
        font-weight: bold;
        color: #ffb000;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Constants ----------------------
SCENARIOS = ["Executive Expectations", "Targeted Performance", "Actual Performance"]
CATEGORIES = ["Executive Support", "Operational Support"]
LEVELS = ["Low", "Mid", "High"]
LEVEL_SCORES = {"Low": 1, "Mid": 2, "High": 3}

# ---------------------- Init State ----------------------
def _empty_perf_df():
    return pd.DataFrame({
        "Category": CATEGORIES,
        "Level": ["Mid"] * len(CATEGORIES),
        "Volume %": [0] * len(CATEGORIES),
    })

def _default_quarter_label():
    y = date.today().year
    m = date.today().month
    q = (m - 1)//3 + 1
    return f"{y}-Q{q}"

def init_state():
    ss = st.session_state
    if "scenarios" not in ss:
        ss.scenarios = {s: _empty_perf_df() for s in SCENARIOS}
    if "execs_supported" not in ss:
        ss.execs_supported = {s: 1 for s in SCENARIOS}
    if "quarterly" not in ss:
        ss.quarterly = pd.DataFrame([{
            "Quarter": _default_quarter_label(),
            "Emails Received": 0, "Emails Sent": 0, "Invites Actioned": 0,
            "Meetings Scheduled": 0, "Reschedules": 0, "Meeting Notes Prepared": 0,
            "Domestic Trips": 0, "International Trips": 0,
            "Onboardings Supported": 0, "Trainings Facilitated": 0,
            "Expense Reports Processed": 0, "Approvals Routed": 0,
            "Projects": 0, "Events": 0,
            "Tasks Delegated": 0, "Tasks Automated": 0, "Tasks Directly Executed": 0,
            "Reactive Work Hours": 0.0, "Overtime Hours": 0.0,
        }])
    if "show_heatmap" not in ss: ss.show_heatmap = False
    if "show_charts" not in ss: ss.show_charts = False
    if "heatmap_img" not in ss: ss.heatmap_img = None

init_state()

# ---------------------- Core math ----------------------
def compute_scores_for_scenario(df: pd.DataFrame, execs_supported: int) -> pd.DataFrame:
    out = df.copy()
    out["LevelScore"] = out["Level"].map(LEVEL_SCORES).astype(float)
    out["VolFrac"] = (out["Volume %"].astype(float).clip(0, 100)) / 100.0
    out["WeightedScore"] = out["LevelScore"] * out["VolFrac"]
    # Exec multiplier
    mask_exec = (out["Category"] == "Executive Support")
    out.loc[mask_exec, "WeightedScore"] = out.loc[mask_exec, "WeightedScore"] * max(1, int(execs_supported))
    return out

def scenario_totals(scenarios_dict, execs_supported_dict):
    return {s: compute_scores_for_scenario(df, execs_supported_dict.get(s,1))["WeightedScore"].sum()
            for s, df in scenarios_dict.items()}

def alignment_pct(actual_total, expectation_total):
    return float(actual_total / expectation_total * 100.0) if expectation_total > 0 else 0.0

def risk_table_all(scenarios_dict, execs_supported_dict):
    exp = compute_scores_for_scenario(scenarios_dict["Executive Expectations"], execs_supported_dict["Executive Expectations"])
    tgt = compute_scores_for_scenario(scenarios_dict["Targeted Performance"], execs_supported_dict["Targeted Performance"])
    act = compute_scores_for_scenario(scenarios_dict["Actual Performance"], execs_supported_dict["Actual Performance"])
    rows = []
    for cat in CATEGORIES:
        e = exp.loc[exp["Category"]==cat, "WeightedScore"].sum()
        t = tgt.loc[tgt["Category"]==cat, "WeightedScore"].sum()
        a = act.loc[act["Category"]==cat, "WeightedScore"].sum()
        rows.append({
            "Category": cat,
            "Expectation": e,
            "Target": t,
            "Actual": a,
            "Gap vs Expectation": e - a,
            "Gap vs Target": t - a
        })
    return pd.DataFrame(rows)

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.markdown("### Controls")
    period = st.text_input("Reporting period", value=date.today().strftime("%Y-%m"))
    selected_scenario = st.selectbox("Scenario to edit", SCENARIOS, index=2)
    hourly_rate = st.number_input("EA Hourly Rate ($/hr)", min_value=0.0, value=60.0, step=5.0)

    st.markdown("### Visuals")
    if st.button("Generate Heatmap"): st.session_state.show_heatmap = True
    if st.button("Generate Charts"): st.session_state.show_charts = True
    if st.button("Hide Visuals"):
        st.session_state.show_heatmap = False
        st.session_state.show_charts = False

# ---------------------- Header + KPI Cards ----------------------
st.markdown(f"## {TITLE}")

totals = scenario_totals(st.session_state.scenarios, st.session_state.execs_supported)
exp_total, tgt_total, act_total = totals["Executive Expectations"], totals["Targeted Performance"], totals["Actual Performance"]
align_val = alignment_pct(act_total, exp_total)

# Example satisfaction/savings placeholders
sat_proj = 85.0
sav_redund, sav_speed = 1000.0, 2000.0

# Row 1
cols = st.columns(5)
with cols[0]:
    st.markdown(f"<div class='card'><h3>Expectation Total</h3><p>{exp_total:.2f}</p></div>", unsafe_allow_html=True)
with cols[1]:
    st.markdown(f"<div class='card'><h3>Target Total</h3><p>{tgt_total:.2f}</p></div>", unsafe_allow_html=True)
with cols[2]:
    st.markdown(f"<div class='card'><h3>Actual Total</h3><p>{act_total:.2f}</p></div>", unsafe_allow_html=True)
with cols[3]:
    st.markdown(f"<div class='card'><h3>Alignment %</h3><p>{align_val:.1f}%</p></div>", unsafe_allow_html=True)
with cols[4]:
    st.markdown(f"<div class='card'><h3>Satisfaction</h3><p>{sat_proj:.1f}%</p></div>", unsafe_allow_html=True)

# Row 2
cols2 = st.columns(3)
with cols2[0]:
    st.markdown(f"<div class='card'><h3>Executives Supported</h3><p>{st.session_state.execs_supported['Actual Performance']}</p></div>", unsafe_allow_html=True)
with cols2[1]:
    st.markdown(f"<div class='card'><h3>Savings (Redundancy)</h3><p>${sav_redund:,.0f}</p></div>", unsafe_allow_html=True)
with cols2[2]:
    st.markdown(f"<div class='card'><h3>Savings (Speed)</h3><p>${sav_speed:,.0f}</p></div>", unsafe_allow_html=True)

st.divider()

# ---------------------- Performance Volume ----------------------
st.markdown("### Performance Volume")
st.caption("Edit levels and volume percentages for the selected scenario (see toggle on the right).")
st.caption("Operational Support = % of work outside direct executive actions (projects, onboarding, events, etc.).")

perf_df = st.session_state.scenarios[selected_scenario].copy()
edited = st.data_editor(
    perf_df, use_container_width=True, disabled=["Category"],
    column_config={
        "Category": st.column_config.TextColumn(),
        "Level": st.column_config.SelectboxColumn(options=LEVELS, required=True),
        "Volume %": st.column_config.NumberColumn(min_value=0, max_value=100, step=5),
    },
    key=f"perf_editor_{selected_scenario}",
)
st.session_state.scenarios[selected_scenario] = edited

st.divider()

# ---------------------- Risk Indicators ----------------------
st.markdown("### Risk Indicators")
st.dataframe(risk_table_all(st.session_state.scenarios, st.session_state.execs_supported), use_container_width=True)

