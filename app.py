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
    .stApp { background-color: #FFFFFF; color: #000000; }
    h1, h2, h3, h4 { color: #000000; }
    /* KPI Cards */
    .card {
        background-color: #000000;
        border: 2px solid #ffb000;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 6px 4px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.35);
    }
    .card h3 {
        color: #FFFFFF;
        font-size: 16px;
        font-weight: 600;
        margin: 0 0 8px 0;
    }
    .card p.value {
        font-size: 28px;
        font-weight: 900;
        color: #ffb000;
        margin: 0 0 6px 0;
        line-height: 1.1;
    }
    .card p.sub {
        font-size: 13px;
        color: #FFFFFF;
        margin: 0;
        opacity: 0.95;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Constants ----------------------
SCENARIOS = ["Executive Expectations", "Targeted Performance", "Actual Performance"]
LEVELS = ["Low", "Mid", "High"]
LEVEL_SCORES = {"Low": 1, "Mid": 2, "High": 3}
CATEGORIES = ["Executive Support", "Operational Support"]

# ---------------------- Workload Weights ----------------------
WORKLOAD_WEIGHTS = {
    "Emails Received": 0.05, "Emails Sent": 0.05, "Invites Actioned": 0.1,
    "Meetings Scheduled": 0.2, "Reschedules": 0.3, "Meeting Notes Prepared": 0.5,
    "Domestic Trips": 1.0, "International Trips": 3.0,
    "Onboardings Supported": 0.5, "Trainings Facilitated": 0.5,
    "Expense Reports Processed": 0.25, "Approvals Routed": 0.25,
    "Projects": 1.0, "Events": 2.0,
    "Tasks Delegated": -0.2, "Tasks Automated": -0.3, "Tasks Directly Executed": 0.2,
    "Reactive Work Hours": 1.0, "Overtime Hours": 1.5,
}

# ---------------------- Init State ----------------------
def _empty_perf_df():
    return pd.DataFrame({
        "Category": CATEGORIES,
        "Level": ["Mid"] * len(CATEGORIES),
        "Volume %": [0] * len(CATEGORIES),
    })

def _default_quarter_label():
    y, m = date.today().year, date.today().month
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
init_state()

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header("Settings")
    period_type = st.radio("Period Type", ["Monthly", "Quarterly"], horizontal=True)
    if period_type == "Monthly":
        month = st.selectbox("Month", list(WORKLOAD_WEIGHTS.keys())[0:12] or [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        year = st.number_input("Year", min_value=2000, max_value=2100, value=date.today().year)
        period_label = f"{month} {year}"
        month_to_q = {
            "January":"Q1","February":"Q1","March":"Q1",
            "April":"Q2","May":"Q2","June":"Q2",
            "July":"Q3","August":"Q3","September":"Q3",
            "October":"Q4","November":"Q4","December":"Q4"
        }
        selected_quarter_label = f"{year}-{month_to_q.get(month,'Q1')}"
    else:
        q_choice = st.selectbox("Quarter", ["Q1","Q2","Q3","Q4"])
        year = st.number_input("Year", min_value=2000, max_value=2100, value=date.today().year)
        period_label = f"{q_choice} {year}"
        selected_quarter_label = f"{year}-{q_choice}"

    annual_rollup = st.checkbox("Annual Rollup (aggregate all four quarters)")
    hourly_rate = st.number_input("EA Hourly Rate ($/hr)", min_value=0.0, value=60.0, step=5.0)
    selected_scenario = st.selectbox("Scenario to edit", SCENARIOS, index=2)

# ---------------------- Scoring ----------------------
def compute_scores_for_scenario(df, execs_supported):
    out = df.copy()
    out["LevelScore"] = out["Level"].map(LEVEL_SCORES).astype(float)
    out["VolFrac"] = (out["Volume %"].astype(float).clip(0,100)) / 100.0
    out["WeightedScore"] = out["LevelScore"] * out["VolFrac"]
    mask_exec = (out["Category"] == "Executive Support")
    out.loc[mask_exec, "WeightedScore"] *= max(1,int(execs_supported))
    return out

def scenario_totals(scenarios_dict, execs_supported_dict):
    return {s: compute_scores_for_scenario(df, execs_supported_dict.get(s,1))["WeightedScore"].sum()
            for s, df in scenarios_dict.items()}

def compute_weighted_load(scenarios_dict, execs_supported_dict, quarterly_df, period_label, annual_year=None):
    base = scenario_totals(scenarios_dict, execs_supported_dict)
    workload_score = 0.0
    if annual_year:
        df_year = quarterly_df[quarterly_df["Quarter"].astype(str).str.startswith(str(annual_year))]
        for _, row in df_year.iterrows():
            for k,v in row.items():
                if k in WORKLOAD_WEIGHTS: workload_score += v * WORKLOAD_WEIGHTS[k]
    else:
        qrow = quarterly_df[quarterly_df["Quarter"]==period_label]
        if not qrow.empty:
            for k,v in qrow.iloc[0].to_dict().items():
                if k in WORKLOAD_WEIGHTS: workload_score += v * WORKLOAD_WEIGHTS[k]
    weighted = {s: base[s] + workload_score for s in base}
    return weighted, workload_score

# ---------------------- Header ----------------------
st.title(TITLE)
st.markdown(f"**Reporting Period:** {period_label}")

# ---------------------- KPIs ----------------------
if annual_rollup:
    totals, workload_score = compute_weighted_load(st.session_state.scenarios, st.session_state.execs_supported, st.session_state.quarterly, period_label, annual_year=year)
    view_label = f"Annual {year}"
else:
    totals, workload_score = compute_weighted_load(st.session_state.scenarios, st.session_state.execs_supported, st.session_state.quarterly, selected_quarter_label)
    view_label = period_label

exp_total = totals["Executive Expectations"]
tgt_total = totals["Targeted Performance"]
act_total = totals["Actual Performance"]
target_pct = int(round((tgt_total/exp_total*100))) if exp_total>0 else 0
actual_pct = int(round((act_total/exp_total*100))) if exp_total>0 else 0

c1,c2,c3,c4,c5 = st.columns(5)
with c1: st.markdown(f"<div class='card'><h3>Expectation</h3><p class='value'>100%</p><p class='sub'>{round(exp_total)} pts</p></div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='card'><h3>Target</h3><p class='value'>{target_pct}%</p><p class='sub'>{round(tgt_total)} pts</p></div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='card'><h3>Actual</h3><p class='value'>{actual_pct}%</p><p class='sub'>{round(act_total)} pts</p></div>", unsafe_allow_html=True)
with c4: st.markdown(f"<div class='card'><h3>Alignment</h3><p class='value'>{actual_pct}%</p><p class='sub'>{round(act_total)} pts delivered</p></div>", unsafe_allow_html=True)
with c5: st.markdown(f"<div class='card'><h3>Workload</h3><p class='value'>{round(workload_score)} pts</p><p class='sub'>People, Ops, Finance</p></div>", unsafe_allow_html=True)

st.divider()

# ---------------------- Quarterly Comparison ----------------------
if annual_rollup:
    df_year = st.session_state.quarterly[st.session_state.quarterly["Quarter"].astype(str).str.startswith(str(year))]
    if not df_year.empty:
        st.subheader(f"Quarterly Comparison — {year}")
        totals_by_q = []
        for qnum in [1,2,3,4]:
            qlabel = f"{year}-Q{qnum}"
            qtotals, _ = compute_weighted_load(st.session_state.scenarios, st.session_state.execs_supported, st.session_state.quarterly, qlabel)
            totals_by_q.append({"Quarter": qlabel, **qtotals})
        comp = pd.DataFrame(totals_by_q)
        st.bar_chart(comp.set_index("Quarter"))

# ---------------------- Risk Indicators ----------------------
st.subheader("Risk Indicators")
def risk_table_all(scenarios_dict, execs_supported_dict):
    exp = compute_scores_for_scenario(scenarios_dict["Executive Expectations"], execs_supported_dict["Executive Expectations"])
    tgt = compute_scores_for_scenario(scenarios_dict["Targeted Performance"], execs_supported_dict["Targeted Performance"])
    act = compute_scores_for_scenario(scenarios_dict["Actual Performance"], execs_supported_dict["Actual Performance"])
    rows = []
    for cat in CATEGORIES:
        e = exp.loc[exp["Category"]==cat, "WeightedScore"].sum()
        t = tgt.loc[tgt["Category"]==cat, "WeightedScore"].sum()
        a = act.loc[act["Category"]==cat, "WeightedScore"].sum()
        rows.append({"Category":cat,"Expectation":e,"Target":t,"Actual":a,"Gap vs Exp":e-a,"Gap vs Tgt":t-a})
    return pd.DataFrame(rows)
st.dataframe(risk_table_all(st.session_state.scenarios, st.session_state.execs_supported), use_container_width=True)

# ---------------------- Exports ----------------------
def export_to_excel(scenarios_dict, execs_supported_dict, quarterly_df, period):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for scen, df in scenarios_dict.items():
            scored = compute_scores_for_scenario(df, execs_supported_dict.get(scen,1))
            scored.to_excel(writer, index=False, sheet_name=scen[:31])
        quarterly_df.to_excel(writer, index=False, sheet_name="Quarterly Raw")
    output.seek(0); return output

def export_to_pdf(scenarios_dict, execs_supported_dict, period):
    buffer = io.BytesIO(); doc = SimpleDocTemplate(buffer); styles = getSampleStyleSheet()
    elements=[]; totals_local=scenario_totals(scenarios_dict,execs_supported_dict)
    elements.append(Paragraph(TITLE, styles["Title"])); elements.append(Paragraph(f"Reporting Period: {period}", styles["Normal"]))
    tbl=[["Scenario","Total Score"]]+[[k,f"{v:.2f}"] for k,v in totals_local.items()]
    t=Table(tbl); t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.black)])); elements.append(t); doc.build(elements)
    buffer.seek(0); return buffer

with st.sidebar:
    excel_bytes=export_to_excel(st.session_state.scenarios, st.session_state.execs_supported, st.session_state.quarterly, period_label)
    st.download_button("📊 Download Excel", data=excel_bytes, file_name=f"EA_Performance_{period_label}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    pdf_bytes=export_to_pdf(st.session_state.scenarios, st.session_state.execs_supported, period_label)
    st.download_button("📄 Download PDF", data=pdf_bytes, file_name=f"EA_Performance_{period_label}.pdf", mime="application/pdf")

