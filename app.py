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

# Scenario rows (kept lean; detailed workload comes from quarterly section)
CATEGORIES = ["Executive Support", "Operational Support"]

# Months list (FIX for sidebar bug)
MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
MONTH_TO_Q = {
    "January":"Q1","February":"Q1","March":"Q1",
    "April":"Q2","May":"Q2","June":"Q2",
    "July":"Q3","August":"Q3","September":"Q3",
    "October":"Q4","November":"Q4","December":"Q4"
}

# ---------------------- Workload Weights (Integrated Model) ----------------------
WORKLOAD_WEIGHTS = {
    # Workforce
    "Emails Received": 0.05, "Emails Sent": 0.05, "Invites Actioned": 0.1,
    "Meetings Scheduled": 0.2, "Reschedules": 0.3, "Meeting Notes Prepared": 0.5,
    # Travel
    "Domestic Trips": 1.0, "International Trips": 3.0,
    # People
    "Onboardings Supported": 0.5, "Trainings Facilitated": 0.5,
    # Finance
    "Expense Reports Processed": 0.25, "Approvals Routed": 0.25,
    # Ops
    "Projects": 1.0, "Events": 2.0,
    # Delegation / Automation (negative reduces load)
    "Tasks Delegated": -0.2, "Tasks Automated": -0.3, "Tasks Directly Executed": 0.2,
    # Work Pattern
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
            # Workforce
            "Emails Received": 0, "Emails Sent": 0, "Invites Actioned": 0,
            "Meetings Scheduled": 0, "Reschedules": 0, "Meeting Notes Prepared": 0,
            # Travel
            "Domestic Trips": 0, "International Trips": 0,
            # People
            "Onboardings Supported": 0, "Trainings Facilitated": 0,
            # Finance
            "Expense Reports Processed": 0, "Approvals Routed": 0,
            # Ops
            "Projects": 0, "Events": 0,
            # Delegation / Automation
            "Tasks Delegated": 0, "Tasks Automated": 0, "Tasks Directly Executed": 0,
            # Work pattern
            "Reactive Work Hours": 0.0, "Overtime Hours": 0.0,
        }])
init_state()

# ---------------------- Sidebar (FIXED) ----------------------
with st.sidebar:
    st.header("Settings")
    period_type = st.radio("Period Type", ["Monthly", "Quarterly"], horizontal=True)
    if period_type == "Monthly":
        month = st.selectbox("Month", MONTHS)
        year = st.number_input("Year", min_value=2000, max_value=2100, value=date.today().year)
        period_label = f"{month} {year}"
        selected_quarter_label = f"{year}-{MONTH_TO_Q[month]}"
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
    # Scale Executive Support by # of executives
    mask_exec = (out["Category"] == "Executive Support")
    out.loc[mask_exec, "WeightedScore"] *= max(1, int(execs_supported))
    return out

def scenario_totals(scenarios_dict, execs_supported_dict):
    return {
        s: compute_scores_for_scenario(df, execs_supported_dict.get(s,1))["WeightedScore"].sum()
        for s, df in scenarios_dict.items()
    }

def compute_weighted_load(scenarios_dict, execs_supported_dict, quarterly_df, period_label, annual_year=None):
    """Exec baseline + weighted workload (people, ops, finance, etc.)."""
    base = scenario_totals(scenarios_dict, execs_supported_dict)
    workload_score = 0.0
    if annual_year:
        df_year = quarterly_df[quarterly_df["Quarter"].astype(str).str.startswith(str(annual_year))]
        for _, row in df_year.iterrows():
            for k, v in row.items():
                if k in WORKLOAD_WEIGHTS:
                    workload_score += v * WORKLOAD_WEIGHTS[k]
    else:
        qrow = quarterly_df[quarterly_df["Quarter"] == period_label]
        if not qrow.empty:
            for k, v in qrow.iloc[0].to_dict().items():
                if k in WORKLOAD_WEIGHTS:
                    workload_score += v * WORKLOAD_WEIGHTS[k]
    weighted = {s: base[s] + workload_score for s in base}
    return weighted, workload_score

# ---------------------- Header ----------------------
st.title(TITLE)
st.markdown(f"**Reporting Period:** {period_label}")

# ---------------------- KPI Overview (percentages, no decimals; points secondary) ----------------------
if annual_rollup:
    totals, workload_score = compute_weighted_load(
        st.session_state.scenarios, st.session_state.execs_supported,
        st.session_state.quarterly, period_label, annual_year=year
    )
    view_label = f"Annual {year}"
else:
    totals, workload_score = compute_weighted_load(
        st.session_state.scenarios, st.session_state.execs_supported,
        st.session_state.quarterly, selected_quarter_label
    )
    view_label = period_label

exp_total = totals["Executive Expectations"]
tgt_total = totals["Targeted Performance"]
act_total = totals["Actual Performance"]

target_pct = int(round((tgt_total/exp_total*100))) if exp_total > 0 else 0
actual_pct = int(round((act_total/exp_total*100))) if exp_total > 0 else 0

k1,k2,k3,k4,k5 = st.columns(5)
with k1:
    st.markdown(f"<div class='card'><h3>Expectation</h3><p class='value'>100%</p><p class='sub'>{round(exp_total)} pts</p></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='card'><h3>Target</h3><p class='value'>{target_pct}%</p><p class='sub'>{round(tgt_total)} pts</p></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='card'><h3>Actual</h3><p class='value'>{actual_pct}%</p><p class='sub'>{round(act_total)} pts</p></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='card'><h3>Alignment</h3><p class='value'>{actual_pct}%</p><p class='sub'>{round(act_total)} pts delivered</p></div>", unsafe_allow_html=True)
with k5:
    cost_est = act_total * float(hourly_rate)
    st.markdown(f"<div class='card'><h3>Workload</h3><p class='value'>{round(workload_score)} pts</p><p class='sub'>People, Ops, Finance</p></div>", unsafe_allow_html=True)

st.divider()

# ---------------------- Performance Volume (Scenario Inputs) ----------------------
st.markdown("### Performance Volume")
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

# ---------------------- Support Alignment (Executives Supported) ----------------------
st.markdown("### Support Alignment")
cols = st.columns(3)
for i, scen in enumerate(SCENARIOS):
    with cols[i]:
        st.session_state.execs_supported[scen] = st.number_input(
            f"{scen} – Executives Supported",
            min_value=1, step=1,
            value=int(st.session_state.execs_supported.get(scen, 1))
        )

st.divider()

# ---------------------- Quarterly Workload Analytics (detailed inputs) ----------------------
st.markdown("### Quarterly Workload Analytics")
q = st.session_state.quarterly

# Ensure selected quarter exists in the table
if selected_quarter_label not in q["Quarter"].astype(str).values:
    base = q.iloc[-1].copy()
    base["Quarter"] = selected_quarter_label
    st.session_state.quarterly = pd.concat([q, pd.DataFrame([base])], ignore_index=True)
    q = st.session_state.quarterly

quarter_options = list(q["Quarter"].astype(str).unique())
current_q = st.selectbox("Quarter to edit", options=quarter_options, index=quarter_options.index(selected_quarter_label))

def edit_block(title, fields):
    st.markdown(f"**{title}**")
    cols = st.columns(2)
    df = st.session_state.quarterly
    idxs = df.index[df["Quarter"]==current_q]
    if len(idxs)==0: return
    idx = idxs[0]
    for i, field in enumerate(fields):
        with cols[i % 2]:
            if "Hours" in field:
                df.at[idx, field] = st.number_input(field, min_value=0.0, step=0.5, value=float(df.at[idx, field]))
            else:
                df.at[idx, field] = st.number_input(field, min_value=0, step=1, value=int(df.at[idx, field]))
    st.session_state.quarterly = df

with st.expander("Email", expanded=True):
    edit_block("Email", ["Emails Received", "Emails Sent", "Invites Actioned"])
with st.expander("Meetings (Calendar Engineering)", expanded=True):
    edit_block("Meetings", ["Meetings Scheduled", "Reschedules", "Meeting Notes Prepared"])
with st.expander("Travel", expanded=True):
    edit_block("Travel", ["Domestic Trips", "International Trips"])
with st.expander("People Ops", expanded=False):
    edit_block("People Ops", ["Onboardings Supported", "Trainings Facilitated"])
with st.expander("Finance", expanded=False):
    edit_block("Finance", ["Expense Reports Processed", "Approvals Routed"])
with st.expander("Operational Support (Quarterly)", expanded=True):
    edit_block("Operational Support", ["Projects", "Events"])
with st.expander("Automation / Delegation", expanded=True):
    edit_block("Automation / Delegation", ["Tasks Delegated", "Tasks Automated", "Tasks Directly Executed"])
with st.expander("Work Pattern", expanded=False):
    edit_block("Work Pattern", ["Reactive Work Hours", "Overtime Hours"])

# Derived quarter KPIs
def quarterly_enriched(df: pd.DataFrame) -> pd.DataFrame:
    qd = df.copy()
    qd["Email Efficiency"] = np.where(qd["Emails Received"]>0, qd["Emails Sent"]/qd["Emails Received"], 0.0)
    qd["Invite Action Rate"] = np.where(qd["Emails Received"]>0, qd["Invites Actioned"]/qd["Emails Received"], 0.0)
    qd["Travel Impact Score"] = qd["Domestic Trips"] + qd["International Trips"]*3.0
    qd["Delegation Total"] = qd["Tasks Delegated"] + qd["Tasks Automated"] + qd["Tasks Directly Executed"]
    qd["% Delegated"] = np.where(qd["Delegation Total"]>0, qd["Tasks Delegated"]/qd["Delegation Total"]*100.0, 0.0)
    qd["% Automated"] = np.where(qd["Delegation Total"]>0, qd["Tasks Automated"]/qd["Delegation Total"]*100.0, 0.0)
    qd["% Direct"] = np.where(qd["Delegation Total"]>0, qd["Tasks Directly Executed"]/qd["Delegation Total"]*100.0, 0.0)
    return qd

q_enriched = quarterly_enriched(st.session_state.quarterly)
if current_q in q_enriched["Quarter"].values:
    row = q_enriched[q_enriched["Quarter"]==current_q].iloc[0]
    qa, qb, qc, qd = st.columns(4)
    with qa: st.markdown(f"<div class='card'><h3>Email Efficiency</h3><p class='value'>{row['Email Efficiency']:.2f}</p><p class='sub'>Sent / Received</p></div>", unsafe_allow_html=True)
    with qb: st.markdown(f"<div class='card'><h3>Invite Action Rate</h3><p class='value'>{row['Invite Action Rate']:.2f}</p><p class='sub'>Actions / Received</p></div>", unsafe_allow_html=True)
    with qc: st.markdown(f"<div class='card'><h3>Travel Impact</h3><p class='value'>{row['Travel Impact Score']:.1f}</p><p class='sub'>Intl weighted ×3</p></div>", unsafe_allow_html=True)
    with qd: st.markdown(f"<div class='card'><h3>Deleg/AUTO/DIRECT</h3><p class='value'>{row['% Delegated']:.0f}% / {row['% Automated']:.0f}% / {row['% Direct']:.0f}%</p><p class='sub'>Share of tasks</p></div>", unsafe_allow_html=True)

# Annual Rollup table + Quarterly comparison chart
def quarterly_summary(df: pd.DataFrame, year: str) -> pd.DataFrame:
    metrics = ["Emails Received","Emails Sent","Invites Actioned",
               "Meetings Scheduled","Domestic Trips","International Trips",
               "Projects","Events","Expense Reports Processed","Approvals Routed"]
    d = df[df["Quarter"].astype(str).str.startswith(f"{year}-Q")].copy()
    rows, totals = [], {m:0 for m in metrics}
    for qnum in [1,2,3,4]:
        qlabel = f"{year}-Q{qnum}"
        dq = d[d["Quarter"]==qlabel]
        vals = {m:int(dq[m].sum()) if not dq.empty else 0 for m in metrics}
        vals["Quarter"] = qlabel
        rows.append(vals)
        for m in metrics: totals[m]+=vals[m]
    ytd = {"Quarter":"YTD"} | totals
    rows.append(ytd)
    return pd.DataFrame(rows, columns=["Quarter"]+metrics)

if annual_rollup:
    st.subheader(f"Annual Rollup — {year}")
    years_present = sorted(set(q["Quarter"].astype(str).str.slice(0,4)))
    if str(year) in years_present:
        qs = quarterly_summary(st.session_state.quarterly, str(year))
        st.dataframe(qs, use_container_width=True)
        # Quarterly comparison using integrated totals
        totals_by_q = []
        for qnum in [1,2,3,4]:
            qlab = f"{year}-Q{qnum}"
            qtot, _ = compute_weighted_load(st.session_state.scenarios, st.session_state.execs_supported, st.session_state.quarterly, qlab)
            totals_by_q.append({"Quarter": qlab, **qtot})
        comp = pd.DataFrame(totals_by_q)
        st.subheader("Quarterly Comparison (Integrated Load)")
        st.bar_chart(comp.set_index("Quarter"))

st.divider()

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
    output.seek(0)
    return output

def export_to_pdf(scenarios_dict, execs_supported_dict, period, totals, workload_score, kpi_percentages, quarterly_df, selected_quarter, annual_year=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    # Title + Period
    elements.append(Paragraph(TITLE, styles["Title"]))
    elements.append(Paragraph(f"Reporting Period: {period}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # KPI Overview
    elements.append(Paragraph("KPI Overview", styles["Heading2"]))
    kpi_table = [
        ["Metric", "Value", "Points"],
        ["Expectation", "100%", f"{round(totals['Executive Expectations'])}"],
        ["Target", f"{kpi_percentages['Target']}%", f"{round(totals['Targeted Performance'])}"],
        ["Actual", f"{kpi_percentages['Actual']}%", f"{round(totals['Actual Performance'])}"],
        ["Alignment", f"{kpi_percentages['Alignment']}%", f"{round(totals['Actual Performance'])} delivered"],
        ["Workload", f"{round(workload_score)} pts", "People, Ops, Finance impact"]
    ]
    t = Table(kpi_table, hAlign="LEFT")
    t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.black),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    # Scenario Totals
    elements.append(Paragraph("Scenario Totals", styles["Heading2"]))
    scen_tbl = [["Scenario","Total Score"]]+[[k,f"{v:.2f}"] for k,v in totals.items()]
    t = Table(scen_tbl, hAlign="LEFT")
    t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.black)]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    # Quarterly Details (only if not annual)
    if not annual_year:
        elements.append(Paragraph(f"Quarterly Workload — {selected_quarter}", styles["Heading2"]))
        row = quarterly_df[quarterly_df["Quarter"]==selected_quarter]
        if not row.empty:
            d = row.iloc[0].to_dict()
            q_tbl = [["Category","Value"]]+[[k,str(v)] for k,v in d.items() if k!="Quarter"]
            t = Table(q_tbl, hAlign="LEFT")
            t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.black)]))
            elements.append(t)
            elements.append(Spacer(1, 12))

    # Annual Rollup (if selected)
    if annual_year:
        elements.append(Paragraph(f"Annual Rollup — {annual_year}", styles["Heading2"]))
        df_year = quarterly_df[quarterly_df["Quarter"].astype(str).str.startswith(str(annual_year))]
        if not df_year.empty:
            year_tbl = [df_year.columns.tolist()] + df_year.astype(str).values.tolist()
            t = Table(year_tbl, hAlign="LEFT")
            t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.black)]))
            elements.append(t)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Export buttons
with st.sidebar:
    excel_bytes = export_to_excel(st.session_state.scenarios, st.session_state.execs_supported, st.session_state.quarterly, period_label)
    st.download_button("📊 Download Excel", data=excel_bytes, file_name=f"EA_Performance_{period_label}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    pdf_bytes = export_to_pdf(
        st.session_state.scenarios,
        st.session_state.execs_supported,
        period_label,
        totals,
        workload_score,
        {"Target": target_pct, "Actual": actual_pct, "Alignment": actual_pct},
        st.session_state.quarterly,
        selected_quarter_label,
        annual_year=year if annual_rollup else None
    )
    st.download_button("📄 Download PDF", data=pdf_bytes, file_name=f"EA_Performance_{period_label}.pdf", mime="application/pdf")

