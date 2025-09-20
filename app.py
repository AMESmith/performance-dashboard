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
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    h1, h2, h3, h4 {
        color: #000000;
    }
    /* KPI Cards */
    .card {
        background-color: #000000;  /* black background */
        border: 2px solid #ffb000;  /* EA Insights gold border */
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 5px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.4);
    }
    .card h3 {
        color: #FFFFFF;   /* white titles */
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .card p {
        font-size: 24px;
        font-weight: 800;
        color: #ffb000;   /* gold numbers */
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

def quarterly_enriched(df: pd.DataFrame) -> pd.DataFrame:
    q = df.copy()
    q["Email Efficiency"] = np.where(q["Emails Received"]>0, q["Emails Sent"]/q["Emails Received"], 0.0)
    q["Invite Action Rate"] = np.where(q["Emails Received"]>0, q["Invites Actioned"]/q["Emails Received"], 0.0)
    q["Travel Impact Score"] = q["Domestic Trips"] + q["International Trips"]*3.0
    q["Delegation Total"] = q["Tasks Delegated"] + q["Tasks Automated"] + q["Tasks Directly Executed"]
    q["% Delegated"] = np.where(q["Delegation Total"]>0, q["Tasks Delegated"]/q["Delegation Total"]*100.0, 0.0)
    q["% Automated"] = np.where(q["Delegation Total"]>0, q["Tasks Automated"]/q["Delegation Total"]*100.0, 0.0)
    q["% Direct"] = np.where(q["Delegation Total"]>0, q["Tasks Directly Executed"]/q["Delegation Total"]*100.0, 0.0)
    return q

def quarterly_summary(df: pd.DataFrame, year: str) -> pd.DataFrame:
    metrics = ["Emails Received","Emails Sent","Invites Actioned",
               "Meetings Scheduled","Domestic Trips","International Trips",
               "Projects","Events","Expense Reports Processed","Approvals Routed"]
    d = df[df["Quarter"].astype(str).str.startswith(f"{year}-Q")].copy()
    rows = []
    totals = {m:0 for m in metrics}
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

# ---------------------- Exports ----------------------
def export_to_excel(scenarios_dict, execs_supported_dict, quarterly_df, period):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for scen, df in scenarios_dict.items():
            compute_scores_for_scenario(df, execs_supported_dict.get(scen,1)).to_excel(writer, index=False, sheet_name=scen[:31])
        quarterly_df.to_excel(writer, index=False, sheet_name="Quarterly Raw")
    output.seek(0)
    return output

def export_to_pdf(scenarios_dict, execs_supported_dict, period, heatmap_img_bytes=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []
    totals = scenario_totals(scenarios_dict, execs_supported_dict)
    elements.append(Paragraph(TITLE, styles["Title"]))
    elements.append(Paragraph(f"Reporting Period: {period}", styles["Normal"]))
    tbl = [["Scenario","Total Score"]]+[[k,f"{v:.2f}"] for k,v in totals.items()]
    t = Table(tbl)
    t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.black)]))
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.markdown("### Controls")
    period = st.text_input("Reporting period", value=date.today().strftime("%Y-%m"))
    selected_scenario = st.selectbox("Scenario to edit", SCENARIOS, index=2)
    hourly_rate = st.number_input("EA Hourly Rate ($/hr)", min_value=0.0, value=60.0, step=5.0)

    st.markdown("### Downloads")
    excel_bytes = export_to_excel(st.session_state.scenarios, st.session_state.execs_supported, st.session_state.quarterly, period)
    st.download_button("📊 Download Excel", data=excel_bytes, file_name=f"EA_Performance_{period}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    pdf_bytes = export_to_pdf(st.session_state.scenarios, st.session_state.execs_supported, period, st.session_state.heatmap_img)
    st.download_button("📄 Download PDF", data=pdf_bytes, file_name=f"EA_Performance_{period}.pdf", mime="application/pdf")

# ---------------------- Header + KPI Cards ----------------------
st.markdown(f"## {TITLE}")

totals = scenario_totals(st.session_state.scenarios, st.session_state.execs_supported)
exp_total, tgt_total, act_total = totals["Executive Expectations"], totals["Targeted Performance"], totals["Actual Performance"]
align_val = alignment_pct(act_total, exp_total)

sat_proj = 85.0
sav_redund, sav_speed = 1000.0, 2000.0

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

# ---------------------- Support Alignment ----------------------
st.markdown("### Support Alignment")
cols = st.columns(3)
for i, scen in enumerate(SCENARIOS):
    with cols[i]:
        st.session_state.execs_supported[scen] = st.number_input(
            f"{scen} – Executives Supported", min_value=1, step=1,
            value=int(st.session_state.execs_supported.get(scen, 1))
        )

st.divider()

# ---------------------- Quarterly Workload Analytics ----------------------
st.markdown("### Quarterly Workload Analytics")
q = st.session_state.quarterly
quarter_options = list(q["Quarter"].astype(str).unique())
current_q = st.selectbox("Quarter to edit", options=quarter_options + ["+ Add new quarter"], index=len(quarter_options)-1 if quarter_options else 0)

if current_q == "+ Add new quarter":
    new_label = st.text_input("New quarter label (YYYY-Q#)", value=_default_quarter_label())
    if st.button("➕ Confirm Add Quarter"):
        if new_label and new_label not in q["Quarter"].values:
            base = q.iloc[-1].copy()
            base["Quarter"] = new_label
            st.session_state.quarterly = pd.concat([q, pd.DataFrame([base])], ignore_index=True)
        st.experimental_rerun()

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

if current_q != "+ Add new quarter":
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

    # Quarter KPIs
    q_enriched = quarterly_enriched(st.session_state.quarterly)
    if current_q in q_enriched["Quarter"].values:
        row = q_enriched[q_enriched["Quarter"]==current_q].iloc[0]
        qa, qb, qc, qd = st.columns(4)
        with qa: st.markdown(f"<div class='card'><h3>Email Efficiency</h3><p>{row['Email Efficiency']:.2f}</p></div>", unsafe_allow_html=True)
        with qb: st.markdown(f"<div class='card'><h3>Invite Action Rate</h3><p>{row['Invite Action Rate']:.2f}</p></div>", unsafe_allow_html=True)
        with qc: st.markdown(f"<div class='card'><h3>Travel Impact</h3><p>{row['Travel Impact Score']:.1f}</p></div>", unsafe_allow_html=True)
        with qd: st.markdown(f"<div class='card'><h3>Deleg/AUTO/DIRECT</h3><p>{row['% Delegated']:.0f}%/{row['% Automated']:.0f}%/{row['% Direct']:.0f}%</p></div>", unsafe_allow_html=True)

    # Quarterly summary
    years_present = sorted(set(q["Quarter"].astype(str).str.slice(0,4)))
    sel_year = st.selectbox("Summary year", years_present, index=len(years_present)-1 if years_present else 0)
    if sel_year:
        qs = quarterly_summary(st.session_state.quarterly, sel_year)
        st.markdown(f"**{sel_year


