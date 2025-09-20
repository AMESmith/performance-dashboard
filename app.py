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

SCENARIOS = ["Executive Expectations", "Targeted Performance", "Actual Performance"]
CATEGORIES = ["Executive Support", "Operational Support"]
LEVELS = ["Low", "Mid", "High"]
LEVEL_SCORES = {"Low": 1, "Mid": 2, "High": 3}

# ---------------------- Init State ----------------------
def _empty_perf_df():
    return pd.DataFrame({
        "Category": CATEGORIES,
        "Level": ["Mid"] * len(CATEGORIES),
        "Volume": [0] * len(CATEGORIES),
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

# ---------------------- Computations ----------------------
def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["LevelScore"] = out["Level"].map(LEVEL_SCORES).astype(float)
    out["WeightedScore"] = out["LevelScore"] * out["Volume"].astype(float)
    return out

def scenario_totals(scenarios_dict, execs_supported):
    totals = {}
    for scen, df in scenarios_dict.items():
        base = compute_scores(df)["WeightedScore"].sum()
        ex_df = df[df["Category"] == "Executive Support"].copy()
        if not ex_df.empty:
            ex_score = compute_scores(ex_df)["WeightedScore"].sum()
            total = (base - ex_score) + (ex_score * max(1, execs_supported.get(scen, 1)))
        else:
            total = base
        totals[scen] = total
    return totals

def alignment_pct(actual_total, expectation_total):
    return float(actual_total / expectation_total * 100.0) if expectation_total > 0 else 0.0

def make_heatmap_matrix(scenarios_dict):
    cats = CATEGORIES
    scens = SCENARIOS
    mat = np.zeros((len(cats), len(scens)), dtype=float)
    for j, scen in enumerate(scens):
        df = compute_scores(scenarios_dict[scen])
        for i, cat in enumerate(cats):
            mat[i, j] = df.loc[df["Category"]==cat, "WeightedScore"].sum()
    return cats, scens, mat

def risk_table(expect_df, actual_df):
    exp = compute_scores(expect_df)
    act = compute_scores(actual_df)
    rows = []
    for cat in CATEGORIES:
        e = exp.loc[exp["Category"]==cat, "WeightedScore"].sum()
        a = act.loc[act["Category"]==cat, "WeightedScore"].sum()
        rows.append({"Category": cat, "Expectation": e, "Actual": a, "Gap": e - a})
    return pd.DataFrame(rows).sort_values("Gap", ascending=False)

def expected_cost_savings(target_df, actual_df, hourly_rate=60.0):
    tgt = compute_scores(target_df)["WeightedScore"].sum()
    act = compute_scores(actual_df)["WeightedScore"].sum()
    redundancy = max(0.0, act - tgt)
    speed = max(0.0, tgt - act) * 0.5
    return redundancy * hourly_rate, speed * hourly_rate

def exec_satisfaction_projection(expect_df, actual_df):
    exp = compute_scores(expect_df)
    act = compute_scores(actual_df)
    vals = []
    for cat in CATEGORIES:
        e = exp.loc[exp["Category"]==cat, "WeightedScore"].sum()
        a = act.loc[act["Category"]==cat, "WeightedScore"].sum()
        if e > 0: vals.append(min(100.0, (a/e)*100.0))
    return np.mean(vals) if vals else 0.0

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

# ---------------------- Header + KPIs ----------------------
st.markdown(f"## {TITLE}")

totals = scenario_totals(st.session_state.scenarios, st.session_state.execs_supported)
exp_total, tgt_total, act_total = totals["Executive Expectations"], totals["Targeted Performance"], totals["Actual Performance"]
align_val = alignment_pct(act_total, exp_total)
sat_proj = exec_satisfaction_projection(st.session_state.scenarios["Executive Expectations"], st.session_state.scenarios["Actual Performance"])
sav_redund, sav_speed = expected_cost_savings(st.session_state.scenarios["Targeted Performance"], st.session_state.scenarios["Actual Performance"], hourly_rate)

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("Expectation Total", f"{exp_total:.1f}")
with k2: st.metric("Target Total", f"{tgt_total:.1f}")
with k3: st.metric("Actual Total", f"{act_total:.1f}")
with k4: st.metric("Alignment %", f"{align_val:.1f}%")
with k5: st.metric("Exec Satisfaction Projection", f"{sat_proj:.1f}%")

k6, k7, k8 = st.columns(3)
with k6: st.metric("Executives Supported (Actual)", st.session_state.execs_supported["Actual Performance"])
with k7: st.metric("Savings (Redundancy)", f"${sav_redund:,.0f}")
with k8: st.metric("Savings (Speed)", f"${sav_speed:,.0f}")

st.divider()

# ---------------------- Support Alignment ----------------------
st.markdown("### Support Alignment")
cols = st.columns(3)
for i, scen in enumerate(SCENARIOS):
    with cols[i]:
        st.session_state.execs_supported[scen] = st.number_input(
            f"{scen} – Executives Supported", min_value=1, step=1, value=int(st.session_state.execs_supported.get(scen, 1))
        )

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
        "Volume": st.column_config.NumberColumn(min_value=0, step=1),
    },
    key=f"perf_editor_{selected_scenario}",
)
st.session_state.scenarios[selected_scenario] = edited

st.divider()

# ---------------------- Quarterly Workload Analytics ----------------------
st.markdown("### Quarterly Workload Analytics")
q = st.session_state.quarterly
quarter_options = list(q["Quarter"].astype(str).unique())
current_q = st.selectbox("Quarter to edit", options=quarter_options + ["+ Add new quarter"], index=len(quarter_options)-1)

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
    idx = df.index[df["Quarter"]==current_q][0]
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

    q_enriched = quarterly_enriched(st.session_state.quarterly)
    if current_q in q_enriched["Quarter"].values:
        row = q_enriched[q["Quarter"]==current_q].iloc[0]
        qa, qb, qc, qd = st.columns(4)
        with qa: st.metric("Email Efficiency", f"{row['Email Efficiency']:.2f}")
        with qb: st.metric("Invite Action Rate", f"{row['Invite Action Rate']:.2f}")
        with qc: st.metric("Travel Impact Score", f"{row['Travel Impact Score']:.1f}")
        with qd: st.metric("Deleg/AUTO/DIRECT", f"{row['% Delegated']:.0f}%/{row['% Automated']:.0f}%/{row['% Direct']:.0f}%")

st.divider()

# ---------------------- Visuals ----------------------
if st.session_state.show_heatmap or st.session_state.show_charts:
    cats, scens, mat = make_heatmap_matrix(st.session_state.scenarios)

if st.session_state.show_heatmap:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(mat, aspect="auto", cmap="Greys")
    ax.set_xticks(np.arange(len(scens))); ax.set_xticklabels(scens, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(cats))); ax.set_yticklabels(cats)
    ax.set_title("Performance Heatmap (Greyscale)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center")
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.session_state.heatmap_img = buf

if st.session_state.show_charts:
    def donut(value, total, title):
        fig, ax = plt.subplots()
        parts = [value, max(total - value, 0)]
        ax.pie(parts, startangle=90, wedgeprops=dict(width=0.4))
        ax.set_title(title)
        st.pyplot(fig)
    cA, cB = st.columns(2)
    with cA: donut(act_total, max(exp_total, act_total), "Actual vs Expectation")
    with cB: donut(tgt_total, max(exp_total, tgt_total), "Target vs Expectation")

    rows = []
    for scen in SCENARIOS:
        df = compute_scores(st.session_state.scenarios[scen])
        by_cat = df.groupby("Category")["WeightedScore"].sum()
        for cat in CATEGORIES:
            rows.append({"Scenario": scen, "Category": cat, "WeightedScore": float(by_cat.get(cat, 0.0))})
    g = pd.DataFrame(rows)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    width = 0.35; x = np.arange(len(CATEGORIES))
    for i, scen in enumerate(SCENARIOS):
        vals = [g[(g["Scenario"]==scen) & (g["Category"]==c)]["WeightedScore"].sum() for c in CATEGORIES]
        ax2.bar(x + (i-1)*width, vals, width=width, label=scen)
    ax2.set_xticks(x); ax2.set_xticklabels(CATEGORIES, rotation=25, ha="right")
    ax2.set_title("Weighted Score by Category & Scenario")
    ax2.legend()
    st.pyplot(fig2)

# ---------------------- Risk Indicators ----------------------
st.markdown("### Risk Indicators (Gaps by Category)")
st.dataframe(
    risk_table(st.session_state.scenarios["Executive Expectations"], st.session_state.scenarios["Actual Performance"]),
    use_container_width=True
)

