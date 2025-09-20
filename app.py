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
CATEGORIES = [
    "Executive Support",
    "Project Support",
    "Event Support",
    "Training",
    "Onboarding",
    "Operational Support",
]
LEVELS = ["Low", "Mid", "High"]
LEVEL_SCORES = {"Low": 1, "Mid": 2, "High": 3}
PRIORITIES = ["Standard", "High", "Critical"]
PRIORITY_WEIGHTS = {"Standard": 1.0, "High": 1.5, "Critical": 2.0}
SCENARIOS = ["Executive Expectations", "Targeted Performance", "Actual Performance"]

# Quarterly fields
QUARTERLY_COLUMNS = [
    "Quarter",                     # e.g., 2025-Q3
    "Emails Received",
    "Emails Sent",
    "Invites Actioned",
    "Meetings Scheduled",
    "Reschedules",
    "Meeting Notes Prepared",
    "Domestic Trips",
    "International Trips",
    "Onboardings Supported",
    "Trainings Facilitated",
    "Ad-hoc Projects",
    "Expense Reports Processed",
    "Approvals Routed",
    "Deep Work Hours",
    "Reactive Work Hours",
    "Overtime Hours",
    "Tasks Delegated",
    "Tasks Automated",
    "Tasks Directly Executed"
]

# ---------------------- Helpers ----------------------
def empty_row():
    return {
        "Category": CATEGORIES[0],
        "Level": "Mid",
        "Volume": 0,
        "Priority": "Standard",
    }

def default_quarter_label():
    y = date.today().year
    m = date.today().month
    q = (m - 1) // 3 + 1
    return f"{y}-Q{q}"

def init_state():
    if "data" not in st.session_state:
        st.session_state.data = {
            "Executive Expectations": pd.DataFrame([empty_row()]),
            "Targeted Performance": pd.DataFrame([empty_row()]),
            "Actual Performance": pd.DataFrame([empty_row()]),
        }
    if "quarterly" not in st.session_state:
        st.session_state.quarterly = pd.DataFrame([{
            "Quarter": default_quarter_label(),
            "Emails Received": 0,
            "Emails Sent": 0,
            "Invites Actioned": 0,
            "Meetings Scheduled": 0,
            "Reschedules": 0,
            "Meeting Notes Prepared": 0,
            "Domestic Trips": 0,
            "International Trips": 0,
            "Onboardings Supported": 0,
            "Trainings Facilitated": 0,
            "Ad-hoc Projects": 0,
            "Expense Reports Processed": 0,
            "Approvals Routed": 0,
            "Deep Work Hours": 0.0,
            "Reactive Work Hours": 0.0,
            "Overtime Hours": 0.0,
            "Tasks Delegated": 0,
            "Tasks Automated": 0,
            "Tasks Directly Executed": 0
        }], columns=QUARTERLY_COLUMNS)

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LevelScore"] = df["Level"].map(LEVEL_SCORES).astype(float)
    df["PriorityWeight"] = df["Priority"].map(PRIORITY_WEIGHTS).astype(float)
    df["WeightedScore"] = df["LevelScore"] * df["Volume"].astype(float)
    df["AdjustedScore"] = df["WeightedScore"] * df["PriorityWeight"]
    return df

def scenario_totals(data_dict):
    totals = {}
    for scen, df in data_dict.items():
        s = compute_scores(df)["AdjustedScore"].sum()
        totals[scen] = float(s)
    return totals

def alignment_pct(actual_total, expectation_total):
    if expectation_total <= 0:
        return 0.0
    return float(actual_total / expectation_total * 100)

def make_heatmap_matrix(data_dict):
    cats = CATEGORIES
    scens = SCENARIOS
    matrix = np.zeros((len(cats), len(scens)), dtype=float)
    for j, scen in enumerate(scens):
        df = compute_scores(data_dict[scen])
        for i, cat in enumerate(cats):
            matrix[i, j] = df.loc[df["Category"] == cat, "AdjustedScore"].sum()
    return cats, scens, matrix

def monte_carlo_probability(data_actual, exp_total, iterations=2000, vol_var_pct=20, level_flip_prob=10):
    if exp_total <= 0:
        return 0.0, np.array([])
    base = compute_scores(data_actual)
    arr = np.zeros(int(iterations), dtype=float)

    def level_to_idx(l): return LEVELS.index(l)
    def idx_to_level(i): return LEVELS[int(np.clip(i, 0, 2))]

    for i in range(int(iterations)):
        rows = []
        for _, r in base.iterrows():
            vol_noise = np.random.normal(loc=0.0, scale=vol_var_pct/100.0)
            sim_vol = max(0, int(round(r["Volume"] * (1 + vol_noise))))
            idx = level_to_idx(r["Level"])
            if np.random.rand() < (level_flip_prob/100.0):
                idx += np.random.choice([-1, 1])
            sim_level = idx_to_level(idx)
            rows.append({
                "Category": r["Category"],
                "Level": sim_level,
                "Volume": sim_vol,
                "Priority": r["Priority"]
            })
        sim_df = pd.DataFrame(rows)
        sim_total = compute_scores(sim_df)["AdjustedScore"].sum()
        arr[i] = 100.0 * (sim_total / exp_total)
    prob_hit = float(np.mean(arr >= 100.0) * 100.0)
    return prob_hit, arr

def expected_cost_savings(data_target, data_actual, cost_per_unit=150.0):
    tgt = data_target.copy(); act = data_actual.copy()
    for df in (tgt, act):
        df["LevelScore"] = df["Level"].map(LEVEL_SCORES).astype(float)
        df["Units"] = df["LevelScore"] * df["Volume"].astype(float)
    tgt_total = tgt["Units"].sum(); act_total = act["Units"].sum()
    redundancy_units = max(0.0, act_total - tgt_total)
    savings_redundancy = redundancy_units * cost_per_unit
    potential_gain_units = max(0.0, tgt_total - act_total)
    savings_speed = potential_gain_units * cost_per_unit * 0.5
    return float(savings_redundancy), float(savings_speed), {
        "target_units": float(tgt_total),
        "actual_units": float(act_total),
        "redundancy_units": float(redundancy_units),
        "potential_gain_units": float(potential_gain_units)
    }

def executive_satisfaction_projection(data_expect, data_actual):
    exp = compute_scores(data_expect); act = compute_scores(data_actual)
    sats = []
    for cat in CATEGORIES:
        exp_val = exp.loc[exp["Category"] == cat, "AdjustedScore"].sum()
        act_val = act.loc[act["Category"] == cat, "AdjustedScore"].sum()
        if exp_val > 0:
            sats.append(min(100.0, (act_val / exp_val) * 100.0))
    if len(sats) == 0: return 0.0
    return float(np.mean(sats))

def risk_table(data_expect, data_actual):
    exp = compute_scores(data_expect); act = compute_scores(data_actual)
    rows = []
    for cat in CATEGORIES:
        exp_val = exp.loc[exp["Category"] == cat, "AdjustedScore"].sum()
        act_val = act.loc[act["Category"] == cat, "AdjustedScore"].sum()
        gap = exp_val - act_val
        rows.append({"Category": cat, "Expectation": exp_val, "Actual": act_val, "Gap": gap})
    df = pd.DataFrame(rows).sort_values("Gap", ascending=False)
    return df

# ---------- Quarterly computations ----------
def quarterly_enriched(dfq: pd.DataFrame) -> pd.DataFrame:
    df = dfq.copy()
    for col in QUARTERLY_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Ratios & scores
    df["Email Efficiency"] = np.where(df["Emails Received"]>0,
                                      df["Emails Sent"] / df["Emails Received"], 0.0)
    df["Invite Action Rate"] = np.where(df["Emails Received"]>0,
                                        df["Invites Actioned"] / df["Emails Received"], 0.0)
    df["Travel Impact Score"] = df["Domestic Trips"] * 1.0 + df["International Trips"] * 3.0

    # Workload mix proxy (arbitrary but consistent units)
    df["Email Touches"] = df["Emails Received"] + df["Emails Sent"]
    df["Calendar Orchestration"] = df["Meetings Scheduled"] + df["Reschedules"]
    df["Project Load"] = df["Ad-hoc Projects"]
    df["People Ops"] = df["Onboardings Supported"] + df["Trainings Facilitated"]
    df["Finance Ops"] = df["Expense Reports Processed"] + df["Approvals Routed"]
    df["Delegation Total"] = df["Tasks Delegated"] + df["Tasks Automated"] + df["Tasks Directly Executed"]

    # Delegation ratios
    df["% Delegated"] = np.where(df["Delegation Total"]>0, df["Tasks Delegated"] / df["Delegation Total"] * 100.0, 0.0)
    df["% Automated"] = np.where(df["Delegation Total"]>0, df["Tasks Automated"] / df["Delegation Total"] * 100.0, 0.0)
    df["% Direct"] = np.where(df["Delegation Total"]>0, df["Tasks Directly Executed"] / df["Delegation Total"] * 100.0, 0.0)

    # Productivity health indicators
    df["Focus Ratio"] = np.where((df["Reactive Work Hours"]+df["Deep Work Hours"])>0,
                                 df["Deep Work Hours"]/(df["Reactive Work Hours"]+df["Deep Work Hours"]), 0.0)
    df["Overtime Ratio"] = np.where((df["Deep Work Hours"]+df["Reactive Work Hours"])>0,
                                    df["Overtime Hours"]/(df["Deep Work Hours"]+df["Reactive Work Hours"]), 0.0)
    return df

def support_mix_row(dfq_enriched: pd.DataFrame, quarter: str):
    row = dfq_enriched.loc[dfq_enriched["Quarter"] == quarter]
    if row.empty:
        return {"Email Touches":0, "Calendar Orchestration":0, "Travel Impact Score":0,
                "Project Load":0, "People Ops":0, "Finance Ops":0}
    r = row.iloc[0]
    return {
        "Email Touches": float(r["Email Touches"]),
        "Calendar Orchestration": float(r["Calendar Orchestration"]),
        "Travel Impact Score": float(r["Travel Impact Score"]),
        "Project Load": float(r["Project Load"]),
        "People Ops": float(r["People Ops"]),
        "Finance Ops": float(r["Finance Ops"])
    }

def export_to_excel(data_dict, sim_stats=None, period="", quarterly_df=None):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for scen, df in data_dict.items():
            compute_scores(df).to_excel(writer, index=False, sheet_name=scen[:31])
        totals_df = pd.DataFrame([scenario_totals(data_dict)]).T.reset_index()
        totals_df.columns = ["Scenario", "Total Adjusted Score"]
        totals_df.to_excel(writer, index=False, sheet_name="Summary")
        if quarterly_df is not None:
            quarterly_df.to_excel(writer, index=False, sheet_name="Quarterly")
        if sim_stats is not None:
            sim_stats.to_excel(writer, index=False, sheet_name="Simulation Stats")
    output.seek(0)
    return output

def export_to_pdf(data_dict, totals, align_pct_val, heatmap_img_bytes, period, quarterly_kpi_table=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph(TITLE, styles['Title']))
    elements.append(Paragraph(f"Reporting Period: {period}", styles['Normal']))
    elements.append(Spacer(1, 10))

    # Totals table
    tbl = [["Scenario", "Total Adjusted Score"]]
    for k, v in totals.items():
        tbl.append([k, f"{v:.1f}"])
    tbl.append(["Alignment % (Actual vs Expectation)", f"{align_pct_val:.1f}%"])
    table = Table(tbl, hAlign="LEFT")
    table.setStyle(TableStyle([("BOX",(0,0),(-1,-1),0.5,colors.black),
                               ("GRID",(0,0),(-1,-1),0.25,colors.grey),
                               ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    elements.append(table)
    elements.append(Spacer(1, 10))

    if quarterly_kpi_table is not None:
        elements.append(Paragraph("Quarterly KPIs", styles["Heading3"]))
        qtable = Table(quarterly_kpi_table, hAlign="LEFT")
        qtable.setStyle(TableStyle([("BOX",(0,0),(-1,-1),0.5,colors.black),
                                    ("GRID",(0,0),(-1,-1),0.25,colors.grey),
                                    ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
        elements.append(qtable)
        elements.append(Spacer(1, 10))

    # Add heatmap image if provided
    if heatmap_img_bytes is not None:
        from reportlab.platypus import Image
        img = Image(heatmap_img_bytes)
        img._restrictSize(480, 320)
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Performance Heatmap", styles["Heading3"]))
        elements.append(img)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------------- UI ----------------------
init_state()
st.markdown(f"## {TITLE}")

with st.sidebar:
    st.markdown("### Settings")
    period = st.text_input("Reporting period", value=date.today().strftime("%Y-%m"))
    selected_scenario = st.selectbox("Preset Scenario (for quick editing)", SCENARIOS, index=2)
    st.caption("Tip: Use the preset selector to focus your edits on one scenario at a time.")

    st.markdown("### Simulation Settings")
    run_sim = st.checkbox("Enable Simulation", value=False)
    iterations = st.number_input("Iterations", min_value=200, max_value=20000, value=2000, step=100, disabled=not run_sim)
    vol_variation = st.slider("Volume variation (+/- %)", 0, 100, 20, 5, disabled=not run_sim)
    level_flip_prob = st.slider("Level change probability (%)", 0, 100, 10, 5, disabled=not run_sim)

    st.markdown("### Cost Model (for $ savings)")
    cost_per_unit = st.number_input("Cost per workload unit ($)", min_value=0.0, value=150.0, step=10.0)

# KPI Cards
totals = scenario_totals(st.session_state.data)
exp_total = totals.get("Executive Expectations", 0.0)
tgt_total = totals.get("Targeted Performance", 0.0)
act_total = totals.get("Actual Performance", 0.0)
align = alignment_pct(act_total, exp_total)

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("Expectation Total", f"{exp_total:.1f}")
with k2: st.metric("Target Total", f"{tgt_total:.1f}")
with k3: st.metric("Actual Total", f"{act_total:.1f}")
with k4: st.metric("Alignment % (Actual vs Expectation)", f"{align:.1f}%")
with k5:
    sat_proj = executive_satisfaction_projection(st.session_state.data["Executive Expectations"],
                                                 st.session_state.data["Actual Performance"])
    st.metric("Executive Satisfaction Projection", f"{sat_proj:.1f}%")

st.divider()

# ---------------------- Interactive Inputs ----------------------
st.markdown("### Interactive Inputs")
st.caption("Use the drop-downs to define rows. Click **Add Row** to add more entries. Repeat for each scenario.")

c1, c2, c3 = st.columns(3)

def editor_for(scen, container):
    df = st.session_state.data[scen]
    with container:
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Category": st.column_config.SelectboxColumn(options=CATEGORIES, required=True),
                "Level": st.column_config.SelectboxColumn(options=LEVELS, required=True),
                "Volume": st.column_config.NumberColumn(min_value=0, step=1),
                "Priority": st.column_config.SelectboxColumn(options=PRIORITIES, required=True),
            },
            key=f"editor_{scen}",
        )
        st.session_state.data[scen] = edited

editor_for("Executive Expectations", c1)
editor_for("Targeted Performance", c2)
editor_for("Actual Performance", c3)

st.divider()

# ---------------------- Quarterly Workload Section ----------------------
st.markdown("## Quarterly Workload Analytics")
st.caption("Capture the breadth of EA support: email intensity, calendar orchestration, travel, people ops, finance ops, and delegation dynamics.")

q_editor = st.data_editor(
    st.session_state.quarterly,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Quarter": st.column_config.TextColumn(help="Format: YYYY-Q#"),
        "Emails Received": st.column_config.NumberColumn(min_value=0, step=1),
        "Emails Sent": st.column_config.NumberColumn(min_value=0, step=1),
        "Invites Actioned": st.column_config.NumberColumn(min_value=0, step=1),
        "Meetings Scheduled": st.column_config.NumberColumn(min_value=0, step=1),
        "Reschedules": st.column_config.NumberColumn(min_value=0, step=1),
        "Meeting Notes Prepared": st.column_config.NumberColumn(min_value=0, step=1),
        "Domestic Trips": st.column_config.NumberColumn(min_value=0, step=1),
        "International Trips": st.column_config.NumberColumn(min_value=0, step=1),
        "Onboardings Supported": st.column_config.NumberColumn(min_value=0, step=1),
        "Trainings Facilitated": st.column_config.NumberColumn(min_value=0, step=1),
        "Ad-hoc Projects": st.column_config.NumberColumn(min_value=0, step=1),
        "Expense Reports Processed": st.column_config.NumberColumn(min_value=0, step=1),
        "Approvals Routed": st.column_config.NumberColumn(min_value=0, step=1),
        "Deep Work Hours": st.column_config.NumberColumn(min_value=0.0, step=0.5),
        "Reactive Work Hours": st.column_config.NumberColumn(min_value=0.0, step=0.5),
        "Overtime Hours": st.column_config.NumberColumn(min_value=0.0, step=0.5),
        "Tasks Delegated": st.column_config.NumberColumn(min_value=0, step=1),
        "Tasks Automated": st.column_config.NumberColumn(min_value=0, step=1),
        "Tasks Directly Executed": st.column_config.NumberColumn(min_value=0, step=1),
    },
    key="quarterly_editor"
)
st.session_state.quarterly = q_editor

q_enriched = quarterly_enriched(st.session_state.quarterly)

# KPIs for the most recent quarter (last row)
if len(q_enriched) > 0:
    last = q_enriched.iloc[-1]
    qa, qb, qc, qd, qe, qf = st.columns(6)
    with qa: st.metric("Email Efficiency", f"{last['Email Efficiency']:.2f}")
    with qb: st.metric("Invites Action Rate", f"{last['Invite Action Rate']:.2f}")
    with qc: st.metric("Travel Impact Score", f"{last['Travel Impact Score']:.1f}")
    with qd: st.metric("Focus Ratio", f"{last['Focus Ratio']:.2f}")
    with qe: st.metric("Overtime Ratio", f"{last['Overtime Ratio']:.2f}")
    with qf: st.metric("Delegated / Automated / Direct", f"{last['% Delegated']:.0f}% / {last['% Automated']:.0f}% / {last['% Direct']:.0f}%")

# Support mix pie for selected quarter
st.markdown("### Support Mix")
quarter_options = list(q_enriched["Quarter"].astype(str).values) if len(q_enriched)>0 else [default_quarter_label()]
sel_quarter = st.selectbox("Select quarter for support mix", quarter_options, index=len(quarter_options)-1 if quarter_options else 0)
mix = support_mix_row(q_enriched, sel_quarter)
fig, ax = plt.subplots()
labels = list(mix.keys()); sizes = list(mix.values())
if sum(sizes) == 0:
    sizes = [1 for _ in sizes]  # avoid zero pie
ax.pie(sizes, labels=labels, autopct=None)
ax.set_title(f"Support Mix – {sel_quarter}")
st.pyplot(fig)

# Trendline charts (if multiple quarters)
if len(q_enriched) >= 2:
    st.markdown("### Quarter-over-Quarter Trends")
    # Emails
    fig1, ax1 = plt.subplots()
    ax1.plot(q_enriched["Quarter"], q_enriched["Emails Received"], marker="o")
    ax1.plot(q_enriched["Quarter"], q_enriched["Emails Sent"], marker="o")
    ax1.set_title("Emails Received vs Sent")
    ax1.set_xticklabels(q_enriched["Quarter"], rotation=25, ha="right")
    st.pyplot(fig1)

    # Travel impact
    fig2, ax2 = plt.subplots()
    ax2.plot(q_enriched["Quarter"], q_enriched["Travel Impact Score"], marker="o")
    ax2.set_title("Travel Impact Score by Quarter")
    ax2.set_xticklabels(q_enriched["Quarter"], rotation=25, ha="right")
    st.pyplot(fig2)

    # Delegation ratios
    fig3, ax3 = plt.subplots()
    ax3.plot(q_enriched["Quarter"], q_enriched["% Delegated"], marker="o")
    ax3.plot(q_enriched["Quarter"], q_enriched["% Automated"], marker="o")
    ax3.plot(q_enriched["Quarter"], q_enriched["% Direct"], marker="o")
    ax3.set_title("Delegation Mix by Quarter")
    ax3.set_xticklabels(q_enriched["Quarter"], rotation=25, ha="right")
    st.pyplot(fig3)

st.divider()

# ---------------------- Action Buttons (Bottom) ----------------------
b1, b2, b3, b4 = st.columns([1,1,1,1])
gen_heatmap = b1.button("Generate Heatmap")
gen_charts = b2.button("Generate Charts")
download_excel = b3.button("Download Excel Report")
download_pdf = b4.button("Download PDF Report")

# ---- Derived analytics for savings and probability ----
sav_redund, sav_speed, model_info = expected_cost_savings(
    st.session_state.data["Targeted Performance"],
    st.session_state.data["Actual Performance"],
    cost_per_unit=cost_per_unit,
)

prob_hit = 0.0
sim_arr = np.array([])
sim_stats_df = None
if run_sim:
    prob_hit, sim_arr = monte_carlo_probability(
        st.session_state.data["Actual Performance"],
        exp_total,
        iterations=iterations,
        vol_var_pct=vol_variation,
        level_flip_prob=level_flip_prob,
    )
    sim_stats_df = pd.DataFrame({
        "Metric": ["Alignment %"],
        "Mean": [float(np.mean(sim_arr)) if len(sim_arr)>0 else 0.0],
        "Std Dev": [float(np.std(sim_arr)) if len(sim_arr)>0 else 0.0],
        "P10": [float(np.percentile(sim_arr, 10)) if len(sim_arr)>0 else 0.0],
        "P50": [float(np.percentile(sim_arr, 50)) if len(sim_arr)>0 else 0.0],
        "P90": [float(np.percentile(sim_arr, 90)) if len(sim_arr)>0 else 0.0],
        "Prob >= 100%": [prob_hit],
    })

# Extra KPI row for transparency
t1, t2, t3, t4 = st.columns(4)
with t1: st.metric("Probability to Hit Productive Targets", f"{prob_hit:.1f}%")
with t2: st.metric("Expected $ Savings (Reduced Redundancy)", f"${sav_redund:,.0f}")
with t3: st.metric("Expected $ Savings (Speed to Execution)", f"${sav_speed:,.0f}")
with t4:
    risks = risk_table(st.session_state.data["Executive Expectations"], st.session_state.data["Actual Performance"])
    high_risk = int((risks["Gap"] > 0.0).sum())
    st.metric("High-Risk Categories (Gap > 0)", f"{high_risk}")

# ---------------------- Visuals ----------------------
heatmap_image_bytes = None

if gen_heatmap or gen_charts:
    cats, scens, mat = make_heatmap_matrix(st.session_state.data)

if gen_heatmap:
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(scens)))
    ax.set_xticklabels(scens, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_title("Performance Heatmap (Adjusted Scores)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center")
    st.pyplot(fig)

    # Save heatmap to bytes for PDF
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    heatmap_image_bytes = buf

if gen_charts:
    # Donut-style pies for Target vs Expectation and Actual vs Expectation
    def donut(value, total, title):
        fig, ax = plt.subplots()
        parts = [value, max(total - value, 0)]
        ax.pie(parts, startangle=90, wedgeprops=dict(width=0.4))
        ax.set_title(title)
        st.pyplot(fig)

    cA, cB = st.columns(2)
    with cA: donut(act_total, max(exp_total, act_total), "Actual vs Expectation")
    with cB: donut(tgt_total, max(exp_total, tgt_total), "Target vs Expectation")

    # Category bar chart by scenario
    grouped_rows = []
    for scen in SCENARIOS:
        df = compute_scores(st.session_state.data[scen])
        totals_cat = df.groupby("Category")["AdjustedScore"].sum()
        for cat in CATEGORIES:
            grouped_rows.append({"Scenario": scen, "Category": cat, "AdjustedScore": float(totals_cat.get(cat, 0.0))})
    grouped = pd.DataFrame(grouped_rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    x = np.arange(len(CATEGORIES))
    for si, scen in enumerate(SCENARIOS):
        vals = [grouped[(grouped["Scenario"] == scen) & (grouped["Category"] == c)]["AdjustedScore"].sum() for c in CATEGORIES]
        ax.bar(x + (si - 1) * width, vals, width=width, label=scen)
    ax.set_xticks(x); ax.set_xticklabels(CATEGORIES, rotation=25, ha="right")
    ax.set_ylabel("AdjustedScore")
    ax.set_title("Adjusted Score by Category & Scenario")
    ax.legend()
    st.pyplot(fig)

# ---------------------- Reports (Downloads) ----------------------
if download_excel:
    excel_bytes = export_to_excel(st.session_state.data, sim_stats_df, period, quarterly_df=q_enriched)
    st.download_button(
        "📊 Click to Save Excel",
        data=excel_bytes,
        file_name=f"EA_Performance_{period}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if download_pdf:
    # Build a simple quarterly KPI table for PDF (most recent quarter if available)
    q_kpi_tbl = None
    if len(q_enriched) > 0:
        last = q_enriched.iloc[-1]
        q_kpi_tbl = [
            ["Quarter", str(last["Quarter"])],
            ["Email Efficiency", f"{last['Email Efficiency']:.2f}"],
            ["Invites Action Rate", f"{last['Invite Action Rate']:.2f}"],
            ["Travel Impact Score", f"{last['Travel Impact Score']:.1f}"],
            ["Focus Ratio", f"{last['Focus Ratio']:.2f}"],
            ["Overtime Ratio", f"{last['Overtime Ratio']:.2f}"],
            ["Delegation Mix", f"{last['% Delegated']:.0f}% / {last['% Automated']:.0f}% / {last['% Direct']:.0f}%"],
        ]
    pdf_bytes = export_to_pdf(st.session_state.data, totals, align, heatmap_image_bytes, period, quarterly_kpi_table=q_kpi_tbl)
    st.download_button(
        "📄 Click to Save PDF",
        data=pdf_bytes,
        file_name=f"EA_Performance_{period}.pdf",
        mime="application/pdf"
    )

# ---------------------- Transparency: Why it Matters ----------------------
st.divider()
st.markdown("### Why this matters now")
colA, colB, colC = st.columns(3)
with colA:
    st.markdown("**Heatmap** reveals overload and under-investment by category so you can realign support to executive priorities.")
with colB:
    st.markdown("**Probability to hit targets** forecasts delivery under uncertainty so leaders have fewer surprises.")
with colC:
    st.markdown("**Cost and quarterly analytics** quantify impact from reduced redundancy, faster execution, and full-spectrum EA support.")

# ---------------------- Diagnostics ----------------------
st.divider()
st.markdown("### Risk Indicators (Gaps by Category)")
st.dataframe(risk_table(st.session_state.data["Executive Expectations"], st.session_state.data["Actual Performance"]))

