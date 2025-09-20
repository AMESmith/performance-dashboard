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
CATEGORIES = ["Executive Support", "Operational Support"]  # trimmed per request
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
        ss.scenarios = {
            "Executive Expectations": _empty_perf_df(),
            "Targeted Performance": _empty_perf_df(),
            "Actual Performance": _empty_perf_df(),
        }
    # Support Alignment: Executives Supported (per scenario)
    if "execs_supported" not in ss:
        ss.execs_supported = {
            "Executive Expectations": 1,
            "Targeted Performance": 1,
            "Actual Performance": 1,
        }
    # Quarterly analytics (grouped categories; OPERATIONAL SUPPORT now includes Projects + Events)
    if "quarterly" not in ss:
        ss.quarterly = pd.DataFrame([{
            "Quarter": _default_quarter_label(),
            # Email
            "Emails Received": 0, "Emails Sent": 0, "Invites Actioned": 0,
            # Calendar Engineering
            "Meetings Scheduled": 0, "Reschedules": 0, "Meeting Notes Prepared": 0,
            # Travel
            "Domestic Trips": 0, "International Trips": 0,
            # People Ops
            "Onboardings Supported": 0, "Trainings Facilitated": 0,
            # Finance
            "Expense Reports Processed": 0, "Approvals Routed": 0,
            # Operational Support (moved here)
            "Projects": 0, "Events": 0,
            # Automation / Delegation
            "Tasks Delegated": 0, "Tasks Automated": 0, "Tasks Directly Executed": 0,
            # Work pattern (minimal)
            "Reactive Work Hours": 0.0, "Overtime Hours": 0.0,
        }])
    # Sidebar visual toggles (persist when buttons clicked)
    if "show_heatmap" not in ss: ss.show_heatmap = False
    if "show_charts" not in ss: ss.show_charts = False
    # Last heatmap image for PDF
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
        # Nudge: include executives supported as an alignment multiplier on Executive Support row only
        # (keeps it simple while reflecting increased alignment load)
        ex_df = df[df["Category"] == "Executive Support"].copy()
        exec_load = 0.0
        if not ex_df.empty:
            ex_score = compute_scores(ex_df)["WeightedScore"].sum()
            # replace that component with (ex_score * execs), then add rest (base - ex_score)
            total = (base - ex_score) + (ex_score * max(1, execs_supported.get(scen, 1)))
        else:
            total = base
        totals[scen] = total
    return totals

def alignment_pct(actual_total, expectation_total):
    if expectation_total <= 0:
        return 0.0
    return float(actual_total / expectation_total * 100.0)

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
    # Treat WeightedScore as "hours". Positive gap above target = redundancy; shortfall = speed opportunity.
    tgt_hours = compute_scores(target_df)["WeightedScore"].sum()
    act_hours = compute_scores(actual_df)["WeightedScore"].sum()
    redundancy_hours = max(0.0, act_hours - tgt_hours)
    speed_hours = max(0.0, tgt_hours - act_hours) * 0.5  # heuristic 50% of shortfall can be reclaimed via enablement
    return float(redundancy_hours * hourly_rate), float(speed_hours * hourly_rate)

def exec_satisfaction_projection(expect_df, actual_df):
    exp = compute_scores(expect_df)
    act = compute_scores(actual_df)
    vals = []
    for cat in CATEGORIES:
        e = exp.loc[exp["Category"]==cat, "WeightedScore"].sum()
        a = act.loc[act["Category"]==cat, "WeightedScore"].sum()
        if e > 0:
            vals.append(min(100.0, (a/e)*100.0))
    return float(np.mean(vals)) if vals else 0.0

# Quarterly enrichment for charts/KPIs
def quarterly_enriched(df: pd.DataFrame) -> pd.DataFrame:
    q = df.copy()
    q["Email Efficiency"] = np.where(q["Emails Received"]>0, q["Emails Sent"]/q["Emails Received"], 0.0)
    q["Invite Action Rate"] = np.where(q["Emails Received"]>0, q["Invites Actioned"]/q["Emails Received"], 0.0)
    q["Travel Impact Score"] = q["Domestic Trips"]*1.0 + q["International Trips"]*3.0
    q["Delegation Total"] = q["Tasks Delegated"] + q["Tasks Automated"] + q["Tasks Directly Executed"]
    q["% Delegated"] = np.where(q["Delegation Total"]>0, (q["Tasks Delegated"]/q["Delegation Total"])*100.0, 0.0)
    q["% Automated"] = np.where(q["Delegation Total"]>0, (q["Tasks Automated"]/q["Delegation Total"])*100.0, 0.0)
    q["% Direct"] = np.where(q["Delegation Total"]>0, (q["Tasks Directly Executed"]/q["Delegation Total"])*100.0, 0.0)
    return q

# ---------------------- Exports ----------------------
def export_to_excel(scenarios_dict, execs_supported, quarterly_df, period):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Scenarios per sheet
        for scen, df in scenarios_dict.items():
            compute_scores(df).to_excel(writer, index=False, sheet_name=scen[:31])
        # Support Alignment
        pd.DataFrame([execs_supported]).T.reset_index().rename(columns={"index":"Scenario", 0:"Executives Supported"})\
            .to_excel(writer, index=False, sheet_name="Support Alignment")
        # Summary with execs multiplier reflected
        totals = pd.DataFrame([scenario_totals(scenarios_dict, execs_supported)]).T.reset_index()
        totals.columns = ["Scenario", "Total Weighted Score"]
        totals.to_excel(writer, index=False, sheet_name="Summary")
        # Quarterly
        quarterly_enriched(quarterly_df).to_excel(writer, index=False, sheet_name="Quarterly")
    output.seek(0)
    return output

def export_to_pdf(scenarios_dict, execs_supported, period, heatmap_img_bytes=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    totals = scenario_totals(scenarios_dict, execs_supported)
    exp_total = totals.get("Executive Expectations", 0.0)
    act_total = totals.get("Actual Performance", 0.0)
    align = alignment_pct(act_total, exp_total)

    elements.append(Paragraph(TITLE, styles["Title"]))
    elements.append(Paragraph(f"Reporting Period: {period}", styles["Normal"]))
    elements.append(Spacer(1, 8))

    # Support Alignment table
    sa_tbl = [["Scenario", "Executives Supported"]]
    for k in SCENARIOS:
        sa_tbl.append([k, str(st.session_state.execs_supported.get(k, 1))])
    t_sa = Table(sa_tbl, hAlign="LEFT")
    t_sa.setStyle(TableStyle([("BOX",(0,0),(-1,-1),0.5,colors.black),
                              ("GRID",(0,0),(-1,-1),0.25,colors.grey),
                              ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    elements.append(Paragraph("Support Alignment", styles["Heading3"]))
    elements.append(t_sa)
    elements.append(Spacer(1, 8))

    # Totals
    tbl = [["Scenario", "Total Weighted Score"]]
    for k, v in totals.items():
        tbl.append([k, f"{v:.1f}"])
    tbl.append(["Alignment % (Actual vs Expectation)", f"{align:.1f}%"])
    t = Table(tbl, hAlign="LEFT")
    t.setStyle(TableStyle([("BOX",(0,0),(-1,-1),0.5,colors.black),
                           ("GRID",(0,0),(-1,-1),0.25,colors.grey),
                           ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    elements.append(Paragraph("Scenario Summary", styles["Heading3"]))
    elements.append(t)
    elements.append(Spacer(1, 10))

    if heatmap_img_bytes is not None:
        from reportlab.platypus import Image
        img = Image(heatmap_img_bytes)
        img._restrictSize(500, 330)
        elements.append(Paragraph("Performance Heatmap", styles["Heading3"]))
        elements.append(img)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------------- Sidebar Controls ----------------------
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

    st.markdown("---")
    # Prepare downloads every run so buttons always work
    excel_bytes = export_to_excel(st.session_state.scenarios, st.session_state.execs_supported, st.session_state.quarterly, period)
    st.download_button("📊 Download Excel", data=excel_bytes,
                       file_name=f"EA_Performance_{period}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    pdf_bytes = export_to_pdf(st.session_state.scenarios, st.session_state.execs_supported, period, st.session_state.heatmap_img)
    st.download_button("📄 Download PDF", data=pdf_bytes,
                       file_name=f"EA_Performance_{period}.pdf",
                       mime="application/pdf")

# ---------------------- Header + KPIs ----------------------
st.markdown(f"## {TITLE}")

totals = scenario_totals(st.session_state.scenarios, st.session_state.execs_supported)
exp_total = totals.get("Executive Expectations", 0.0)
tgt_total = totals.get("Targeted Performance", 0.0)
act_total = totals.get("Actual Performance", 0.0)
align_val = alignment_pct(act_total, exp_total)
sat_proj = exec_satisfaction_projection(
    st.session_state.scenarios["Executive Expectations"],
    st.session_state.scenarios["Actual Performance"]
)
sav_redund, sav_speed = expected_cost_savings(
    st.session_state.scenarios["Targeted Performance"],
    st.session_state.scenarios["Actual Performance"],
    hourly_rate
)

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("Expectation Total", f"{exp_total:.1f}")
with k2: st.metric("Target Total", f"{tgt_total:.1f}")
with k3: st.metric("Actual Total", f"{act_total:.1f}")
with k4: st.metric("Alignment %", f"{align_val:.1f}%")
with k5: st.metric("Exec Satisfaction Projection", f"{sat_proj:.1f}%")

k6, k7, k8 = st.columns(3)
with k6: st.metric("Executives Supported (Actual)", st.session_state.execs_supported["Actual Performance"])
with k7: st.metric("Expected Savings (Redundancy)", f"${sav_redund:,.0f}")
with k8: st.metric("Expected Savings (Speed)", f"${sav_speed:,.0f}")

st.divider()

# ---------------------- Support Alignment ----------------------
st.markdown("### Support Alignment")
cols = st.columns(3)
for i, scen in enumerate(SCENARIOS):
    with cols[i]:
        st.session_state.execs_supported[scen] = st.number_input(
            f"{scen} – Executives Supported",
            min_value=1, step=1, value=int(st.session_state.execs_supported.get(scen, 1))
        )

st.divider()

# ---------------------- Performance Volume (Scenario-Specific) ----------------------
st.markdown("### Performance Volume")
st.caption("Edit levels and volumes for the *selected* scenario. (Training/Onboarding removed; Projects/Events moved to Quarterly → Operational Support.)")

perf_df = st.session_state.scenarios[selected_scenario].copy()
edited = st.data_editor(
    perf_df,
    use_container_width=True,
    disabled=["Category"],  # fixed 2 rows (Executive Support, Operational Support)
    column_config={
        "Category": st.column_config.TextColumn(),
        "Level": st.column_config.SelectboxColumn(options=LEVELS, required=True),
        "Volume": st.column_config.NumberColumn(min_value=0, step=1),
    },
    key=f"perf_editor_{selected_scenario}",
)
st.session_state.scenarios[selected_scenario] = edited

st.divider()

# ---------------------- Quarterly Workload Analytics (Grouped) ----------------------
st.markdown("### Quarterly Workload Analytics")
q = st.session_state.quarterly
quarter_options = list(q["Quarter"].astype(str).unique())
current_q = st.selectbox("Quarter to edit", options=quarter_options + ["+ Add new quarter"], index=len(quarter_options)-1 if quarter_options else 0)

if current_q == "+ Add new quarter":
    new_label = st.text_input("New quarter label (YYYY-Q#)", value=_default_quarter_label())
    if st.button("Add quarter"):
        if new_label and new_label not in q["Quarter"].values:
            base = q.iloc[-1].copy()
            base["Quarter"] = new_label
            st.session_state.quarterly = pd.concat([q, pd.DataFrame([base])], ignore_index=True)
        q = st.session_state.quarterly
        quarter_options = list(q["Quarter"].astype(str).unique())
        current_q = quarter_options[-1]

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

with st.expander("Email", expanded=True):
    edit_block("Email", ["Emails Received", "Emails Sent", "Invites Actioned"])

with st.expander("Meetings (Calendar Engineering)", expanded=True):
    edit_block("Calendar Engineering", ["Meetings Scheduled", "Reschedules", "Meeting Notes Prepared"])

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

# Current-quarter KPIs
if not q_enriched.empty and current_q in q_enriched["Quarter"].values:
    row = q_enriched[q_enriched["Quarter"]==current_q].iloc[0]
    qa, qb, qc, qd = st.columns(4)
    with qa: st.metric("Email Efficiency", f"{row['Email Efficiency']:.2f}")
    with qb: st.metric("Invite Action Rate", f"{row['Invite Action Rate']:.2f}")
    with qc: st.metric("Travel Impact Score", f"{row['Travel Impact Score']:.1f}")
    with qd: st.metric("Deleg/AUTO/DIRECT", f"{row['% Delegated']:.0f}% / {row['% Automated']:.0f}% / {row['% Direct']:.0f}%")

st.divider()

# ---------------------- Visuals (Shown when toggled) ----------------------
if st.session_state.show_heatmap or st.session_state.show_charts:
    cats, scens, mat = make_heatmap_matrix(st.session_state.scenarios)

if st.session_state.show_heatmap:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(mat, aspect="auto", cmap="Greys")  # greyscale heatmap
    ax.set_xticks(np.arange(len(scens))); ax.set_xticklabels(scens, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(cats))); ax.set_yticklabels(cats)
    ax.set_title("Performance Heatmap (Weighted Scores, Greyscale)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center")
    st.pyplot(fig)

    # Save greyscale heatmap for PDF
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.session_state.heatmap_img = buf

if st.session_state.show_charts:
    # Donuts
    def donut(value, total, title):
        fig, ax = plt.subplots()
        parts = [value, max(total - value, 0)]
        ax.pie(parts, startangle=90, wedgeprops=dict(width=0.4))
        ax.set_title(title)
        st.pyplot(fig)

    cA, cB = st.columns(2)
    with cA: donut(act_total, max(exp_total, act_total), "Actual vs Expectation")
    with cB: donut(tgt_total, max(exp_total, tgt_total), "Target vs Expectation")

    # Category bar chart
    rows = []
    for scen in SCENARIOS:
        df = compute_scores(st.session_state.scenarios[scen])
        by_cat = df.groupby("Category")["WeightedScore"].sum()
        for cat in CATEGORIES:
            rows.append({"Scenario": scen, "Category": cat, "WeightedScore": float(by_cat.get(cat, 0.0))})
    g = pd.DataFrame(rows)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    width = 0.35
    x = np.arange(len(CATEGORIES))
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
    risk_table(st.session_state.scenarios["Executive Expectations"],
               st.session_state.scenarios["Actual Performance"]),
    use_container_width=True
)
