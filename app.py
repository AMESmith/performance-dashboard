import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import io
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# ------------------ Constants ------------------
EXEC_SECTION = "Executive Support"

WORK_ANALYSIS_CATEGORIES = [
    "Project Support",
    "Event Support",
    "Training",
    "Onboarding",
    "Operational Support",
]

ALL_CATEGORIES = [EXEC_SECTION] + WORK_ANALYSIS_CATEGORIES

LEVEL_SCORES = {"Low": 1, "Mid": 2, "High": 3}
PRIORITY_WEIGHTS = {"Standard": 1.0, "High": 1.5, "Critical": 2.0}
SCENARIOS = ["Executive Expectations", "Targeted Performance", "Actual Performance"]
DATA_FILE = "performance_data.csv"

# ------------------ Helpers ------------------
def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LevelScore"] = df["Level"].map(LEVEL_SCORES).astype(float)
    df["WeightedScore"] = df["LevelScore"] * df["Volume"].astype(float)
    df["PriorityWeight"] = df["Priority"].map(PRIORITY_WEIGHTS).astype(float)
    df["AdjustedScore"] = df["WeightedScore"] * df["PriorityWeight"]
    return df

def executive_support_row(key_prefix: str) -> dict:
    c1, c2, c3, c4 = st.columns([2,1,1,1])
    with c1:
        st.text(EXEC_SECTION)
    with c2:
        level = st.selectbox(
            "Level",
            options=list(LEVEL_SCORES.keys()),
            index=1,
            key=f"{key_prefix}_exec_level",
            label_visibility="collapsed",
        )
    with c3:
        volume = st.number_input(
            "# of executives",
            min_value=0,
            step=1,
            value=0,
            key=f"{key_prefix}_exec_vol",
        )
    with c4:
        priority = st.selectbox(
            "Priority",
            options=list(PRIORITY_WEIGHTS.keys()),
            index=0,
            key=f"{key_prefix}_exec_prio",
            label_visibility="collapsed",
        )
    return {"Category": EXEC_SECTION, "Level": level, "Volume": volume, "Priority": priority}

def work_analysis_rows(key_prefix: str) -> list:
    rows = []
    for i, cat in enumerate(WORK_ANALYSIS_CATEGORIES):
        c1, c2, c3, c4 = st.columns([2,1,1,1])
        with c1:
            st.text(cat)
        with c2:
            level = st.selectbox(
                "Level",
                options=list(LEVEL_SCORES.keys()),
                index=1,
                key=f"{key_prefix}_level_{i}",
                label_visibility="collapsed",
            )
        with c3:
            vol_label = {
                "Project Support": "# of projects",
                "Event Support": "# of events",
                "Training": "# of trainings",
                "Onboarding": "# of onboardings",
                "Operational Support": "# of operational needs",
            }.get(cat, "Volume")
            volume = st.number_input(
                vol_label, min_value=0, step=1, value=0, key=f"{key_prefix}_vol_{i}"
            )
        with c4:
            priority = st.selectbox(
                "Priority",
                options=list(PRIORITY_WEIGHTS.keys()),
                index=0,
                key=f"{key_prefix}_prio_{i}",
                label_visibility="collapsed",
            )
        rows.append({"Category": cat, "Level": level, "Volume": volume, "Priority": priority})
    return rows

def scenario_section(title: str, key_prefix: str) -> pd.DataFrame:
    st.markdown(f"### {title}")
    # Header row
    hc1, hc2, hc3, hc4 = st.columns([2,1,1,1])
    with hc1: st.markdown("**Category**")
    with hc2: st.markdown("**Level**")
    with hc3: st.markdown("**Volume**")
    with hc4: st.markdown("**Priority**")

    # Executive Support first
    exec_row = executive_support_row(f"{key_prefix}_exec")

    # Then Work Analysis categories
    wa_rows = work_analysis_rows(f"{key_prefix}_wa")

    rows = [exec_row] + wa_rows
    df = pd.DataFrame(rows)
    df.insert(0, "Scenario", title)
    return compute_scores(df)

def donut_chart(label: str, value_pct: float):
    fig, ax = plt.subplots()
    portion = [max(value_pct, 0), max(100 - value_pct, 0)]
    ax.pie(portion, startangle=90, wedgeprops=dict(width=0.4))
    ax.set_title(f"{label}: {round(value_pct)}%")
    st.pyplot(fig)

def bar_chart(df: pd.DataFrame, value_col: str, title: str):
    grouped = df.groupby(["Scenario", "Category"])[value_col].sum().reset_index()
    categories = ALL_CATEGORIES
    scenarios = SCENARIOS
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    x = np.arange(len(categories))
    for si, scen in enumerate(scenarios):
        vals = [grouped[(grouped["Scenario"] == scen) & (grouped["Category"] == c)][value_col].sum() for c in categories]
        ax.bar(x + (si - 1) * width, vals, width=width, label=scen)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

def export_to_excel(df, view_label=""):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Detailed Data")
        summary_df = df.groupby("Scenario")["AdjustedScore"].sum().reset_index()
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
    output.seek(0)
    return output

def export_to_pdf(df, totals, align_pct, view_label, cost_estimates):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Executive Assistant Performance Dashboard", styles['Title']))
    elements.append(Paragraph(f"Reporting View: {view_label}", styles['Normal']))
    elements.append(Spacer(1, 12))
    summary = [["Scenario", "Score"]] + [[k, f"{v:.1f}"] for k,v in totals.items()]
    summary.append(["Alignment %", f"{round(align_pct)}%"])
    elements.append(Table(summary))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Estimated Costs", styles['Heading2']))
    for scen, cost in cost_estimates.items():
        elements.append(Paragraph(f"{scen}: ${cost:,.0f}", styles['Normal']))
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ------------------ Styled KPI Card ------------------
def kpi_card(title, value, subtitle):
    st.markdown(
        f"""
        <div style="
            background-color:#000000;
            border-radius:12px;
            padding:20px;
            text-align:center;
            color:#ffffff;
        ">
            <h4 style="margin:0;">{title}</h4>
            <h2 style="margin:5px 0; color:#ffb000;">{value}</h2>
            <p style="margin:0; font-size:14px; color:#ffffff;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------ UI ------------------
st.set_page_config(page_title="Executive Assistant Performance Dashboard", layout="wide")
st.title("Executive Assistant Performance Dashboard")

# Sidebar
with st.sidebar:
    st.header("Settings")
    period_type = st.radio("Select Period Type", ["Monthly", "Quarterly"], horizontal=True)

    if period_type == "Monthly":
        month = st.selectbox("Select Month", [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        year = st.number_input("Year", min_value=2000, max_value=2100, value=date.today().year)
        period = f"{month} {year}"
        quarter = None
    else:
        quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])
        year = st.number_input("Year", min_value=2000, max_value=2100, value=date.today().year)
        period = f"{quarter} {year}"

    annual_rollup = st.checkbox("Annual Rollup (aggregate all four quarters)")
    hourly_rate = st.number_input("Hourly Rate ($)", min_value=0, value=100, step=10)

st.markdown(f"**Reporting Period:** {period}")

# Scenarios
exp_df = scenario_section("Executive Expectations", "exp")
tgt_df = scenario_section("Targeted Performance", "tgt")
act_df = scenario_section("Actual Performance", "act")

# Attach period metadata
for df in (exp_df, tgt_df, act_df):
    df["Year"] = year
    if period_type == "Quarterly":
        df["Quarter"] = quarter
    else:
        month_to_q = {
            "January": "Q1", "February": "Q1", "March": "Q1",
            "April": "Q2", "May": "Q2", "June": "Q2",
            "July": "Q3", "August": "Q3", "September": "Q3",
            "October": "Q4", "November": "Q4", "December": "Q4",
        }
        month_name = period.split()[0]
        df["Quarter"] = month_to_q.get(month_name, "Q1")
    df["Period"] = period

all_df = pd.concat([exp_df, tgt_df, act_df], ignore_index=True)

# Save
if st.button("💾 Save Quarter Data"):
    if os.path.exists(DATA_FILE):
        existing = pd.read_csv(DATA_FILE)
        combined = pd.concat([existing, all_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Year","Quarter","Scenario","Category"], keep="last").reset_index(drop=True)
    else:
        combined = all_df
    combined.to_csv(DATA_FILE, index=False)
    st.success(f"Data saved for {period}.")

if os.path.exists(DATA_FILE):
    data_store = pd.read_csv(DATA_FILE)
else:
    data_store = pd.DataFrame(columns=all_df.columns)

# View selection
if annual_rollup and not data_store.empty:
    df_view = data_store[data_store["Year"] == year]
    view_label = f"Annual {year}"
else:
    df_view = all_df
    view_label = period

# ------------------ KPIs ------------------
totals = df_view.groupby("Scenario")["AdjustedScore"].sum().to_dict()
exp_total = totals.get("Executive Expectations", 0.0)
tgt_total = totals.get("Targeted Performance", 0.0)
act_total = totals.get("Actual Performance", 0.0)

target_pct = (tgt_total / exp_total * 100) if exp_total > 0 else 0
actual_pct = (act_total / exp_total * 100) if exp_total > 0 else 0
align_pct = actual_pct

st.subheader(f"Executive KPI Overview — {view_label}")
colA, colB, colC, colD = st.columns(4)
with colA:
    kpi_card("Expectation", "100%", f"{round(exp_total)} pts")
with colB:
    kpi_card("Target", f"{round(target_pct)}%", f"{round(tgt_total)} pts")
with colC:
    kpi_card("Actual", f"{round(actual_pct)}%", f"{round(act_total)} pts")
with colD:
    kpi_card("Alignment", f"{round(align_pct)}%", f"{round(act_total)} pts delivered")

# ------------------ Costs ------------------
cost_estimates = {
    "Expectation": exp_total * hourly_rate,
    "Target": tgt_total * hourly_rate,
    "Actual": act_total * hourly_rate,
}
st.markdown("### Estimated Costs")
for scen, cost in cost_estimates.items():
    st.markdown(f"- **{scen}:** ${cost:,.0f}")

# ------------------ Charts ------------------
col1, col2 = st.columns(2)
with col1:
    donut_chart("Actual vs Expectation", actual_pct)
with col2:
    donut_chart("Target vs Expectation", target_pct)

st.subheader(f"Category Breakdown — {view_label}")
bar_chart(df_view, "AdjustedScore", "Adjusted Score by Category & Scenario")

# ------------------ Detail ------------------
st.subheader("Detailed Table")
st.dataframe(df_view[
    ["Period","Year","Quarter","Scenario","Category","Level","Volume","Priority","LevelScore","WeightedScore","PriorityWeight","AdjustedScore"]
])

# ------------------ Exports ------------------
excel_file = export_to_excel(df_view, view_label)
st.download_button(
    label="📊 Download Excel Report",
    data=excel_file,
    file_name=f"EA_Performance_{view_label}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

pdf_file = export_to_pdf(df_view, totals, align_pct, view_label, cost_estimates)
st.download_button(
    label="📄 Download PDF Report",
    data=pdf_file,
    file_name=f"EA_Performance_{view_label}.pdf",
    mime="application/pdf"
)

