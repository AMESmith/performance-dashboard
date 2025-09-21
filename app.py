import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import io
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# ---------- Constants ----------
CATEGORIES = [
    "Executive Support",
    "Project Support",
    "Event Support",
    "Training",
    "Onboarding",
    "Operational Support",
]

LEVEL_SCORES = {"Low": 1, "Mid": 2, "High": 3}
PRIORITY_WEIGHTS = {"Standard": 1.0, "High": 1.5, "Critical": 2.0}
SCENARIOS = ["Executive Expectations", "Targeted Performance", "Actual Performance"]
DATA_FILE = "performance_data.csv"

# ---------- Helper Functions ----------
def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LevelScore"] = df["Level"].map(LEVEL_SCORES).astype(float)
    df["WeightedScore"] = df["LevelScore"] * df["Volume"].astype(float)
    df["PriorityWeight"] = df["Priority"].map(PRIORITY_WEIGHTS).astype(float)
    df["AdjustedScore"] = df["WeightedScore"] * df["PriorityWeight"]
    return df

def scenario_block(title: str, key_prefix: str) -> pd.DataFrame:
    st.subheader(title)
    rows = []
    for i, cat in enumerate(CATEGORIES):
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
                "Executive Support": "# of executives",
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
        rows.append(
            {
                "Scenario": title,
                "Category": cat,
                "Level": level,
                "Volume": volume,
                "Priority": priority,
            }
        )
    frame = pd.DataFrame(rows)
    return compute_scores(frame)

def donut_chart(label: str, value: float, total: float):
    fig, ax = plt.subplots()
    portion = [value, max(total - value, 0)]
    ax.pie(portion, startangle=90, wedgeprops=dict(width=0.4))
    ax.set_title(f"{label}: {round(value)}%")
    st.pyplot(fig)

def bar_chart(df: pd.DataFrame, value_col: str, title: str):
    grouped = df.groupby(["Scenario", "Category"])[value_col].sum().reset_index()
    categories = CATEGORIES
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

def export_to_excel(df, period=""):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
        summary_df = df.groupby(["Period","Scenario"])["AdjustedScore"].sum().reset_index()
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
    output.seek(0)
    return output

def export_to_pdf(df, totals, align_pct, period, cost_estimates):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []
    
    elements.append(Paragraph("Executive Assistant Performance Dashboard", styles['Title']))
    elements.append(Paragraph(f"Reporting Period: {period}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    summary = [["Scenario", "Score"]] + [[k, f"{v:.1f}"] for k,v in totals.items()]
    summary.append(["Alignment %", f"{round(align_pct)}%"])
    elements.append(Table(summary))
    
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Estimated Costs:", styles['Heading2']))
    for scen, cost in cost_estimates.items():
        elements.append(Paragraph(f"{scen}: ${cost:,.0f}", styles['Normal']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------- UI ----------
st.set_page_config(page_title="Executive Assistant Performance Dashboard", layout="wide")
st.title("Executive Assistant Performance Dashboard")

with st.sidebar:
    st.header("Settings")

    # Period type: monthly or quarterly
    period_type = st.radio("Select Period Type", ["Monthly", "Quarterly"], horizontal=True)

    if period_type == "Monthly":
        month = st.selectbox("Select Month", [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        year = st.number_input("Year", min_value=2000, max_value=2100, value=date.today().year)
        period = f"{month} {year}"
    else:
        quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])
        year = st.number_input("Year", min_value=2000, max_value=2100, value=date.today().year)
        period = f"{quarter} {year}"

    hourly_rate = st.number_input("Hourly Rate ($)", min_value=0, value=100, step=10)

st.markdown(f"**Reporting Period:** {period}")

# --- Scenarios Input ---
exp_df = scenario_block("Executive Expectations", "exp")
tgt_df = scenario_block("Targeted Performance", "tgt")
act_df = scenario_block("Actual Performance", "act")

all_df = pd.concat([exp_df, tgt_df, act_df], ignore_index=True)
all_df["Period"] = period
all_df["Year"] = year

# Save button
if st.button("💾 Save Data"):
    if os.path.exists(DATA_FILE):
        existing = pd.read_csv(DATA_FILE)
        combined = pd.concat([existing, all_df], ignore_index=True)
    else:
        combined = all_df
    combined.to_csv(DATA_FILE, index=False)
    st.success(f"Data for {period} saved.")

# Load existing data
if os.path.exists(DATA_FILE):
    data_store = pd.read_csv(DATA_FILE)
else:
    data_store = pd.DataFrame()

# Choose view
st.sidebar.markdown("### View Options")
view_mode = st.sidebar.radio("Select View", ["Current Period", "Annual Rollup"])

if view_mode == "Current Period":
    df_view = all_df
    view_label = period
else:
    df_view = data_store[data_store["Year"] == year]
    view_label = f"Annual {year}"

# --- KPI Calculations ---
totals = df_view.groupby("Scenario")["AdjustedScore"].sum().to_dict()
exp_total = totals.get("Executive Expectations", 0.0)
tgt_total = totals.get("Targeted Performance", 0.0)
act_total = totals.get("Actual Performance", 0.0)

target_pct = (tgt_total / exp_total * 100) if exp_total > 0 else 0
actual_pct = (act_total / exp_total * 100) if exp_total > 0 else 0
align_pct = actual_pct

st.subheader(f"KPI Overview — {view_label}")
colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Expectation", "100%", f"{round(exp_total)} pts")
with colB:
    st.metric("Target", f"{round(target_pct)}%", f"{round(tgt_total)} pts")
with colC:
    st.metric("Actual", f"{round(actual_pct)}%", f"{round(act_total)} pts")
with colD:
    st.metric("Alignment", f"{round(align_pct)}%", f"{round(act_total)} pts delivered")

# Cost Estimates
cost_estimates = {
    "Expectation": exp_total * hourly_rate,
    "Target": tgt_total * hourly_rate,
    "Actual": act_total * hourly_rate,
}
st.markdown("### Estimated Costs")
for scen, cost in cost_estimates.items():
    st.markdown(f"- **{scen}:** ${cost:,.0f}")

# Charts
col1, col2 = st.columns(2)
with col1:
    donut_chart("Actual vs Expectation", actual_pct, 100)
with col2:
    donut_chart("Target vs Expectation", target_pct, 100)

st.subheader("Category Breakdown")
bar_chart(df_view, "AdjustedScore", f"Adjusted Score by Category & Scenario — {view_label}")

st.subheader("Detailed Table")
st.dataframe(df_view[
    ["Period","Scenario","Category","Level","Volume","Priority","LevelScore","WeightedScore","PriorityWeight","AdjustedScore"]
])

# Export buttons
excel_file = export_to_excel(df_view, period)
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
