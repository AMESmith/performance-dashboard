import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import io
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

# ---------- Helper Functions ----------
def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LevelScore"] = df["Level"].map(LEVEL_SCORES).astype(float)
    df["WeightedScore"] = df["LevelScore"] * df["Volume"].astype(float)
    if "Priority" in df.columns:
        df["PriorityWeight"] = df["Priority"].map(PRIORITY_WEIGHTS).astype(float)
        df["AdjustedScore"] = df["WeightedScore"] * df["PriorityWeight"]
    else:
        df["PriorityWeight"] = 1.0
        df["AdjustedScore"] = df["WeightedScore"]
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
    ax.set_title(f"{label}: {round(value)}% of {round(total)}%")
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

def histogram(data: np.ndarray, title: str):
    fig, ax = plt.subplots()
    ax.hist(data, bins=20)
    ax.set_title(title)
    st.pyplot(fig)

# ---------- Export Functions ----------
def export_to_excel(all_df, sim_stats=None, period=""):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        all_df.to_excel(writer, index=False, sheet_name="Detailed Data")
        summary_df = all_df.groupby("Scenario")["AdjustedScore"].sum().reset_index()
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        if sim_stats is not None:
            sim_stats.to_excel(writer, index=False, sheet_name="Simulation Stats")
    output.seek(0)
    return output

def export_to_pdf(all_df, totals, align_pct, period):
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
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------- UI ----------
st.set_page_config(page_title="Executive Assistant Performance Dashboard", layout="wide")
st.title("Executive Assistant Performance Dashboard")

with st.sidebar:
    st.header("Settings")
    period = st.text_input("Reporting period", value=date.today().strftime("%Y-%m"))
    simulate = st.checkbox("Run Monte Carlo Simulation")
    iterations = st.number_input("Iterations", min_value=100, max_value=20000, value=2000, step=100, disabled=not simulate)
    vol_variation = st.slider("Volume variation (+/- %)", min_value=0, max_value=100, value=20, step=5, disabled=not simulate)
    level_flip_prob = st.slider("Level change probability (%)", min_value=0, max_value=100, value=10, step=5, disabled=not simulate)

st.markdown(f"**Reporting Period:** {period}")

exp_df = scenario_block("Executive Expectations", "exp")
tgt_df = scenario_block("Targeted Performance", "tgt")
act_df = scenario_block("Actual Performance", "act")

all_df = pd.concat([exp_df, tgt_df, act_df], ignore_index=True)

st.divider()
st.subheader("Scores & Gaps")

totals = all_df.groupby("Scenario")["AdjustedScore"].sum().to_dict()
exp_total = totals.get("Executive Expectations", 0.0)
tgt_total = totals.get("Targeted Performance", 0.0)
act_total = totals.get("Actual Performance", 0.0)

# --- Percentages ---
target_pct = (tgt_total / exp_total * 100) if exp_total > 0 else 0
actual_pct = (act_total / exp_total * 100) if exp_total > 0 else 0
align_pct = actual_pct

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Expectation", "100%")
with colB:
    st.metric("Target",

