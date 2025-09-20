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
    ax.set_title(f"{label}: {value:.1f} / {total:.1f}")
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
    summary.append(["Alignment %", f"{align_pct:.1f}%"])
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

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Expectation Total", f"{exp_total:.1f}")
with colB:
    st.metric("Target Total", f"{tgt_total:.1f}")
with colC:
    st.metric("Actual Total", f"{act_total:.1f}")
with colD:
    align_pct = (act_total / exp_total * 100) if exp_total > 0 else 0.0
    st.metric("Alignment % (Actual vs Expectation)", f"{align_pct:.1f}%")

col1, col2 = st.columns(2)
with col1:
    donut_chart("Actual vs Expectation", act_total, max(exp_total, act_total))
with col2:
    donut_chart("Target vs Expectation", tgt_total, max(exp_total, tgt_total))

st.subheader("Category Breakdown")
bar_chart(all_df, "AdjustedScore", "Adjusted Score by Category & Scenario")

st.subheader("Detailed Table")
st.dataframe(all_df[
    ["Scenario","Category","Level","Volume","Priority","LevelScore","WeightedScore","PriorityWeight","AdjustedScore"]
])

# ---------- Export Buttons ----------
excel_file = export_to_excel(all_df, None, period)
st.download_button(
    label="📊 Download Excel Report",
    data=excel_file,
    file_name=f"EA_Performance_{period}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

pdf_file = export_to_pdf(all_df, totals, align_pct, period)
st.download_button(
    label="📄 Download PDF Report",
    data=pdf_file,
    file_name=f"EA_Performance_{period}.pdf",
    mime="application/pdf"
)

# ---------- Simulation ----------
if simulate:
    st.divider()
    st.subheader("Monte Carlo Simulation (Uncertainty Analysis)")

    base_actual = act_df.copy()
    def level_to_idx(level: str) -> int:
        return ["Low", "Mid", "High"].index(level)
    def idx_to_level(idx: int) -> str:
        idx = int(np.clip(idx, 0, 2))
        return ["Low", "Mid", "High"][idx]

    exp_total_arr = np.full(int(iterations), exp_total, dtype=float)
    act_total_arr = np.zeros(int(iterations), dtype=float)
    align_arr = np.zeros(int(iterations), dtype=float)

    for i in range(int(iterations)):
        sim_rows = []
        for _, row in base_actual.iterrows():
            vol_noise = np.random.normal(loc=0.0, scale=vol_variation/100.0)
            sim_vol = max(0, int(round(row["Volume"] * (1 + vol_noise))))
            lvl_idx = level_to_idx(row["Level"])
            if np.random.rand() < (level_flip_prob/100.0):
                lvl_idx += np.random.choice([-1, 1])
            sim_level = idx_to_level(lvl_idx)
            sim_rows.append({
                "Scenario": "Actual Performance (Sim)",
                "Category": row["Category"],
                "Level": sim_level,
                "Volume": sim_vol,
                "Priority": row["Priority"],
            })
        sim_df = pd.DataFrame(sim_rows)
        sim_df = compute_scores(sim_df)
        act_total_arr[i] = sim_df["AdjustedScore"].sum()
        align_arr[i] = (act_total_arr[i] / exp_total_arr[i] * 100) if exp_total_arr[i] > 0 else 0.0

    histogram(act_total_arr, "Distribution of Actual Totals")
    histogram(align_arr, "Distribution of Alignment % (Actual vs Expectation)")

    sim_stats = pd.DataFrame({
        "Metric": ["Actual Total", "Alignment %"],
        "Mean": [np.mean(act_total_arr), np.mean(align_arr)],
        "Std Dev": [np.std(act_total_arr), np.std(align_arr)],
        "P10": [np.percentile(act_total_arr, 10), np.percentile(align_arr, 10)],
        "P50": [np.percentile(act_total_arr, 50), np.percentile(align_arr, 50)],
        "P90": [np.percentile(act_total_arr, 90), np.percentile(align_arr, 90)],
    })
    st.dataframe(sim_stats)

    # Export simulation-inclusive Excel
    excel_file = export_to_excel(all_df, sim_stats, period)
    st.download_button(
        label="📊 Download Excel Report (with Simulation)",
        data=excel_file,
        file_name=f"EA_Performance_{period}_sim.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
