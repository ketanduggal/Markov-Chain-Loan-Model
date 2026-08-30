"""
Single-Family Loan Markov Chain (SLMC) Model
Streamlit Application

Models loan delinquency state transitions using Markov Chains with:
  - MLE-based Transition Probability Matrix estimation
  - Chapman-Kolmogorov n-step forecast calculator
  - Absorbing-state (90+ Days Delinquent) default probability calculator
  - Portfolio summary statistics

Data source: Fannie Mae Single-Family Loan Performance Data
  https://datadynamics.fanniemae.com/data-dynamics/#/downloadLoanData/Single-Family

Author: Ketan Duggal - Senior Actuarial Analyst & Quantitative Modeler | Visual Artist
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SLMC Model – Single-Family Loan Markov Chain",
    page_icon="🏠",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS Injection - Luxury Editorial Gallery Aesthetic
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main app background - Warm Gallery Linen/Ivory */
    .stApp {
        background-color: #F9F8F6;
    }

    /* Overall text color for body */
    body {
        color: #2A2A2A !important;
    }

    /* Main content area text */
    .main .block-container {
        color: #2A2A2A !important;
    }

    /* All markdown content */
    .stMarkdown {
        color: #2A2A2A !important;
    }

    /* All text elements */
    p, span, div, label {
        color: #2A2A2A !important;
    }

    /* Headers with specific targeting */
    h1, h2, h3, h4, h5, h6 {
        color: #1A1A1A !important;
    }

    /* Links */
    a {
        color: #C5A059 !important;
    }

    a:hover {
        color: #B8964A !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F3EFEA;
        border-right: 1px solid #E5E0D8;
    }

    /* Sidebar text - improved contrast */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #1A1A1A !important;
    }

    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label {
        color: #2A2A2A !important;
    }

    /* Container styling - White floating cards with soft shadows */
    [data-testid="stContainer"] {
        background-color: #FFFFFF;
        border: 1px solid #EBE6DF;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Container text styling */
    [data-testid="stContainer"] .stMarkdown,
    [data-testid="stContainer"] .stText,
    [data-testid="stContainer"] p {
        color: #2A2A2A !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #EBE6DF;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        padding: 1rem;
    }

    /* Typography - Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1A1A1A !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 600;
    }

    /* Typography - Subheaders and captions */
    .stCaption, .stSubheader {
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #5A5245 !important;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* General text styling for better readability */
    .stMarkdown, .stText {
        color: #2A2A2A !important;
    }

    /* All text inside markdown */
    .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #2A2A2A !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] .stText {
        color: #2A2A2A !important;
    }

    /* Specific targeting for expander headers */
    [data-testid="stExpander"] > div > div {
        color: #1A1A1A !important;
    }

    /* Expander header text specifically */
    [data-testid="stExpander"] .streamlit-expanderHeader {
        color: #1A1A1A !important;
    }

    /* Target all text content specifically */
    .stText, .stMarkdown, .stCaption {
        color: #2A2A2A !important;
    }

    /* All Streamlit text elements */
    [data-testid="stMarkdownContainer"],
    [data-testid="stText"] {
        color: #2A2A2A !important;
    }

    /* Fix inline styles that might override */
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stText"] * {
        color: #2A2A2A !important;
    }

    /* Button styling - Muted Gold */
    .stButton > button {
        background-color: #C5A059;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #B8964A;
        box-shadow: 0 2px 8px rgba(197, 160, 89, 0.3);
    }

    /* Tab styling */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 8px;
    }

    [data-testid="stTabs"] [role="tab"] {
        background-color: #FFFFFF;
        border: 1px solid #EBE6DF;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #5A5245 !important;
        font-weight: 500;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background-color: #C5A059;
        color: white !important;
        border-color: #C5A059;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #FFFFFF;
        border: 1px solid #EBE6DF;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        color: #1A1A1A !important;
        font-weight: 500;
    }

    /* Expander content */
    .streamlit-expanderContent {
        color: #2A2A2A !important;
    }

    /* Expander arrow icon */
    .streamlit-expanderHeader svg {
        color: #1A1A1A !important;
    }

    /* All expander text elements */
    .streamlit-expanderHeader span,
    .streamlit-expanderContent p,
    .streamlit-expanderContent li,
    .streamlit-expanderContent span {
        color: #2A2A2A !important;
    }

    /* Input styling */
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: #FFFFFF;
        border: 1px solid #EBE6DF;
        border-radius: 8px;
    }

    /* Input labels and text */
    .stSelectbox label, 
    .stNumberInput label, 
    .stSlider label {
        color: #2A2A2A !important;
    }

    /* Metric text */
    [data-testid="stMetricValue"] {
        color: #1A1A1A !important;
    }

    [data-testid="stMetricLabel"] {
        color: #5A5245 !important;
    }

    /* DataFrame styling */
    .dataframe {
        border: 1px solid #EBE6DF;
        border-radius: 8px;
        overflow: hidden;
    }

    /* DataFrame text */
    .dataframe td, .dataframe th {
        color: #2A2A2A !important;
    }

    /* Success and info messages */
    .stSuccess, .stInfo, .stWarning {
        color: #1A1A1A !important;
    }

    /* Error messages */
    .stError {
        color: #8B261D !important;
    }

    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #E5E0D8, transparent);
        margin: 2rem 0;
    }

    /* Title badge styling */
    .title-badge {
        background-color: #C5A059;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
ALL_STATES = [
    "Current",
    "30 Days Delinquent",
    "60 Days Delinquent",
    "90+ Days Delinquent",
]

# Fannie Mae single-family pipe-delimited file column headers
COLUMN_HEADERS = [
    "Reference Pool ID", "Loan Identifier", "Monthly Reporting Period", "Channel",
    "Seller Name", "Servicer Name", "Master Servicer", "Original Interest Rate",
    "Current Interest Rate", "Original UPB", "UPB at Issuance", "Current Actual UPB",
    "Original Loan Term", "Origination Date", "First Payment Date", "Loan Age",
    "Remaining Months to Legal Maturity", "Remaining Months To Maturity", "Maturity Date",
    "Original Loan to Value Ratio (LTV)", "Original Combined Loan to Value Ratio (CLTV)",
    "Number of Borrowers", "Debt-To-Income (DTI)", "Borrower Credit Score at Origination",
    "Co-Borrower Credit Score at Origination", "First Time Home Buyer Indicator",
    "Loan Purpose", "Property Type", "Number of Units", "Occupancy Status",
    "Property State", "Metropolitan Statistical Area (MSA)", "Zip Code Short",
    "Mortgage Insurance Percentage", "Amortization Type", "Prepayment Penalty Indicator",
    "Interest Only Loan Indicator",
    "Interest Only First Principal And Interest Payment Date",
    "Months to Amortization", "Current Loan Delinquency Status", "Loan Payment History",
    "Modification Flag", "Mortgage Insurance Cancellation Indicator", "Zero Balance Code",
    "Zero Balance Effective Date", "UPB at the Time of Removal", "Repurchase Date",
    "Scheduled Principal Current", "Total Principal Current", "Unscheduled Principal Current",
    "Last Paid Installment Date", "Foreclosure Date", "Disposition Date",
    "Foreclosure Costs", "Property Preservation and Repair Costs",
    "Asset Recovery Costs", "Miscellaneous Holding Expenses and Credits",
    "Associated Taxes for Holding Property", "Net Sales Proceeds",
    "Credit Enhancement Proceeds", "Repurchase Make Whole Proceeds",
    "Other Foreclosure Proceeds", "Principal Forgiveness Amount",
    "Modification-Related Non-Interest Bearing UPB", "Original List Start Date",
    "Original List Price", "Current List Start Date", "Current List Price",
    "Borrower Credit Score At Issuance", "Co-Borrower Credit Score At Issuance",
    "Borrower Credit Score Current", "Co-Borrower Credit Score Current",
    "Mortgage Insurance Type", "Servicing Activity Indicator",
    "Current Period Modification Loss Amount", "Cumulative Modification Loss Amount",
    "Current Period Credit Event Net Gain or Loss",
    "Cumulative Credit Event Net Gain or Loss", "Special Eligibility Program",
    "Foreclosure Principal Write-off Amount", "Relocation Mortgage Indicator",
    "Zero Balance Code Change Date", "Loan Holdback Indicator",
    "Loan Holdback Effective Date", "Delinquent Accrued Interest",
    "High Balance Loan Indicator", "Property Valuation Method", "ARM Product Type",
    "Initial Fixed-Rate Period", "Interest Rate Adjustment Frequency",
    "Next Interest Rate Adjustment Date", "Next Payment Change Date", "Index",
    "ARM Initial Fixed-Rate Period ≤ 5 YR Indicator", "ARM Cap Structure",
    "Initial Interest Rate Cap Up Percent", "Periodic Interest Rate Cap Up Percent",
    "Lifetime Interest Rate Cap Up Percent", "Mortgage Margin", "ARM Plan Number",
    "ARM Balloon Indicator", "Borrower Assistance Plan",
    "High Loan to Value (HLTV) Refinance Option Indicator", "Deal Name",
    "Repurchase Make Whole Proceeds Flag", "Alternative Delinquency Resolution Count",
    "Alternative Delinquency Resolution", "Total Deferral Amount",
    "Payment Deferral Modification Event Indicator", "Interest Bearing UPB",
]

# ──────────────────────────────────────────────────────────────────────────────
# Placeholder / baseline transition matrix (used when no data is uploaded)
# Values are illustrative; rows sum to 1.
# ──────────────────────────────────────────────────────────────────────────────
PLACEHOLDER_MATRIX = pd.DataFrame(
    data=[
        [0.9550, 0.0450, 0.0000, 0.0000],   # Current → ...      (sums to 1)
        [0.6000, 0.2500, 0.1500, 0.0000],   # 30 Days → ...      (sums to 1)
        [0.2500, 0.2000, 0.3500, 0.2000],   # 60 Days → ...      (sums to 1)
        [0.0500, 0.0300, 0.0200, 0.9000],   # 90+ Days → ...     (sums to 1)
    ],
    index=ALL_STATES,
    columns=ALL_STATES,
    dtype=float,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def categorize_delinquency(status) -> str:
    """Map raw numeric delinquency code to a labelled state."""
    try:
        s = int(status)
    except (ValueError, TypeError):
        return "Unknown"
    if s == 0:
        return "Current"
    elif s == 1:
        return "30 Days Delinquent"
    elif s == 2:
        return "60 Days Delinquent"
    elif s >= 3:
        return "90+ Days Delinquent"
    return "Unknown"


@st.cache_data(show_spinner="Processing loan data…")
def load_and_process_data(
    uploaded_bytes_list: list,
    file_names: list,
    lgd: float,
    lookback_months: int,
) -> dict:
    """
    Parse uploaded Fannie Mae pipe-delimited CSV files, compute the transition
    matrix via MLE, and return portfolio summary statistics.

    Parameters
    ----------
    uploaded_bytes_list : list of bytes
        Raw file content for each uploaded file (bytes, for cache-key stability).
    file_names : list of str
        Original file names (used for cache key only).
    lgd : float
        Loss Given Default fraction used in expected-loss calculation.
    lookback_months : int
        Number of most-recent months to restrict the data to.

    Returns
    -------
    dict with keys:
        transition_matrix, unique_loans, total_loan_value,
        avg_loan_balance, delinquency_rate, expected_loss,
        state_counts
    """
    import io

    frames = []
    for raw in uploaded_bytes_list:
        chunk = pd.read_csv(
            io.BytesIO(raw),
            sep="|",
            header=None,
            names=COLUMN_HEADERS,
            low_memory=False,
        )
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)

    # ── Coerce types ──────────────────────────────────────────────────────────
    df["Current Loan Delinquency Status"] = pd.to_numeric(
        df["Current Loan Delinquency Status"], errors="coerce"
    )
    df["Current Actual UPB"] = pd.to_numeric(
        df["Current Actual UPB"], errors="coerce"
    )

    # ── Parse and sort by reporting period ───────────────────────────────────
    df["Monthly Reporting Period"] = pd.to_datetime(
        df["Monthly Reporting Period"], errors="coerce", infer_datetime_format=True
    )
    df = df.sort_values(["Loan Identifier", "Monthly Reporting Period"])

    # ── Apply lookback filter ─────────────────────────────────────────────────
    if lookback_months > 0:
        max_date = df["Monthly Reporting Period"].max()
        cutoff = max_date - pd.DateOffset(months=lookback_months)
        df = df[df["Monthly Reporting Period"] >= cutoff]

    # ── State labelling ───────────────────────────────────────────────────────
    df["State"] = df["Current Loan Delinquency Status"].apply(categorize_delinquency)
    df = df[df["State"] != "Unknown"]

    # ── Compute transitions (consecutive rows per loan) ───────────────────────
    df["Previous State"] = df.groupby("Loan Identifier")["State"].shift(1)
    df_transitions = df.dropna(subset=["Previous State"])

    # ── Count matrix ──────────────────────────────────────────────────────────
    counts = (
        df_transitions
        .groupby(["Previous State", "State"])
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.reindex(index=ALL_STATES, columns=ALL_STATES, fill_value=0)

    # ── MLE: row-normalise ────────────────────────────────────────────────────
    row_sums = counts.sum(axis=1)
    # Avoid divide-by-zero; rows with zero observations get uniform dist.
    transition_matrix = counts.div(row_sums.replace(0, np.nan), axis=0).fillna(
        1 / len(ALL_STATES)
    )

    # ── Apply logical delinquency-progression constraints ─────────────────────
    # A loan cannot skip from Current straight to 60 Days or 90+ Days.
    # A loan cannot skip from 30 Days Delinquent straight to 90+ Days.
    transition_matrix.loc["Current", "60 Days Delinquent"] = 0.0
    transition_matrix.loc["Current", "90+ Days Delinquent"] = 0.0
    transition_matrix.loc["30 Days Delinquent", "90+ Days Delinquent"] = 0.0

    # Re-normalise after zeroing constrained cells
    row_sums2 = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix.div(
        row_sums2.replace(0, np.nan), axis=0
    ).fillna(1 / len(ALL_STATES))

    # ── Portfolio statistics ──────────────────────────────────────────────────
    unique_loans = int(df["Loan Identifier"].nunique())
    total_loan_value = float(
        df.groupby("Loan Identifier")["Current Actual UPB"].max().sum()
    )
    avg_loan_balance = float(df["Current Actual UPB"].mean())
    delinquency_rate = float((df["Current Loan Delinquency Status"] > 0).mean())
    expected_loss = delinquency_rate * total_loan_value * lgd

    state_counts = df["State"].value_counts().reindex(ALL_STATES, fill_value=0)

    return {
        "transition_matrix": transition_matrix,
        "unique_loans": unique_loans,
        "total_loan_value": total_loan_value,
        "avg_loan_balance": avg_loan_balance,
        "delinquency_rate": delinquency_rate,
        "expected_loss": expected_loss,
        "state_counts": state_counts,
    }


def create_heatmap(df: pd.DataFrame, title: str = "", custom_colors: list = None):
    """Create an interactive Plotly heatmap with editorial gallery aesthetic."""
    if custom_colors:
        colorscale = custom_colors
    else:
        # Default editorial gradient: Soft Linen -> Amber Gold -> Deep Terracotta
        colorscale = [
            [0.0, '#F9F8F6'],    # Soft Linen
            [0.3, '#E5A93C'],    # Amber Gold
            [0.7, '#C5A059'],    # Muted Gold
            [1.0, '#8B261D']     # Deep Terracotta/Crimson
        ]
    
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale=colorscale,
        zmid=0.5,
        zmin=0,
        zmax=1,
        text=df.values.round(4),
        texttemplate="%{text:.4f}",
        textfont={"size": 12, "color": "#1A1A1A"},
        colorbar=dict(
            title="Probability",
            tickfont={"color": "#7A7265"}
        ),
        hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>Probability: %{z:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=title,
        title_font_size=16,
        title_font_color="#1A1A1A",
        title_x=0.5,
        xaxis_title="Target State",
        yaxis_title="Source State",
        xaxis_title_font_color="#7A7265",
        yaxis_title_font_color="#7A7265",
        xaxis_tickfont_color="#7A7265",
        yaxis_tickfont_color="#7A7265",
        width=600,
        height=500,
        margin=dict(l=100, r=50, t=60, b=50)
    )
    
    return fig


def create_bar_chart(df: pd.DataFrame, title: str = ""):
    """Create an editorial bar chart with warm gold/ochre gradient."""
    colors = ['#C5A059', '#D9822B', '#A8322D']  # Gold, Warm Ochre, Brick Red
    
    fig = go.Figure(data=go.Bar(
        x=df.index,
        y=df.iloc[:, 0],
        marker=dict(color=colors),
        text=df.iloc[:, 0].round(4),
        textposition='outside',
        textfont=dict(color="#1A1A1A", size=12),
    ))
    
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=title,
        title_font_size=14,
        title_font_color="#1A1A1A",
        title_x=0.5,
        xaxis_title="Starting State",
        yaxis_title="Probability",
        xaxis_title_font_color="#7A7265",
        yaxis_title_font_color="#7A7265",
        xaxis_tickfont_color="#7A7265",
        yaxis_tickfont_color="#7A7265",
        yaxis_range=[0, 1],
        showlegend=False,
        height=400,
        margin=dict(l=60, r=30, t=50, b=50)
    )
    
    return fig


def n_step_matrix(tm: pd.DataFrame, n: int) -> np.ndarray:
    """Raise transition matrix to the n-th power (Chapman-Kolmogorov)."""
    return np.linalg.matrix_power(tm.values.astype(float), n)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")

    lookback_period = st.slider(
        "Lookback Period (Months)",
        min_value=1,
        max_value=36,
        value=3,
        help="Number of most recent months of data used to estimate transition probabilities.",
    )
    lgd = st.number_input(
        "Loss Given Default (LGD)",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.01,
        format="%.2f",
        help="Fraction of outstanding balance lost upon default. Typically 0.20–0.60 for residential mortgages.",
    )

    st.markdown("---")
    st.markdown("### 📂 Data Source")
    st.markdown(
        "**Fannie Mae Single-Family Loan Data:**  \n"
        "[datadynamics.fanniemae.com]"
        "(https://datadynamics.fanniemae.com/data-dynamics/#/downloadLoanData/Single-Family)"
    )
    st.markdown(
        "**Example upload files:**  \n"
        "[Google Drive sample]"
        "(https://drive.google.com/drive/folders/1E0qwFf4Cg5Xw69ybYzorTticVxBHkaWO?usp=sharing)"
    )

    st.markdown("---")
    st.markdown("### 👤 About the Author")
    st.markdown(
        """
**Ketan Duggal**  
Senior Actuarial Analyst & Quantitative Modeler  
Visual Artist

[![GitHub](https://img.shields.io/badge/GitHub-333?style=flat&logo=github&logoColor=white)](https://github.com/ketanduggal)
[![Portfolio](https://img.shields.io/badge/Portfolio-C5A059?style=flat&logo=artstation&logoColor=white)](https://ketanduggal.github.io/artist-portfolio)
        """
    )
    st.caption("Quantitative Risk Lab")


# ──────────────────────────────────────────────────────────────────────────────
# Main header with title badge
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="title-badge">KETAN DUGGAL // QUANTITATIVE RISK LAB</div>', unsafe_allow_html=True)

st.markdown("## Markov Chain Single-Family Loan Model")
st.markdown(
    """
This application enables you to build and explore a **Markov Chain model** for single-family mortgage
delinquency using Fannie Mae loan performance data.  
Upload one or more pipe-delimited (`|`) CSV files below. All sections remain active with a
**placeholder baseline matrix** when no data has been uploaded.

> ⚠️ *Large datasets may exceed the memory limits of the hosted server. For heavy workloads,
> clone the [source repository](https://github.com/romanmichaelpaolucci/SLMC_Model) and run locally.*
"""
)

# ──────────────────────────────────────────────────────────────────────────────
# File upload
# ──────────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("### 📁 Single-Family Data Upload")

    uploaded_files = st.file_uploader(
        "Upload Fannie Mae loan performance CSV files (pipe-delimited, multiple chunks allowed)",
        type="csv",
        accept_multiple_files=True,
        help=(
            "Files must be contiguous monthly chunks from Fannie Mae's Single-Family dataset. "
            "Each file should use the standard pipe-delimited format with no header row."
        ),
    )

# ── Resolve data: uploaded or placeholder ─────────────────────────────────────
data_is_live = bool(uploaded_files)

if data_is_live:
    try:
        # Read bytes once; pass bytes for cache-key stability
        bytes_list = [f.read() for f in uploaded_files]
        names_list = [f.name for f in uploaded_files]

        result = load_and_process_data(
            bytes_list, names_list, lgd, lookback_period
        )

        transition_matrix = result["transition_matrix"]
        unique_loans      = result["unique_loans"]
        total_loan_value  = result["total_loan_value"]
        avg_loan_balance  = result["avg_loan_balance"]
        delinquency_rate  = result["delinquency_rate"]
        expected_loss     = result["expected_loss"]
        state_counts      = result["state_counts"]

        st.success(
            f"✅ Loaded **{len(uploaded_files)} file(s)** · "
            f"**{unique_loans:,} unique loans** identified."
        )
    except Exception as exc:
        st.error(f"❌ Error processing files: {exc}")
        data_is_live = False

if not data_is_live:
    st.info(
        "ℹ️ No data uploaded yet. The app is running with a **placeholder baseline matrix** "
        "so you can explore the model structure and calculators."
    )
    transition_matrix = PLACEHOLDER_MATRIX.copy()
    unique_loans      = 0
    total_loan_value  = 0.0
    avg_loan_balance  = 0.0
    delinquency_rate  = 0.0
    expected_loss     = 0.0
    state_counts      = pd.Series(0, index=ALL_STATES)

# ──────────────────────────────────────────────────────────────────────────────
# Portfolio Summary Statistics
# ──────────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("### 📊 Portfolio Summary Statistics")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Unique Loans", f"{unique_loans:,}" if data_is_live else "—")
    col2.metric("Total Portfolio UPB", f"${total_loan_value:,.0f}" if data_is_live else "—")
    col3.metric("Avg Loan Balance", f"${avg_loan_balance:,.0f}" if data_is_live else "—")
    col4.metric("Delinquency Rate", f"{delinquency_rate:.2%}" if data_is_live else "—")
    col5.metric(
        "Expected Loss (LGD-adjusted)",
        f"${expected_loss:,.0f}" if data_is_live else "—",
        help=f"Delinquency Rate × Total UPB × LGD ({lgd:.0%})",
    )

if data_is_live:
    st.markdown("#### State Distribution (current snapshot)")
    state_df = state_counts.reset_index()
    state_df.columns = ["Delinquency State", "Loan Count"]
    state_df["Share (%)"] = (state_df["Loan Count"] / state_df["Loan Count"].sum() * 100).round(2)
    st.dataframe(state_df, width='stretch', hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Advanced Analysis Section with Tabs
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 🔬 Advanced Risk Analysis")

analysis_tabs = st.tabs([
    "📊 Transition Matrix", 
    "🔭 Forward-Looking Probabilities", 
    "⚠️ Default Risk Analysis"
])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Transition Matrix Estimation
# ──────────────────────────────────────────────────────────────────────────────
with analysis_tabs[0]:
    with st.expander("📐 MLE Methodology & LaTeX Derivation", expanded=False):
        st.markdown(
            r"""
### Transition Matrix Estimation via Maximum Likelihood Estimation (MLE)

We model the evolution of a loan's delinquency status as a **discrete-time, finite-state Markov Chain**
over the state space:

$$S = \{\text{Current},\; \text{30 Days Delinquent},\; \text{60 Days Delinquent},\; \text{90+ Days Delinquent}\}$$

The goal is to estimate the **transition probability matrix** $P \in \mathbb{R}^{4 \times 4}$, where each
entry $P_{ij}$ denotes the probability of moving from state $i$ to state $j$ in one period (month).

---

#### Maximum Likelihood Estimate

Let $N_{ij}$ be the number of observed one-month transitions from state $i$ to state $j$
across all loans and all time periods. The MLE for each transition probability is:

$$\hat{P}_{ij} = \frac{N_{ij}}{\displaystyle\sum_{k \in S} N_{ik}}$$

This estimator ensures that:
1. Every row sums to 1: $\displaystyle\sum_{j \in S} \hat{P}_{ij} = 1 \quad \forall i$
2. All entries are non-negative: $\hat{P}_{ij} \geq 0$

---

#### Logical Constraints (Delinquency Progression)

Mortgage delinquency follows a natural progression. A loan that is *Current* cannot immediately
become *60 Days Delinquent* without first passing through *30 Days Delinquent*. Formally:

$$\hat{P}_{\text{Current},\;\text{60 Days}} = 0, \qquad \hat{P}_{\text{Current},\;\text{90+ Days}} = 0$$
$$\hat{P}_{\text{30 Days},\;\text{90+ Days}} = 0$$

After zeroing these cells the rows are **re-normalised** so they continue to sum to 1:

$$\hat{P}_{ij}^{*} = \frac{\hat{P}_{ij}}{\displaystyle\sum_{k} \hat{P}_{ik}}$$

---

#### Step-by-Step Process

1. **Categorise** each loan-month observation into one of the four states.  
2. **Sort** by loan ID and reporting period; shift states by one period to create `(Previous State, Current State)` pairs.  
3. **Count** transitions: build the $4 \times 4$ count matrix $N$.  
4. **Normalise** row-wise to obtain $\hat{P}$.  
5. **Apply constraints** and re-normalise to obtain $\hat{P}^*$.

> 📖 Further reading on MLE:
> [towardsdatascience.com/maximum-likelihood-estimation](https://towardsdatascience.com/maximum-likelihood-estimation-4a1a866dfa70)
            """
        )

    # ── Lookback Period expander ───────────────────────────────────────────────────
    with st.expander("🕐 Lookback Period Selection Guide"):
        st.markdown(
            """
The **lookback period** controls how many recent months of data feed into the MLE.

| Window | Best For |
|--------|----------|
| **1–3 months** | Near-real-time stress detection; highly sensitive to recent shocks |
| **3–6 months** | Short-term risk assessment; adapts quickly to economic shifts |
| **6–12 months** | Balanced; smooths seasonal variation while staying responsive |
| **12–18 months** | Regulatory / annual capital models; stable but slower to react |
| **18+ months** | Long-run equilibrium; use for stress-test baseline matrices |

Use the sidebar slider to adjust the lookback period and re-estimate the matrix instantly.
            """
        )

    # ── Display baseline / estimated transition matrix ────────────────────────────
    with st.container(border=True):
        st.markdown(
            "#### Baseline Transition Probability Matrix"
            + ("" if data_is_live else "  *(Placeholder — upload data to estimate)*")
        )

        st.markdown(
            "Each cell $P_{ij}$ is the estimated monthly probability of moving from **row state** → **column state**."
        )
        st.plotly_chart(create_heatmap(transition_matrix, "Baseline Transition Probability Matrix"), width='stretch')

        # Row-sum verification
        row_sums = transition_matrix.sum(axis=1).rename("Row Sum").to_frame()
        with st.expander("✅ Row-sum verification (all rows should equal 1.000)"):
            st.dataframe(row_sums.round(6), width='stretch')


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: Forward-Looking Probabilities
# ──────────────────────────────────────────────────────────────────────────────
with analysis_tabs[1]:
    st.markdown("#### Forward-Looking Probability Calculator")
    
    with st.expander("📖 Chapman-Kolmogorov Equation", expanded=False):
        st.markdown(
            r"""
### Chapman-Kolmogorov Equation

For a time-homogeneous Markov Chain with one-step transition matrix $P$, the
**$n$-step transition probability** from state $i$ to state $j$ is the $(i,j)$ entry of $P^n$:

$$P^{(n)}_{ij} = \left[P^n\right]_{ij}$$

This follows from the Chapman-Kolmogorov identity:

$$P^{(m+n)}_{ij} = \sum_{k \in S} P^{(m)}_{ik} \cdot P^{(n)}_{kj}$$

which says: to transition from $i$ to $j$ in $m+n$ steps, the chain must pass through
*some* intermediate state $k$ after $m$ steps.

#### Steady-State Distribution

As $n \to \infty$, for an **ergodic** (irreducible + aperiodic) chain, every row of $P^n$
converges to the unique **stationary distribution** $\pi$:

$$\pi_j = \lim_{n \to \infty} P^{(n)}_{ij} \quad (\text{independent of } i)$$

$\pi$ satisfies: $\pi P = \pi$ and $\sum_j \pi_j = 1$.

> 📽️ Video walkthrough: [YouTube – Chapman-Kolmogorov](https://www.youtube.com/watch?v=L3FqYBDw9fE)
        """
    )

    with st.container(border=True):
        st.markdown("##### Single Transition Probability")
        col_a, col_b, col_c = st.columns([2, 2, 1])
        start_state = col_a.selectbox("Start State", ALL_STATES, key="ck_start")
        end_state   = col_b.selectbox("End State",   ALL_STATES, key="ck_end")
        num_steps   = col_c.number_input("Months (n)", min_value=1, max_value=360, value=12, key="ck_n")

        if st.button("📊 Calculate Probability", key="ck_calc"):
            try:
                mat_n = n_step_matrix(transition_matrix, num_steps)
                i_idx = ALL_STATES.index(start_state)
                j_idx = ALL_STATES.index(end_state)
                prob  = mat_n[i_idx, j_idx]
                st.success(
                    f"P( **{start_state}** → **{end_state}** in **{num_steps}** month(s) ) = **{prob:.6f}** ({prob:.2%})"
                )
            except Exception as e:
                st.error(f"Calculation error: {e}")

    with st.container(border=True):
        st.markdown("##### Full n-Step Transition Matrix")
        n_full = st.number_input(
            "Months (n) for full matrix view", min_value=1, max_value=360, value=12, key="ck_full_n"
        )
        if st.button("📋 Show Full n-Step Matrix", key="ck_full_calc"):
            try:
                mat_n = n_step_matrix(transition_matrix, n_full)
                df_n  = pd.DataFrame(mat_n, index=ALL_STATES, columns=ALL_STATES)
                st.markdown(f"**$P^{{{n_full}}}$** — {n_full}-month forward transition probabilities:")
                st.plotly_chart(create_heatmap(df_n, f"{n_full}-Month Transition Matrix"), width='stretch')

                # Note on convergence
                if n_full >= 24:
                    st.info(
                        "ℹ️ At long horizons the rows tend to converge — this is the **stationary distribution** "
                        "of the chain. All starting states eventually reach the same long-run probability of being "
                        "in each delinquency bucket."
                    )
            except Exception as e:
                st.error(f"Calculation error: {e}")

    with st.expander("📚 Steady-State & Long-Run Equilibrium – Deep Dive"):
        st.markdown(
            r"""
### Steady-State Probabilities

A Markov Chain is **ergodic** (and thus has a unique stationary distribution) if it is:

1. **Irreducible** – every state is reachable from every other state.  
2. **Aperiodic** – the chain does not get stuck in deterministic cycles.  
3. **Positive Recurrent** – the expected return time to every state is finite.

The stationary distribution $\pi$ satisfies the **balance equation**:

$$\pi P = \pi, \quad \sum_{j} \pi_j = 1, \quad \pi_j \geq 0$$

Practically, $\pi$ can be found by:

- **Power iteration**: compute $\pi_0 P^n$ for large $n$.  
- **Eigenvalue method**: $\pi$ is the normalised left eigenvector of $P$ for eigenvalue 1.  
- **Linear algebra**: solve $\pi(P - I) = 0$ subject to $\mathbf{1}^\top \pi = 1$.

**Risk insight**: In the absorbing-state model (next section), the 90+ Days state *absorbs*
probability mass, so the chain is **not** ergodic and $P^n$ does not converge to a proper
stationary distribution — instead all probability mass flows into the absorbing state.
            """
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: Default Risk Analysis
# ──────────────────────────────────────────────────────────────────────────────
with analysis_tabs[2]:
    st.markdown("#### 90+ Days Delinquent as an Absorbing State")
    
    with st.expander("📖 Absorbing State Theory & Default Probability", expanded=False):
        st.markdown(
            r"""
### Absorbing State Formulation

In many credit-risk frameworks, once a loan reaches **90+ Days Delinquent** it is treated as
effectively in **default** — i.e., there is no realistic path back to performing status within
the model horizon.  We encode this by making the state **absorbing**:

$$P_{\text{90+, j}} = \begin{cases} 1 & \text{if } j = \text{90+ Days Delinquent} \\ 0 & \text{otherwise} \end{cases}$$

This modifies the matrix from an **ergodic** chain to an **absorbing** Markov Chain.

---

#### Fundamental Matrix & Eventual-Default Probability

Partition the state space into transient states $\mathcal{T}$ and the absorbing state $\mathcal{A}$:

$$P^* = \begin{pmatrix} Q & R \\ \mathbf{0} & 1 \end{pmatrix}$$

where $Q$ is the $|\mathcal{T}| \times |\mathcal{T}|$ sub-matrix of transitions among transient states
and $R$ is the $|\mathcal{T}| \times 1$ vector of transition probabilities into the absorbing state.

The **fundamental matrix** $N = (I - Q)^{-1}$ gives the expected number of periods spent in each
transient state before absorption. The **absorption probability vector** is:

$$b = N \cdot R = (I - Q)^{-1} R$$

$b_i$ is the probability that a loan currently in transient state $i$ will *eventually* default.

---

The Chapman-Kolmogorov calculator below uses $n$-step matrix powers of $P^*$ to show how
probability mass flows into the absorbing default state over time.
        """
    )

    # Build the absorbing matrix
    absorbing_matrix = transition_matrix.copy()
    absorbing_matrix.loc["90+ Days Delinquent"] = [0.0, 0.0, 0.0, 1.0]

    # Re-normalise remaining rows just in case
    for state in ["Current", "30 Days Delinquent", "60 Days Delinquent"]:
        row_sum = absorbing_matrix.loc[state].sum()
        if row_sum > 0:
            absorbing_matrix.loc[state] = absorbing_matrix.loc[state] / row_sum

    with st.container(border=True):
        st.markdown("##### Absorbing-State Transition Matrix $(P^*)$")
        st.markdown(
            "The **90+ Days Delinquent** row has been set to `[0, 0, 0, 1]` — once absorbed, always absorbed."
        )
        # Use warm terracotta gradient for absorbing matrix
        absorbing_colors = [
            [0.0, '#F9F8F6'],    # Soft Linen
            [0.4, '#E5A93C'],    # Amber Gold
            [0.7, '#A8322D'],    # Brick Red
            [1.0, '#8B261D']     # Deep Terracotta
        ]
        st.plotly_chart(create_heatmap(absorbing_matrix, "Absorbing-State Transition Matrix", absorbing_colors), width='stretch')

    # ── Fundamental-matrix analysis ───────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("##### Fundamental Matrix Analysis — Eventual Default Probabilities")
        try:
            transient_states = ["Current", "30 Days Delinquent", "60 Days Delinquent"]
            Q = absorbing_matrix.loc[transient_states, transient_states].values.astype(float)
            R = absorbing_matrix.loc[transient_states, "90+ Days Delinquent"].values.astype(float)

            I = np.eye(len(transient_states))
            N_fund = np.linalg.inv(I - Q)                # Fundamental matrix
            absorption_probs = N_fund @ R                 # b = N · R

            fund_df = pd.DataFrame(N_fund, index=transient_states, columns=transient_states)
            abs_df  = pd.DataFrame(
                {"Eventual Default Probability": absorption_probs},
                index=transient_states,
            )

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.markdown("**Fundamental Matrix** $N = (I - Q)^{-1}$")
                st.markdown("*Entry $N_{ij}$: expected months spent in state $j$ before default, starting from state $i$.*")
                # Use warm blue gradient for fundamental matrix
                fund_colors = [
                    [0.0, '#F9F8F6'],    # Soft Linen
                    [0.5, '#D4A574'],    # Warm Brown
                    [1.0, '#8B7355']     # Deep Brown
                ]
                st.plotly_chart(create_heatmap(fund_df, "Fundamental Matrix", fund_colors), width='stretch')
            with col_f2:
                st.markdown("**Eventual Default Probability** $b = N \\cdot R$")
                st.markdown("*The long-run probability that a loan currently in state $i$ will eventually default.*")
                st.plotly_chart(create_bar_chart(abs_df, "Eventual Default Probability by State"), width='stretch')

            # Expected Loss with absorbing-state probabilities
            if data_is_live:
                st.markdown("###### Expected Loss — Absorbing-State Model")
                st.markdown(
                    "Combining the **eventual default probability** with **LGD** and **current UPB** by state:"
                )
                el_rows = []
                for state in transient_states:
                    idx   = transient_states.index(state)
                    p_def = absorption_probs[idx]
                    el_rows.append(
                        {
                            "State": state,
                            "Eventual Default Prob": p_def,
                            f"Exp. Loss (LGD={lgd:.0%})": f"${p_def * total_loan_value * lgd:,.0f}",
                        }
                    )
                st.dataframe(pd.DataFrame(el_rows), width='stretch', hide_index=True)

        except np.linalg.LinAlgError:
            st.warning("⚠️ Fundamental matrix could not be computed (singular matrix). This can occur with placeholder data.")

    # ── Absorbing-state n-step calculator ─────────────────────────────────────────
    with st.container(border=True):
        st.markdown("##### Absorbing-State Transition Probability Calculator")
        st.markdown(
            "Use this calculator to see how quickly probability mass flows into the **90+ Days Delinquent** absorbing state."
        )

        col_x, col_y, col_z = st.columns([2, 2, 1])
        abs_start = col_x.selectbox("Start State", ALL_STATES, key="abs_start")
        abs_end   = col_y.selectbox("End State",   ALL_STATES, key="abs_end",
                                     index=ALL_STATES.index("90+ Days Delinquent"))
        abs_n     = col_z.number_input("Months (n)", min_value=1, max_value=360, value=24, key="abs_n")

        if st.button("📊 Calculate Absorbing Probability", key="abs_calc"):
            try:
                mat_abs_n = n_step_matrix(absorbing_matrix, abs_n)
                i_idx = ALL_STATES.index(abs_start)
                j_idx = ALL_STATES.index(abs_end)
                prob_abs = mat_abs_n[i_idx, j_idx]
                st.success(
                    f"P*( **{abs_start}** → **{abs_end}** in **{abs_n}** month(s) ) = **{prob_abs:.6f}** ({prob_abs:.2%})"
                )
                if abs_end == "90+ Days Delinquent":
                    exp_loss_abs = prob_abs * total_loan_value * lgd if data_is_live else None
                    if exp_loss_abs is not None:
                        st.metric(
                            label=f"Expected Loss over {abs_n} months (LGD={lgd:.0%})",
                            value=f"${exp_loss_abs:,.0f}",
                        )
            except Exception as e:
                st.error(f"Calculation error: {e}")

    with st.container(border=True):
        st.markdown("##### Full Absorbing n-Step Matrix")
        abs_n_full = st.number_input(
            "Months (n) for full absorbing matrix view", min_value=1, max_value=360, value=24, key="abs_full_n"
        )
        if st.button("📋 Show Full Absorbing n-Step Matrix", key="abs_full_calc"):
            try:
                mat_abs_n = n_step_matrix(absorbing_matrix, abs_n_full)
                df_abs_n  = pd.DataFrame(mat_abs_n, index=ALL_STATES, columns=ALL_STATES)
                st.markdown(f"**$(P^*)^{{{abs_n_full}}}$** — {abs_n_full}-month absorbing forward probabilities:")
                st.plotly_chart(create_heatmap(df_abs_n, f"{abs_n_full}-Month Absorbing Transition Matrix", absorbing_colors), width='stretch')
                st.info(
                    "Notice the **90+ Days Delinquent column** grows monotonically as $n$ increases — "
                    "all probability mass eventually flows into the absorbing default state. "
                    "This is *not* a stationary distribution; it reflects the eventual absorption of all loans."
                )
            except Exception as e:
                st.error(f"Calculation error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**SLMC Model** · Markov Chain Single-Family Loan Delinquency Modelling · "
    "Author: [Ketan Duggal](https://github.com/ketanduggal) · "
    "Powered by [Streamlit](https://streamlit.io), [pandas](https://pandas.pydata.org), "
    "[NumPy](https://numpy.org), [Plotly](https://plotly.com)"
)
