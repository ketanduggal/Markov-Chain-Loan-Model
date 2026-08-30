# 🏛️ Markov Chain Single-Family Loan Delinquency Model

### Quantitative Risk Laboratory & Stochastic Credit Transition Modeling

> **Author:** Ketan Duggal — *Senior Actuarial Analyst & Quantitative Modeler | Visual Artist*

An enterprise-grade quantitative credit-risk engine that estimates and
forecasts mortgage delinquency migration using **discrete-time Markov Chains**
and **Maximum Likelihood Estimation (MLE)** on residential loan performance
data. The tool combines stochastic transition-matrix estimation,
Chapman-Kolmogorov forward projection, and absorbing-state fundamental-matrix
decomposition into a single interactive dashboard for expected credit-loss
analysis.

---

## 📌 Executive Overview

This modeling engine treats a residential mortgage portfolio as a finite-state
stochastic process. Each loan migrates between delinquency states on a monthly
cadence; the engine estimates the underlying transition probabilities from
empirical data and uses them to forecast portfolio-level credit migration and
expected loss.

### Core Methodologies

| Component | Mathematical Foundation | Application |
|-----------|------------------------|-------------|
| **Transition Matrix Estimation** | MLE: $\hat{P}_{ij} = \frac{N_{ij}}{\sum_{k} N_{ik}}$ | One-month discrete transition probabilities across four loan performance states |
| **Multi-Period Forecasting** | Chapman-Kolmogorov: $P^{(n)}_{ij} = [P^n]_{ij}$ | Portfolio-level delinquency migration horizons via matrix power exponentiation |
| **Terminal Absorption & Expected Loss** | Fundamental matrix: $N = (I - Q)^{-1}$ | Eventual default probability $b = N \cdot R$ and LGD-adjusted expected credit loss |
| **Interactive Dashboard** | Plotly heatmaps & dynamic calibration | Real-time visual interrogation of matrices, forecasts, and sensitivity |

### State Space

$$S = \{\text{Current},\; \text{30 Days Delinquent},\; \text{60 Days Delinquent},\; \text{90+ Days Delinquent}\}$$

The **90+ Days Delinquent** state is treated as an **absorbing state**
(default) in the terminal-risk framework — once a loan enters this state,
there is no modeled path back to performing status.

---

## 🧮 Mathematical Framework

### 1. Transition Probability Estimation (MLE)

Let $N_{ij}$ be the observed count of one-month transitions from state $i$
to state $j$. The maximum likelihood estimator for the transition probability
matrix is:

$$\hat{P}_{ij} = \frac{N_{ij}}{\displaystyle\sum_{k \in S} N_{ik}}$$

**Logical delinquency-progression constraints** are enforced: a loan cannot
skip states (e.g., Current → 60 Days is impossible without passing through
30 Days). Constrained cells are zeroed and rows are re-normalised:

$$\hat{P}_{ij}^{*} = \frac{\hat{P}_{ij}}{\displaystyle\sum_{k} \hat{P}_{ik}}$$

### 2. Chapman-Kolmogorov Forward Equations

For a time-homogeneous chain with one-step matrix $P$, the $n$-step
transition probability is the $(i,j)$ entry of $P^n$:

$$P^{(n)}_{ij} = \left[P^n\right]_{ij}$$

This follows from the Chapman-Kolmogorov identity:

$$P^{(m+n)}_{ij} = \sum_{k \in S} P^{(m)}_{ik} \cdot P^{(n)}_{kj}$$

### 3. Absorbing-State Fundamental Matrix Decomposition

Partition the state space into transient states $\mathcal{T}$ and the
absorbing default state $\mathcal{A}$:

$$P^* = \begin{pmatrix} Q & R \\ \mathbf{0} & 1 \end{pmatrix}$$

The **fundamental matrix** $N = (I - Q)^{-1}$ gives the expected number of
months spent in each transient state before absorption. The **eventual
default probability vector** is:

$$b = N \cdot R = (I - Q)^{-1} R$$

where $b_i$ is the probability that a loan currently in state $i$ will
eventually default. **Expected credit loss** is then:

$$\text{EL} = \sum_{i \in \mathcal{T}} b_i \cdot \text{UPB}_i \cdot \text{LGD}$$

---

## 🎨 Luxury Editorial Gallery Theme

The dashboard is styled with a refined editorial gallery aesthetic:

| Element | Specification |
|---------|--------------|
| **Canvas** | Warm linen/ivory `#F9F8F6` |
| **Cards** | Pure white `#FFFFFF` with subtle shadow and border |
| **Primary text** | Deep charcoal `#1A1A1A` |
| **Accent** | Warm sandstone gold `#C5A059` — buttons, active tabs, highlights |
| **Subheaders / metadata** | Tracked uppercase `letter-spacing: 1.5px; color: #7A7265` |
| **Plotly palettes** | Soft Linen → Amber Gold → Terracotta editorial gradients |

All matrices (transition, absorbing, fundamental) and probability
distributions are rendered as interactive Plotly heatmaps and bar charts with
`template="plotly_white"` and transparent backgrounds matching the ivory
canvas.

---

## 🛠️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ketanduggal/Markov-Chain-Loan-Model.git
   cd Markov-Chain-Loan-Model
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   # venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

The dashboard opens at `http://localhost:8501`.

---

## 📂 Data Source

The model consumes **Fannie Mae Single-Family Loan Performance Data**:
pipe-delimited (`|`) CSV files with no header row.

- **Official source:**
  [Fannie Mae Data Dynamics](https://datadynamics.fanniemae.com/data-dynamics/#/downloadLoanData/Single-Family)
- **Example upload files:**
  [Google Drive sample](https://drive.google.com/drive/folders/1E0qwFf4Cg5Xw69ybYzorTticVxBHkaWO?usp=sharing)

When no data is uploaded, the application runs with a **placeholder baseline
matrix** so all calculators and visualizations remain fully interactive.

---

## 📊 Dashboard Features

### Portfolio Summary Statistics
- Unique loan count
- Total portfolio unpaid principal balance (UPB)
- Average loan balance
- Delinquency rate
- LGD-adjusted expected loss

### Transition Matrix Estimation
- Interactive Plotly heatmap of the MLE-estimated transition matrix
- Row-sum verification (all rows sum to 1.000)
- MLE methodology with full LaTeX derivation
- Configurable lookback period (1–36 months)

### Forward-Looking Probability Calculator
- Single-transition probability: $P^{(n)}_{ij}$ for any state pair and horizon
- Full $n$-step matrix view ($P^n$) as an interactive heatmap
- Chapman-Kolmogorov equation derivation
- Steady-state & long-run equilibrium deep dive

### Default Risk Analysis (Absorbing State)
- Absorbing-state transition matrix $P^*$ visualization
- Fundamental matrix $N = (I - Q)^{-1}$ heatmap
- Eventual default probability $b = N \cdot R$ bar chart
- Absorbing-state $n$-step probability calculator
- LGD-adjusted expected loss by starting state

---

## 📦 Dependencies

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `streamlit` | 1.28.0 | Interactive web dashboard |
| `pandas` | 2.0.0 | Data processing & transition counting |
| `numpy` | 1.24.0 | Matrix algebra & linear algebra |
| `plotly` | 5.15.0 | Interactive heatmaps & charts |

---

## 🏗️ Project Structure

```
Markov-Chain-Loan-Model/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit & theme configuration
└── README.md                 # This file
```

---

## 👤 About the Author

**Ketan Duggal**
*Senior Actuarial Analyst & Quantitative Modeler | Visual Artist*

- [GitHub](https://github.com/ketanduggal)
- [Portfolio](https://ketanduggal.github.io/artist-portfolio)

---

## 📄 License

This project is intended for educational and portfolio demonstration purposes.
The underlying Fannie Mae data is subject to Fannie Mae's own terms of use.

---

*Powered by [Streamlit](https://streamlit.io),
[pandas](https://pandas.pydata.org),
[NumPy](https://numpy.org), and
[Plotly](https://plotly.com).*
