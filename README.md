# 🏛️ Markov Chain Single-Family Loan Delinquency Model
### Quantitative Risk Laboratory & Stochastic Credit Transition Modeling

An enterprise-grade quantitative risk modeling engine that estimates and forecasts mortgage delinquency migration using discrete-time Markov Chains and Maximum Likelihood Estimation (MLE) on residential loan performance data.

---

## 📌 Executive Overview
* **Stochastic Transition Matrices**: Estimates one-month discrete transition probability matrices across loan performance states:
  $$\hat{P}_{ij} = \frac{N_{ij}}{\sum_{k} N_{ik}}$$
* **Multi-Period Forecasting**: Implements Chapman-Kolmogorov equations ($P^n$) via matrix power exponentiation to project portfolio-level delinquency migration horizons.
* **Terminal Absorption & Expected Loss**: Evaluates terminal default (90+ Days Delinquent) through fundamental matrix decomposition ($N = (I - Q)^{-1}$) and calculates Loss Given Default (LGD)-adjusted expected losses.
* **Interactive Quantitative Dashboard**: Features customized Plotly visualizations, dynamic lookback calibration, and sensitivity analysis.

---

## 🛠️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ketanduggal/Markov-Chain-Loan-Model.git](https://github.com/ketanduggal/Markov-Chain-Loan-Model.git)
   cd Markov-Chain-Loan-Model
