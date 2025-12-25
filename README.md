# Hybrid Quantamental Optimizer

A portfolio construction tool designed to fix the "rear-view mirror" bias of Modern Portfolio Theory (MPT).

The project explicitly **decouples** the estimation of returns, the modeling of risk, and the mathematical optimization.

---

## 1. The Return Engine (Forecasting $\mu$)
We do not rely on past price performance to predict future returns. The engine supports two distinct modes of operation:

### Mode A: View-Based Alpha (Simple)
Best for tactical adjustments based on macro views or specific price targets.
*   **Logic:** $\mu = \text{Market Return} + \delta_{\text{alpha}}$
*   **Inputs:** A baseline market return (e.g., 0.08) and an asset-specific "Alpha Delta" (e.g., 0.02 for outperformance).

### Mode B: Fundamental Implied CAGR (Complex)
Best for long-term strategic investors. We derive an **Implied Annual Return** by simulating business performance over an $N$-year holding period.
*   **Linear Margin Ramp:** The model interpolates a path from current Net Profit Margin (NPM) to a target Terminal Margin.
*   **The Sales Boost:** Annual sales grow by the sum of organic growth and the current year's profit margin. This treats the business as a compounding machine where earnings are reinvested into top-line expansion.
*   **Organic Growth Note:** Includes Stock-Based Compensation (SBC), but excludes buybacks and dividend reinvestment.
*   **Logic:** $ \mu = (MC_{exit} / MC_{current})^{1/N} - 1 $

---

## 2. The Risk Engine (Forecasting $\Sigma$)
Historical volatility is often a lagging indicator. This engine uses a **Forward-Looking Risk** model.

*   **Implied Volatility (IV):** We anchor the risk of each asset to current options market pricing (IV).
*   **Correlation:** We preserve structural historical correlations to understand how assets move relative to one another.

---

## 3. The Optimization Engine (The Solver)
Once $\mu$ (Returns) and $\Sigma$ (Risk) are calculated, they are passed to a solver that maximizes the Sharpe Ratio.

**Key Features:**
*   **Asset-Specific Constraints:** Directional limits per asset (Long/Short/Both).
*   **Efficient Frontier Calculation:** Visualizes the optimal portfolio against the feasible region.

---

## Workflow
1. **Configure Universe:** Choose your mode (Simple or Fundamental) and set ticker-specific parameters.
2. **Fetch Market Data:** The script retrieves weekly price history and current market capitalizations.
3. **Generate Metrics:** The engine outputs `universe.csv` and `metrics.csv` containing forward-looking returns and IV.
4. **Optimize:** Ingest these files into the optimizer to find the Maximum Sharpe Ratio weights.