# Hybrid Quantamental Optimizer

> **Modern Portfolio Theory evolved for the fundamental investor.**

A portfolio construction engine that fixes the "rear-view mirror" bias of traditional MPT by explicitly decoupling forward-looking return forecasting, risk modeling, and mathematical optimization.

---

## 🌐 Try It Online

**Live Web Interface**: [https://bsachart.github.io/hybrid-quantamental-optimizer/](https://bsachart.github.io/hybrid-quantamental-optimizer/)

Use the web interface to:
- Upload your price history and asset metrics CSVs
- Configure optimization parameters interactively
- Visualize the Efficient Frontier and Capital Market Line
- Explore different risk/return allocations with real-time updates

*No installation required - runs entirely in your browser using Stlite (Streamlit + WebAssembly).*

---

## 1. Project Overview

Traditional MPT relies on historical returns to predict the future. This is fundamentally backward-looking and fails during regime changes.

**Our Approach**: Treat return forecasting, risk modeling, and optimization as three independent problems:

1.  **Return Engine**: Generate forward-looking expected returns (Fundamental CAGR).
2.  **Risk Engine**: Model covariance using forward-looking volatility (Implied Vol).
3.  **Optimization Engine**: A two-stage process:
    - **Stage 1**: Solve for the Tangency Portfolio (Pure Equity).
    - **Stage 2**: Construct the final portfolio along the Capital Market Line (Cash Mixing).

---

## 2. Quick Start

### Option A: Use the Web Interface (Recommended for exploration)

1. Visit [https://bsachart.github.io/hybrid-quantamental-optimizer/](https://bsachart.github.io/hybrid-quantamental-optimizer/)
2. Upload your data files (see [Data Specifications](#3-data-specifications))
3. Configure optimization parameters
4. View results and explore allocations

### Option B: Python API (For programmatic use)

#### Step 1: Generate Data

Fetch historical prices and compute fundamental metrics using the provided utility:

```bash
python src/scripts/generate_universe.py
```

#### Step 2: Run Optimization

A single function call orchestrates data loading, alignment, risk modeling, and solving.

```python
from src.engine.portfolio_engine import optimize_portfolio, target_portfolio, generate_cml
from src.engine.risk import RiskModel

# --- STAGE 1: Find the Tangency Portfolio ---
# This calculates the optimal mix of risky assets (Max Sharpe Ratio).
tangency_result = optimize_portfolio(
    price_source="tmp/universe.csv",
    metric_source="tmp/metrics.csv",
    risk_model=RiskModel.FORWARD_LOOKING,
    risk_free_rate=0.04
)

print(f"Max Sharpe: {tangency_result['sharpe_ratio']:.2f}")
print(f"Risky Volatility: {tangency_result['volatility']:.2%}")


# --- STAGE 2: Construct Final Portfolio (Target Risk) ---
# Scale the tangency portfolio to a specific volatility target (e.g., 10%)
# by mixing with Cash (Risk-Free Asset).
final_allocation = target_portfolio(
    tangency_portfolio=tangency_result,
    target_volatility=0.10,
    risk_free_rate=0.04
)

print(f"Cash Weight: {final_allocation['cash_weight']:.2%}")


# --- UTILITY: Generate Capital Market Line ---
# Generate points for plotting the Efficient Frontier / CML
# Default: Steps of 1% volatility
cml_points = generate_cml(
    tangency_portfolio=tangency_result,
    risk_free_rate=0.04,
    vol_step=0.01
)
```

---

## 3. Data Specifications

The engine requires two inputs (CSV files or Polars DataFrames).

### A. Price History (`universe.csv`)

Used to calculate correlation matrices ($\rho$).

- **Format**: Time-series.
- **Columns**: `date` (YYYY-MM-DD), followed by one column per ticker.

```csv
date,AAPL,GOOG,TSLA
2023-01-31,150.23,105.44,250.67
2023-02-28,152.11,108.22,255.33
```

### B. Asset Metrics (`metrics.csv`)

Used for Expected Returns ($\mu$), Volatilities ($\sigma$), and Constraints.

- **Format**: Cross-sectional.
- **Units**: **Decimals** (e.g., 0.12 = 12%).

```csv
ticker,expected_return,implied_volatility,min_weight,max_weight
AAPL,0.12,0.25,0.0,1.0
GOOG,0.15,0.28,0.0,1.0
TSLA,0.03,0.10,-0.5,0.5
```

| Column               | Description                            | Required For                |
| :------------------- | :------------------------------------- | :-------------------------- |
| `ticker`             | Symbol matching price CSV              | All                         |
| `expected_return`    | Annualized expected return (Decimal)   | All                         |
| `implied_volatility` | Forward-looking annual vol (Decimal)   | `RiskModel.FORWARD_LOOKING` |
| `min_weight`         | Minimum allocation (0.0 = long only)   | All                         |
| `max_weight`         | Maximum allocation (1.0 = no leverage) | All                         |

---

## 4. Methodology: Risk Models ($\Sigma$)

### Option A: Forward-Looking (Recommended)

Combines the **structure** of the past with the **magnitude** of the future.

- **Correlations**: Derived from price history.
- **Volatility**: Derived from Implied Volatility (Options Market).

```python
risk_model=RiskModel.FORWARD_LOOKING
# Requires 'implied_volatility' column in metrics.csv
```

### Option B: Historical

Classic MPT approach using sample covariance of historical returns.

```python
risk_model=RiskModel.HISTORICAL, annualization_factor=252
# annualization_factor is required (e.g., 252 for daily data)
```

---

## 5. Methodology: Two-Stage Optimization

The engine explicitly separates the mathematical solving from the portfolio construction.

### Stage 1: The Solver (`optimize_portfolio`)

Finds the **Tangency Portfolio** (Maximum Sharpe Ratio) considering only risky assets.
$$\text{maximize} \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}}$$

Subject to:

1.  $\sum w_i = 1$
2.  $w_{\min} \leq w_i \leq w_{\max}$

### Stage 2: The Constructor (`target_portfolio`)

Allocates capital between the Tangency Portfolio and the Risk-Free Asset to achieve a precise `target_volatility` ($\sigma_{target}$).

The weight allocated to the risky portfolio ($w_{risky}$) is:

$$w_{risky} = \min\left( \frac{\sigma_{target}}{\sigma_{tangency}}, 1.0 \right)$$

- If $\sigma_{target} < \sigma_{tangency}$: We hold Cash + Equity (Lending portfolio).
- If $\sigma_{target} \ge \sigma_{tangency}$: We hold 100% Tangency Portfolio (Leverage is explicitly capped at 1.0).
