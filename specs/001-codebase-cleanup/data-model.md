# Data Model: Codebase Cleanup And Calculation Verification

## Universe

- **Purpose**: Represents the aligned optimization input set.
- **Fields**:
  - `prices`: time series with `date` plus one column per ticker
  - `metrics`: per-ticker expected return, implied volatility, and weight bounds
  - `tickers`: sorted list shared by `prices` and `metrics`
- **Invariants**:
  - `prices` and `metrics` contain the same effective ticker set
  - `tickers` ordering drives the alignment of returns, volatilities, bounds,
    and portfolio weights

## Portfolio Metrics

- **Purpose**: Represents the optimizer output exposed by the engine.
- **Fields**:
  - `weights`
  - `expected_return`
  - `volatility`
  - `sharpe_ratio`
  - `cash_weight`
  - `tickers`
  - `asset_returns`
  - `asset_vols`
- **Invariants**:
  - risky-asset weights sum to `1.0` for a tangency portfolio
  - risky-asset weights plus `cash_weight` sum to `1.0` for a target portfolio
  - `asset_vols` match the square root of the covariance diagonal used for the
    selected risk model

## Risk Inputs

- **Purpose**: Represents the data needed to build a covariance matrix.
- **Fields**:
  - historical prices
  - selected risk model
  - optional annualization factor
  - optional implied volatility vector
- **Invariants**:
  - historical mode requires an annualization factor
  - forward-looking mode requires one implied volatility per asset

## Portfolio Target Request

- **Purpose**: Represents a request to scale the tangency portfolio along the
  capital market line.
- **Fields**:
  - base tangency portfolio
  - target volatility or list of target volatilities
  - risk-free rate
- **Invariants**:
  - target volatility below the tangency volatility introduces cash
  - target volatility at or above the tangency volatility does not introduce
    leverage in the current design
