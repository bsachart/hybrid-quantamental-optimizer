# Python API Contract: Portfolio Engine

## Scope

This contract captures the public behavior that the cleanup pass must preserve
for the portfolio engine entry points.

## `optimize_portfolio`

- Accepts price and metric sources compatible with the active engine backend.
- Returns a labeled tangency portfolio with:
  - risky-asset `weights`
  - `expected_return`
  - `volatility`
  - `sharpe_ratio`
  - `cash_weight` equal to `0.0`
  - `tickers`, `asset_returns`, and `asset_vols`
- Historical mode requires an annualization factor.
- Forward-looking mode uses implied volatilities from the metrics input.

## `target_portfolio`

- Accepts a tangency portfolio, a single target volatility or list of targets,
  and a risk-free rate.
- Returns:
  - one labeled portfolio for a scalar target
  - a list of labeled portfolios for a list input
- Preserves `tickers`, `asset_returns`, and `asset_vols`.
- Does not introduce leverage beyond the tangency portfolio in the current
  design; requests above tangency volatility are capped at the tangency point.
- Returns a 100% cash portfolio when the tangency portfolio has effectively
  zero volatility.

## `generate_cml`

- Accepts a tangency portfolio and risk-free rate.
- Generates portfolios from zero volatility to the tangency point.
- Supports either a fixed `vol_step` or an explicit `num_points` override.
- Always includes the exact tangency point at the end of the result set.
