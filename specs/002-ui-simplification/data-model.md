# Data Model: UI Simplification For Portfolio Workflow

## Workflow Stage

- **Purpose**: Represents a user-visible step in the optimizer journey.
- **Fields**:
  - `name`
  - `title`
  - `description`
  - `status`
- **Invariants**:
  - stages appear in a fixed order from upload to solve to final allocation
  - a stage can be inactive, ready, active, or complete
  - the current stage messaging must match the user’s available next action

## Input Readiness

- **Purpose**: Represents whether the user has supplied everything needed to
  solve the risky portfolio.
- **Fields**:
  - `prices_loaded`
  - `metrics_loaded`
  - `ready_to_solve`
  - `status_title`
  - `status_message`
- **Invariants**:
  - `ready_to_solve` is true only when both required files are present
  - missing-file guidance must point to the next required action

## Portfolio Summary

- **Purpose**: Represents the headline metrics shown for either the tangency
  portfolio or the final cash-adjusted portfolio.
- **Fields**:
  - `label`
  - `expected_return`
  - `volatility`
  - `sharpe_ratio`
  - optional `cash_weight`
- **Invariants**:
  - tangency summaries do not introduce cash
  - final summaries show cash weight whenever a final allocation is displayed

## Results Workspace

- **Purpose**: Represents the solved state that connects the risky portfolio,
  target-volatility adjustment, and final allocation.
- **Fields**:
  - `tangency_portfolio`
  - `target_volatility`
  - `final_portfolio`
  - `cml_points`
  - optional `error_message`
- **Invariants**:
  - target-volatility changes update only the final allocation layer
  - chart, allocation table, and export output derive from the same final
    portfolio data
  - a failed solve keeps the setup controls available for correction
