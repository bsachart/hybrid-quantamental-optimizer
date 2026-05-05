# UI Workflow Contract: Portfolio Optimizer Screen

## Scope

This contract captures the user-visible behavior that the simplified optimizer
screen must preserve or improve.

## Initial Setup State

- The screen shows a concise product header and a clearly ordered setup
  workflow.
- The user can immediately identify:
  - where to upload the price history file
  - where to upload the asset metrics file
  - where to set the risk-free rate and risk model
  - where the primary solve action lives
- The solve action remains unavailable until both required files are present.
- Sample file guidance remains accessible from the upload area without
  dominating the default view.

## Ready-To-Solve State

- Once both files are present, the screen confirms readiness in plain language.
- The user can see which files are loaded and which assumptions will be used
  for the next solve.
- The screen exposes one clear primary action to solve the risky portfolio.

## Solved State

- After a successful solve, the screen presents:
  - the tangency portfolio summary first
  - the target-volatility adjustment second
  - the final portfolio summary and allocation analysis third
- The final allocation remains on the same screen as the target-volatility
  control and visual analysis.
- The result export action remains available.

## Error And Recovery State

- If the solve fails, the screen shows a plain-language error message that helps
  the user understand what needs correction.
- A failed solve does not hide the setup controls needed for recovery.
- Missing-input states always explain what is still required before solving.

## Adjustment Behavior

- Changing target volatility updates the final allocation without requiring the
  user to re-upload files.
- Changing target volatility does not require the user to rerun the risky
  portfolio solve unless the source files or core setup inputs change.
