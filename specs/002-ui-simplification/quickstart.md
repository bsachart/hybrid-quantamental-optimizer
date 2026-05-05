# Quickstart: UI Simplification For Portfolio Workflow

## 1. Run the app locally

```bash
streamlit run src/streamlit_app.py
```

## 2. Validate the default setup state

- Open the app with no files loaded.
- Confirm the upload area, model controls, and primary solve action are visible
  in a single workflow.
- Confirm the solve action is disabled and the status copy explains what is
  still needed.

## 3. Validate the solved flow with example files

- Upload `examples/sample_prices.csv`.
- Upload `examples/sample_metrics.csv`.
- Keep the default assumptions and solve the portfolio.
- Confirm the tangency summary appears before the target-volatility control and
  the final portfolio summary.

## 4. Validate target-volatility adjustment

- Move the target-volatility control below the tangency volatility.
- Confirm the final allocation updates without re-uploading data or rerunning
  the risky-portfolio solve.
- Confirm the final summary, allocation chart, and allocation table remain in
  sync.
- Move the target-volatility control above the tangency portfolio.
- Confirm the borrowing segment of the Capital Market Line appears and the
  final allocation reports borrowing instead of cash.

## 5. Run focused verification

```bash
pytest src/components/workflow_state_test.py \
  src/components/results_display_test.py \
  src/engine/portfolio_engine_pandas_test.py \
  src/engine/portfolio_math_test.py \
  src/engine/portfolio_engine_test.py
```
