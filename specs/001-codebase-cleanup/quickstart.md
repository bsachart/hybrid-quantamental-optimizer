# Quickstart: Codebase Cleanup And Calculation Verification

## 1. Run the core verification suite

```bash
pytest src/engine/optimizer_test.py \
  src/engine/risk_test.py \
  src/engine/data_loader_test.py \
  src/engine/portfolio_engine_test.py
```

## 2. Run the full engine-focused test sweep

```bash
pytest src/engine
```

## 3. Inspect the active feature work

```bash
git status --short
git log --oneline -5
```

## 4. Validate the public engine flow manually if needed

```python
from src.engine.portfolio_engine import optimize_portfolio
from src.engine.risk import RiskModel

result = optimize_portfolio(
    price_source="tmp/universe.csv",
    metric_source="tmp/metrics.csv",
    risk_model=RiskModel.FORWARD_LOOKING,
    risk_free_rate=0.04,
)

print(result["weights"])
print(result["expected_return"], result["volatility"], result["sharpe_ratio"])
```
