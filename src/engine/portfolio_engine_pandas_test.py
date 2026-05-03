import numpy as np
import pandas as pd

from src.engine.portfolio_engine_pandas import optimize_portfolio, target_portfolio
from src.engine.risk_pandas import RiskModel


def mock_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "a": [100.0, 101.0, 102.0, 103.0],
            "b": [100.0, 105.0, 95.0, 105.0],
        }
    )


def mock_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["a", "b"],
            "expected_return": [0.05, 0.15],
            "implied_volatility": [0.10, 0.30],
            "min_weight": [0.0, 0.0],
            "max_weight": [1.0, 1.0],
        }
    )


def test_pandas_optimize_portfolio_default_risk_model():
    result = optimize_portfolio(
        price_source=mock_prices(),
        metric_source=mock_metrics(),
        risk_free_rate=0.04,
    )

    assert result["tickers"] == ["a", "b"]
    assert np.isclose(np.sum(result["weights"]), 1.0)
    assert result["cash_weight"] == 0.0


def test_pandas_target_portfolio_scaling():
    tangency = optimize_portfolio(
        price_source=mock_prices(),
        metric_source=mock_metrics(),
        risk_model=RiskModel.FORWARD_LOOKING,
        risk_free_rate=0.02,
    )

    final = target_portfolio(
        tangency_portfolio=tangency,
        target_volatility=tangency["volatility"] * 0.5,
        risk_free_rate=0.02,
    )

    assert np.isclose(final["cash_weight"], 0.5)
    assert np.isclose(np.sum(final["weights"]) + final["cash_weight"], 1.0)
