import numpy as np

from src.engine.portfolio_math import (
    LabeledPortfolioMetrics,
    build_labeled_portfolio_metrics,
    create_cash_portfolio,
    generate_target_volatility_series,
    scale_portfolio_to_target_volatility,
)


def sample_portfolio() -> LabeledPortfolioMetrics:
    return {
        "weights": np.array([0.4, 0.6]),
        "expected_return": 0.12,
        "volatility": 0.20,
        "sharpe_ratio": 0.50,
        "cash_weight": 0.0,
        "tickers": ["A", "B"],
        "asset_returns": [0.08, 0.16],
        "asset_vols": [0.10, 0.30],
    }


def test_build_labeled_portfolio_metrics():
    raw_metrics = {
        "weights": np.array([0.25, 0.75]),
        "expected_return": 0.11,
        "volatility": 0.18,
        "sharpe_ratio": 0.40,
        "cash_weight": 0.0,
    }
    expected_returns = np.array([0.07, 0.15])
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.09]])

    result = build_labeled_portfolio_metrics(
        raw_metrics=raw_metrics,
        tickers=["A", "B"],
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
    )

    assert result["tickers"] == ["A", "B"]
    assert result["asset_returns"] == [0.07, 0.15]
    np.testing.assert_allclose(result["asset_vols"], [0.2, 0.3])


def test_scale_portfolio_to_target_volatility_scalar():
    tangency = sample_portfolio()

    result = scale_portfolio_to_target_volatility(tangency, 0.10, 0.02)

    np.testing.assert_allclose(result["weights"], [0.2, 0.3])
    assert np.isclose(result["cash_weight"], 0.5)
    assert np.isclose(result["expected_return"], 0.07)
    assert np.isclose(result["volatility"], 0.10)
    assert result["tickers"] == ["A", "B"]


def test_scale_portfolio_to_target_volatility_caps_at_tangency():
    tangency = sample_portfolio()

    result = scale_portfolio_to_target_volatility(tangency, 0.35, 0.02)

    np.testing.assert_allclose(result["weights"], tangency["weights"])
    assert np.isclose(result["cash_weight"], 0.0)
    assert np.isclose(result["expected_return"], tangency["expected_return"])
    assert np.isclose(result["volatility"], tangency["volatility"])


def test_scale_portfolio_to_target_volatility_list_input():
    tangency = sample_portfolio()

    result = scale_portfolio_to_target_volatility(tangency, [0.05, 0.10], 0.02)

    assert len(result) == 2
    assert np.isclose(result[0]["cash_weight"], 0.75)
    assert np.isclose(result[1]["cash_weight"], 0.5)


def test_create_cash_portfolio():
    tangency = sample_portfolio()

    result = create_cash_portfolio(tangency, 0.03)

    np.testing.assert_allclose(result["weights"], [0.0, 0.0])
    assert np.isclose(result["expected_return"], 0.03)
    assert result["tickers"] == tangency["tickers"]


def test_zero_volatility_portfolio_scales_to_cash():
    tangency = sample_portfolio()
    tangency["volatility"] = 0.0

    result = scale_portfolio_to_target_volatility(tangency, 0.10, 0.02)

    np.testing.assert_allclose(result["weights"], [0.0, 0.0])
    assert np.isclose(result["cash_weight"], 1.0)
    assert np.isclose(result["expected_return"], 0.02)


def test_generate_target_volatility_series_includes_endpoint():
    targets = generate_target_volatility_series(0.25, vol_step=0.10)

    np.testing.assert_allclose(targets, [0.0, 0.1, 0.2, 0.25])


def test_generate_target_volatility_series_rejects_non_positive_step():
    try:
        generate_target_volatility_series(0.25, vol_step=0.0)
    except ValueError as exc:
        assert "vol_step must be positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive step size")
