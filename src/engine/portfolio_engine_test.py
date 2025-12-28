"""
Integration Tests for Portfolio Engine.

Tests the interaction between Data Loading, Risk Modeling, and Optimization.
"""

import pytest
import polars as pl
import numpy as np
from src.engine.portfolio_engine import (
    optimize_portfolio,
    target_portfolio,
    generate_cml,
)
from src.engine.risk import RiskModel


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_prices():
    """
    Deterministic prices.
    Asset A: Low Vol
    Asset B: High Vol
    """
    return pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "A": [100.0, 101.0, 102.0, 103.0],
            "B": [100.0, 105.0, 95.0, 105.0],
        }
    )


@pytest.fixture
def mock_metrics():
    """
    Metrics corresponding to Asset A and B.
    """
    return pl.DataFrame(
        {
            "ticker": ["A", "B"],
            "expected_return": [0.05, 0.15],
            "implied_volatility": [0.10, 0.30],
            "min_weight": [0.0, 0.0],
            "max_weight": [1.0, 1.0],
        }
    )


# ==============================================================================
# Optimization Tests
# ==============================================================================


def test_optimize_portfolio_returns_tangency(mock_prices, mock_metrics):
    """
    Test that optimize_portfolio returns a labeled, fully invested equity portfolio.
    """
    result = optimize_portfolio(
        price_source=mock_prices,
        metric_source=mock_metrics,
        risk_model=RiskModel.FORWARD_LOOKING,
        risk_free_rate=0.04,
        # annualization_factor is NOT passed (optional for forward-looking)
    )

    # Structure Check
    assert "tickers" in result
    assert result["tickers"] == ["A", "B"]
    assert "weights" in result

    # Logic Check: Tangency portfolio has 0% cash
    assert result["cash_weight"] == 0.0
    assert np.isclose(np.sum(result["weights"]), 1.0)
    assert result["sharpe_ratio"] > 0


def test_historical_risk_model_requires_factor(mock_prices, mock_metrics):
    """
    Verify that HISTORICAL model raises error if annualization_factor is missing.
    """
    with pytest.raises(ValueError, match="annualization_factor is required"):
        optimize_portfolio(
            price_source=mock_prices,
            metric_source=mock_metrics,
            risk_model=RiskModel.HISTORICAL,
            risk_free_rate=0.04,
            # Missing annualization_factor
        )


def test_historical_risk_model_success(mock_prices, mock_metrics):
    """
    Verify HISTORICAL model works when factor is provided.
    """
    result = optimize_portfolio(
        price_source=mock_prices,
        metric_source=mock_metrics,
        risk_model=RiskModel.HISTORICAL,
        risk_free_rate=0.04,
        annualization_factor=252,
    )
    assert result["volatility"] > 0


# ==============================================================================
# Targeting & CML Tests
# ==============================================================================


def test_target_portfolio_scaling(mock_prices, mock_metrics):
    """
    Test that target_portfolio correctly scales the tangency portfolio.
    """
    # 1. Get Tangency
    tangency = optimize_portfolio(
        mock_prices,
        mock_metrics,
        risk_model=RiskModel.FORWARD_LOOKING,
        risk_free_rate=0.02,
    )
    max_vol = tangency["volatility"]

    # 2. Scale to 50% risk
    target_vol = max_vol * 0.5
    final = target_portfolio(
        tangency_portfolio=tangency, target_volatility=target_vol, risk_free_rate=0.02
    )

    # Verify Scaling
    assert np.isclose(final["volatility"], target_vol)
    assert np.isclose(final["cash_weight"], 0.5)
    assert np.isclose(np.sum(final["weights"]) + final["cash_weight"], 1.0)

    # Verify Tickers Preserved
    assert final["tickers"] == ["A", "B"]


def test_efficient_frontier_list_input(mock_prices, mock_metrics):
    """
    Test generating multiple points via list input to target_portfolio.
    """
    tangency = optimize_portfolio(
        mock_prices,
        mock_metrics,
        risk_model=RiskModel.FORWARD_LOOKING,
        risk_free_rate=0.04,
    )

    targets = [0.05, 0.10, 0.15]
    frontier = target_portfolio(
        tangency, target_volatility=targets, risk_free_rate=0.04
    )

    assert isinstance(frontier, list)
    assert len(frontier) == 3
    assert np.isclose(frontier[0]["volatility"], 0.05)
    assert np.isclose(frontier[1]["volatility"], 0.10)
    assert frontier[2]["tickers"] == ["A", "B"]


def test_generate_cml_default_step(mock_prices, mock_metrics):
    """
    Test generate_cml uses vol_step=0.01 by default.
    """
    tangency = optimize_portfolio(
        mock_prices,
        mock_metrics,
        risk_model=RiskModel.FORWARD_LOOKING,
        risk_free_rate=0.04,
    )

    # Use default vol_step=0.01
    cml = generate_cml(tangency, risk_free_rate=0.04)

    assert len(cml) > 1

    # Check that the spacing between the first two points is ~0.01
    # Point 0 is Vol=0.0 (Cash)
    # Point 1 should be Vol=0.01
    assert np.isclose(cml[0]["volatility"], 0.0)
    if cml[-1]["volatility"] > 0.01:
        assert np.isclose(cml[1]["volatility"], 0.01)

    # Ensure last point is exactly the tangency portfolio
    assert np.isclose(cml[-1]["volatility"], tangency["volatility"])
    assert np.isclose(cml[-1]["expected_return"], tangency["expected_return"])


def test_generate_cml_custom_step(mock_prices, mock_metrics):
    """
    Test generate_cml with a custom step size.
    """
    tangency = optimize_portfolio(
        mock_prices,
        mock_metrics,
        risk_model=RiskModel.FORWARD_LOOKING,
        risk_free_rate=0.04,
    )

    # Large step size
    cml = generate_cml(tangency, risk_free_rate=0.04, vol_step=0.10)

    vols = [p["volatility"] for p in cml]

    # If tangency vol is e.g. 0.25, we expect [0.0, 0.1, 0.2, 0.25]
    # Check intermediate spacing
    if len(vols) > 2:
        assert np.isclose(vols[1] - vols[0], 0.10)


def test_generate_cml_num_points_override(mock_prices, mock_metrics):
    """
    Test that num_points overrides vol_step if provided.
    """
    tangency = optimize_portfolio(
        mock_prices,
        mock_metrics,
        risk_model=RiskModel.FORWARD_LOOKING,
        risk_free_rate=0.04,
    )

    # Request exactly 5 points
    cml = generate_cml(tangency, risk_free_rate=0.04, num_points=5)

    assert len(cml) == 5
    # First point = Cash
    assert np.isclose(cml[0]["volatility"], 0.0)
    # Last point = Tangency
    assert np.isclose(cml[-1]["volatility"], tangency["volatility"])
