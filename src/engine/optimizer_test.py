"""
Tests for the Mean-Variance Optimizer.

Covering:
1. Mathematical correctness (Tangency portfolio logic).
2. Structural features (CML projection, Vectorized inputs).
3. Constraints (Bounds).
4. Edge cases (Zero returns/volatility).
"""

import pytest
import numpy as np
from src.engine.optimizer import find_tangency_portfolio, project_along_cml


@pytest.fixture
def sample_data():
    """
    Creates a simple 2-asset universe for testing.

    Asset A: High Return (15%), High Vol (20%) -> Variance 0.04
    Asset B: Low Return (5%), Low Vol (5%)     -> Variance 0.0025
    Correlation: 0.0 (Uncorrelated)
    """
    mu = np.array([0.15, 0.05])

    # Covariance Matrix
    # [ Var(A)   Cov(A,B) ]
    # [ Cov(B,A) Var(B)   ]
    sigma = np.array([[0.04, 0.00], [0.00, 0.0025]])
    return mu, sigma


def test_find_tangency_portfolio_math(sample_data):
    """
    Verifies the optimizer finds the mathematically correct Tangency Portfolio.
    """
    mu, sigma = sample_data
    rf = 0.02  # 2% Risk Free Rate

    result = find_tangency_portfolio(mu, sigma, risk_free_rate=rf)

    # --- 1. Check Weights Sum to 1.0 (Fully Invested) ---
    assert np.isclose(np.sum(result["weights"]), 1.0)
    assert result["cash_weight"] == 0.0

    # --- 2. Mathematical Logic Check ---
    # Formula for weight ratio with 0 correlation: (Excess Return) / Variance
    # Score A = (0.15 - 0.02) / 0.04   = 3.25
    # Score B = (0.05 - 0.02) / 0.0025 = 12.0
    #
    # Total Score = 15.25
    # Expected Weight A = 3.25 / 15.25 ≈ 21.3%
    # Expected Weight B = 12.0 / 15.25 ≈ 78.7%

    assert np.isclose(result["weights"][1], 0.7868, atol=0.001)

    # --- 3. Check Sharpe Ratio Calculation ---
    # Manually calc portfolio stats
    exp_ret = (0.2131 * 0.15) + (0.7869 * 0.05)
    exp_var = (0.2131**2 * 0.04) + (0.7869**2 * 0.0025)
    exp_vol = np.sqrt(exp_var)
    exp_sharpe = (exp_ret - rf) / exp_vol

    assert np.isclose(result["sharpe_ratio"], exp_sharpe, atol=0.01)


def test_project_cml_single(sample_data):
    """
    Test projecting to a single specific volatility target (CML).
    """
    mu, sigma = sample_data
    rf = 0.02

    # Step 1: Get Tangency
    tangency = find_tangency_portfolio(mu, sigma, risk_free_rate=rf)
    base_vol = tangency["volatility"]

    # Step 2: Target exactly 50% of the tangency volatility
    target_vol = base_vol * 0.5

    cml_port = project_along_cml(
        tangency, target_volatility=target_vol, risk_free_rate=rf
    )

    # --- Assertions ---
    # 1. Volatility should match target exactly
    assert np.isclose(cml_port["volatility"], target_vol)

    # 2. Weights should be exactly half of tangency weights
    np.testing.assert_allclose(cml_port["weights"], tangency["weights"] * 0.5)

    # 3. Cash should be exactly 0.5
    assert np.isclose(cml_port["cash_weight"], 0.5)


def test_project_cml_list(sample_data):
    """
    Test generating multiple points (Efficient Frontier) by passing a list.
    Verifies that the engine returns a list of results corresponding 1:1 to inputs.
    """
    mu, sigma = sample_data
    rf = 0.02

    # 1. Get Baseline
    tangency = find_tangency_portfolio(mu, sigma, risk_free_rate=rf)
    max_vol = tangency["volatility"]

    # 2. Define targets:
    # - 0.0 (Cash)
    # - 50% of Tangency (Mix)
    # - 150% of Tangency (Impossible -> Should Cap)
    targets = [0.0, max_vol * 0.5, max_vol * 1.5]

    # 3. Call with List
    results = project_along_cml(tangency, target_volatility=targets, risk_free_rate=rf)

    # --- Assertions ---
    assert isinstance(results, list)
    assert len(results) == 3

    # Point 0: 100% Cash
    assert np.isclose(results[0]["volatility"], 0.0)
    assert np.isclose(results[0]["cash_weight"], 1.0)

    # Point 1: 50% Risk
    assert np.isclose(results[1]["volatility"], max_vol * 0.5)

    # Point 2: Capped at Max (Tangency)
    # Even though we asked for 1.5x vol, it should return the Tangency portfolio
    assert np.isclose(results[2]["volatility"], max_vol)
    assert np.isclose(results[2]["cash_weight"], 0.0)


def test_bounds_list(sample_data):
    """Test the new list-of-tuples bounds interface."""
    mu, sigma = sample_data

    # Force Asset 0 to be between 40% and 50%
    # Force Asset 1 to be between 50% and 60%
    bounds = [(0.4, 0.5), (0.5, 0.6)]

    result = find_tangency_portfolio(mu, sigma, risk_free_rate=0.02, bounds=bounds)

    w = result["weights"]
    assert 0.4 <= w[0] <= 0.5
    assert 0.5 <= w[1] <= 0.6
    assert np.isclose(np.sum(w), 1.0)


def test_zero_returns():
    """
    Edge Case: When expected returns == risk_free_rate (0.0),
    the Sharpe Ratio is 0.0 for ALL portfolios.

    The optimizer should basically 'stay put' at the initial guess (Equal Weights)
    rather than drifting arbitrarily or failing.
    """
    mu = np.array([0.0, 0.0])
    # Diagonal covariance (independent assets)
    sigma = np.eye(2) * 0.01

    # rf = 0.0 implies Excess Return is 0.0 everywhere.
    result = find_tangency_portfolio(mu, sigma, risk_free_rate=0.0)

    # 1. Check Weights: Should be Equal Weights (0.5, 0.5)
    # Why? Because we initialize the solver with equal weights.
    # Since the gradient is zero (flat objective), it shouldn't move.
    np.testing.assert_allclose(result["weights"], [0.5, 0.5], atol=1e-6)

    # 2. Check Sharpe: Should be exactly 0.0
    assert np.isclose(result["sharpe_ratio"], 0.0)

    # 3. Check Return: 0.0
    assert np.isclose(result["expected_return"], 0.0)
