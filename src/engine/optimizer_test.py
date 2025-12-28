import pytest
import numpy as np
from src.engine.optimizer import find_tangency_portfolio, project_along_cml


@pytest.fixture
def sample_data():
    mu = np.array([0.15, 0.05])
    sigma = np.array([[0.04, 0.00], [0.00, 0.0025]])
    return mu, sigma


def test_find_tangency_portfolio_math(sample_data):
    mu, sigma = sample_data
    result = find_tangency_portfolio(mu, sigma, risk_free_rate=0.02)

    # Mathematical Logic Check (same as before)
    assert np.isclose(result["weights"][1], 0.7868, atol=0.001)
    assert np.isclose(np.sum(result["weights"]), 1.0)


def test_project_cml(sample_data):
    mu, sigma = sample_data
    tangency = find_tangency_portfolio(mu, sigma, risk_free_rate=0.02)
    target_vol = tangency["volatility"] * 0.5

    cml_port = project_along_cml(
        tangency, target_volatility=target_vol, risk_free_rate=0.02
    )

    assert np.isclose(cml_port["volatility"], target_vol)
    assert np.isclose(cml_port["cash_weight"], 0.5)


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
