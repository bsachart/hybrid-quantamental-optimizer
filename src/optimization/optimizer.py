"""
Module for Mean-Variance Optimization (MVO).

Philosophy:
    - "Strategic programming": Design the optimizer to be extensible (e.g., adding constraints later).
    - Maximize Sharpe Ratio is the default objective.
    - Robustness: Handle edge cases where risk-free rate > portfolio return (though rare in long-only equity).

Reference:
    - Modern Portfolio Theory (Markowitz).
"""

import numpy as np
import scipy.optimize as sco
import numpy.typing as npt
from typing import Dict, Tuple, Optional, List


def portfolio_performance(
    weights: npt.NDArray[np.float64],
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_free_rate: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Calculates portfolio return, volatility, and Sharpe ratio.

    Args:
        weights: Asset weights (sum to 1).
        expected_returns: Annualized expected returns for each asset.
        cov_matrix: Annualized covariance matrix.
        risk_free_rate: Risk-free rate for Sharpe calculation.

    Returns:
        (return, volatility, sharpe_ratio)
    """
    weights = np.array(weights)
    ret = np.sum(weights * expected_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def negative_sharpe_ratio(
    weights: npt.NDArray[np.float64],
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Objective function to minimize (negative Sharpe Ratio).
    """
    _, _, sharpe = portfolio_performance(
        weights, expected_returns, cov_matrix, risk_free_rate
    )
    return -sharpe


def optimize_portfolio(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_free_rate: float = 0.02,
    max_weight: float = 1.0,
) -> Dict[str, any]:
    """
    Finds the optimal portfolio weights that maximize the Sharpe Ratio.

    Constraints:
    1. Sum of weights = 1 (Fully invested).
    2. Weights >= 0 (Long only).
    3. Weights <= max_weight (Concentration limit).

    Args:
        expected_returns: 1D array of annualized expected returns.
        cov_matrix: 2D array of annualized covariance.
        risk_free_rate: Risk-free rate (default 2%).
        max_weight: Maximum weight for a single asset (default 1.0 = no limit).

    Returns:
        Dictionary containing:
        - 'weights': Optimal weights.
        - 'return': Portfolio return.
        - 'volatility': Portfolio volatility.
        - 'sharpe': Sharpe ratio.
        - 'success': Boolean indicating optimizer success.
        - 'message': Optimizer status message.
    """
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix, risk_free_rate)

    # Initial guess: Equal weights
    initial_weights = num_assets * [
        1.0 / num_assets,
    ]

    # Constraints
    # 1. Sum of weights = 1
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    # Bounds
    # 0 <= weight <= max_weight
    bounds = tuple((0.0, max_weight) for _ in range(num_assets))

    result = sco.minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        # Fallback or warning?
        # For now, we return the result but mark success as False
        pass

    optimal_weights = result.x
    ret, vol, sharpe = portfolio_performance(
        optimal_weights, expected_returns, cov_matrix, risk_free_rate
    )

    return {
        "weights": optimal_weights,
        "return": ret,
        "volatility": vol,
        "sharpe": sharpe,
        "success": result.success,
        "message": result.message,
    }


def portfolio_volatility(
    weights: npt.NDArray[np.float64], cov_matrix: npt.NDArray[np.float64]
) -> float:
    """
    Helper to calculate just volatility for minimization.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def generate_random_portfolios(
    num_portfolios: int,
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    frontier_points: Optional[List[Dict[str, float]]] = None,
) -> List[Dict[str, float]]:
    """
    Generates random portfolios to visualize the feasible set.
    
    Simple approach: Dirichlet sampling with varied concentration parameters
    to get a mix of concentrated and diversified portfolios.
    """
    results = []
    num_assets = len(expected_returns)
    
    # Simple Dirichlet sampling with varying concentration
    # Low alpha = concentrated portfolios (sparse, near vertices)
    # High alpha = diversified portfolios (near center)
    alphas = [0.1, 0.3, 0.5, 1.0, 2.0]
    n_per_alpha = num_portfolios // len(alphas)
    
    for alpha in alphas:
        for _ in range(n_per_alpha):
            weights = np.random.dirichlet(np.ones(num_assets) * alpha)
            ret = np.sum(weights * expected_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            results.append({"return": ret, "volatility": vol, "weights": weights})
    
    # Fill any remainder
    remainder = num_portfolios - len(results)
    for _ in range(remainder):
        weights = np.random.dirichlet(np.ones(num_assets))
        ret = np.sum(weights * expected_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results.append({"return": ret, "volatility": vol, "weights": weights})
    
    return results


def get_min_volatility_portfolio(
    expected_returns: npt.NDArray[np.float64], cov_matrix: npt.NDArray[np.float64]
) -> Dict[str, float]:
    """
    Finds the global minimum volatility portfolio.
    """
    num_assets = len(expected_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    result = sco.minimize(
        portfolio_volatility,
        initial_weights,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    ret = np.sum(result.x * expected_returns)

    return {"return": ret, "volatility": result.fun, "weights": result.x}


def calculate_efficient_frontier(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    num_points: int = 20,
) -> List[Dict[str, float]]:
    """
    Calculates the efficient frontier by minimizing volatility for a range of target returns.

    Args:
        expected_returns: Annualized expected returns.
        cov_matrix: Annualized covariance matrix.
        num_points: Number of points to calculate on the frontier.

    Returns:
        List of dictionaries [{'return': r, 'volatility': v}, ...]
    """
    num_assets = len(expected_returns)

    # Determine range of returns
    # Start from Global Minimum Variance Portfolio
    min_vol_port = get_min_volatility_portfolio(expected_returns, cov_matrix)
    min_ret = min_vol_port["return"]
    max_ret = np.max(expected_returns)

    # Create target returns grid
    # We add a small buffer to ensure feasible solutions
    if min_ret >= max_ret:
        # Edge case: all assets have same return or something weird
        target_returns = np.array([min_ret])
    else:
        target_returns = np.linspace(min_ret, max_ret, num_points)

    frontier_points = []

    # Initial guess
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))

    for target in target_returns:
        # Constraints:
        # 1. Sum of weights = 1
        # 2. Portfolio return = target
        # Note: Use default arg (t=target) to capture by value, not reference
        constraints = (
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {"type": "eq", "fun": lambda x, t=target: np.sum(x * expected_returns) - t},
        )

        result = sco.minimize(
            portfolio_volatility,
            initial_weights,
            args=(cov_matrix,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            frontier_points.append({"return": target, "volatility": result.fun, "weights": result.x})

    return frontier_points
