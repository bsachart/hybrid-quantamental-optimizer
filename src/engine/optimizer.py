"""
Portfolio Optimizer - Mean-Variance Optimization engine.

Philosophy:
    - Deep Module: Separates 'Solving' (finding optimal weights) from 'Scaling' (risk targeting).
    - Robustness: Handles zero-volatility edge cases.
    - Vectorized: Handles lists of target volatilities efficiently.
"""

import numpy as np
import scipy.optimize as sco
from typing import TypedDict, Union, List, Tuple, Optional

# ==============================================================================
# Type Definitions
# ==============================================================================


class PortfolioMetrics(TypedDict):
    """
    Standardized portfolio output.
    Using TypedDict ensures autocomplete and documents the expected keys.
    """

    weights: np.ndarray  # Risky Asset weights
    expected_return: float
    volatility: float
    sharpe_ratio: float
    cash_weight: float  # Weight in Risk-Free Asset


# ==============================================================================
# Public Interface
# ==============================================================================


def find_tangency_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.04,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> PortfolioMetrics:
    """
    Solves for the Maximum Sharpe Ratio portfolio (The Tangency Portfolio).

    Args:
        expected_returns: Expected return vector (n_assets,)
        cov_matrix: Covariance matrix (n_assets, n_assets)
        risk_free_rate: Risk-free rate for Sharpe calculation
        bounds: List of (min, max) tuples for each asset.
               If None, defaults to (0.0, 1.0) for all assets (long-only).
    """
    n_assets = len(expected_returns)

    # Default to long-only (0.0 to 1.0) if no specific bounds provided
    if bounds is None:
        bounds = [(0.0, 1.0)] * n_assets

    # Objective: Minimize Negative Sharpe
    def negative_sharpe(w):
        ret = w @ expected_returns
        vol = np.sqrt(w @ cov_matrix @ w)
        if vol < 1e-8:
            return 0.0
        return -(ret - risk_free_rate) / vol

    # Constraints: Weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Initial Guess: Equal weights
    # Note: If bounds don't allow equal weights, SLSQP handles this by projecting
    x0 = np.ones(n_assets) / n_assets

    # Optimize
    result = sco.minimize(
        negative_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        tol=1e-8,
    )

    # Extract Results
    weights = result.x if result.success else x0

    # Calculate Metrics
    p_ret = weights @ expected_returns
    p_vol = np.sqrt(weights @ cov_matrix @ weights)
    sharpe = (p_ret - risk_free_rate) / p_vol if p_vol > 1e-8 else 0.0

    return {
        "weights": weights,
        "expected_return": p_ret,
        "volatility": p_vol,
        "sharpe_ratio": sharpe,
        "cash_weight": 0.0,
    }


def project_along_cml(
    tangency_portfolio: PortfolioMetrics,
    target_volatility: Union[float, List[float]],
    risk_free_rate: float = 0.04,
) -> Union[PortfolioMetrics, List[PortfolioMetrics]]:
    """
    Projects the Tangency Portfolio onto the Capital Market Line (CML).
    """
    # Vectorized handling for lists
    if isinstance(target_volatility, list):
        return [
            project_along_cml(tangency_portfolio, vol, risk_free_rate)  # type: ignore
            for vol in target_volatility
        ]

    # --- Single Value Logic ---
    t_vol = tangency_portfolio["volatility"]

    if t_vol < 1e-8:
        ratio = 0.0
    else:
        # Cap at 1.0 (No Leverage)
        ratio = min(target_volatility / t_vol, 1.0)  # type: ignore

    # Scale weights and metrics
    final_weights = tangency_portfolio["weights"] * ratio
    cash_weight = 1.0 - ratio

    final_ret = (tangency_portfolio["expected_return"] * ratio) + (
        risk_free_rate * cash_weight
    )
    final_vol = t_vol * ratio
    sharpe = tangency_portfolio["sharpe_ratio"] if ratio > 1e-8 else 0.0

    return {
        "weights": final_weights,
        "expected_return": final_ret,
        "volatility": final_vol,
        "sharpe_ratio": sharpe,
        "cash_weight": cash_weight,
    }
