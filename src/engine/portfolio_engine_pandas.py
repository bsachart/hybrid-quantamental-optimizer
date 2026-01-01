"""
Portfolio Engine - Pandas version for browser deployment.

Philosophy:
    - Deep Module: Simple public API hiding complex data loading and math.
    - Two-Stage Process:
        1. optimize_portfolio: Finds the optimal risky mix (Tangency).
        2. target_portfolio: Scales that mix to desired risk (Capital Market Line).
"""

import numpy as np
from typing import Union, List, Optional, cast

from src.engine.data_loader_pandas import load_universe, FileInput
from src.engine.risk_pandas import calculate_covariance, RiskModel

# FIX: Import only PortfolioMetrics (base), not LabeledPortfolioMetrics
from src.engine.optimizer import find_tangency_portfolio, PortfolioMetrics


# FIX: Define LabeledPortfolioMetrics here (Domain Concern), not in the Optimizer (Math Concern).
class LabeledPortfolioMetrics(PortfolioMetrics):
    """
    Extends standard metrics to include universe context (tickers, stats).
    """

    tickers: List[str]
    asset_returns: List[float]
    asset_vols: List[float]


def optimize_portfolio(
    price_source: FileInput,
    metric_source: FileInput,
    risk_free_rate: float,
    risk_model: RiskModel = "forward-looking",
    annualization_factor: Optional[int] = None,
) -> LabeledPortfolioMetrics:
    """
    Step 1: The Heavy Lifter.
    Loads data, builds the risk model, finds the Tangency Portfolio,
    and extracts universe statistics in a single pass.

    Args:
        price_source: Path to CSV or Pandas DataFrame (Prices).
        metric_source: Path to CSV or Pandas DataFrame (Metrics).
        risk_model: RiskModel.HISTORICAL or RiskModel.FORWARD_LOOKING.
        risk_free_rate: Risk-free rate (decimal, e.g. 0.04) for Sharpe calculation.
        annualization_factor: Required only if risk_model is HISTORICAL.

    Returns:
        The optimal risky portfolio (100% equity, 0% cash) enriched with asset stats.
    """
    # 1. Load and Align Data
    universe = load_universe(price_source, metric_source)
    tickers = universe.tickers

    # 2. Extract Optimization Inputs
    expected_returns = universe.metrics["expected_return"].values
    bounds = list(
        zip(
            universe.metrics["min_weight"].values,
            universe.metrics["max_weight"].values,
        )
    )

    # 3. Calculate Risk (Covariance)
    implied_vols = None
    if risk_model == RiskModel.FORWARD_LOOKING:
        implied_vols = universe.metrics["implied_volatility"].values

    cov_matrix = calculate_covariance(
        prices=universe.prices,
        risk_model=risk_model,
        annualization_factor=annualization_factor,
        implied_vols=implied_vols,
    )

    # 4. Solve for Tangency
    raw_metrics = find_tangency_portfolio(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate,
        bounds=bounds,
    )

    # 5. Extract Universe Statistics (for UI Visualization)
    asset_vols = np.sqrt(np.diag(cov_matrix))

    # 6. Inject Labels and Context
    # We create a new dict that satisfies LabeledPortfolioMetrics
    result: LabeledPortfolioMetrics = {
        "weights": raw_metrics["weights"],
        "expected_return": raw_metrics["expected_return"],
        "volatility": raw_metrics["volatility"],
        "sharpe_ratio": raw_metrics["sharpe_ratio"],
        "cash_weight": raw_metrics["cash_weight"],
        "tickers": tickers,
        "asset_returns": expected_returns.tolist(),
        "asset_vols": asset_vols.tolist(),
    }

    return result


def target_portfolio(
    tangency_portfolio: LabeledPortfolioMetrics,
    target_volatility: Union[float, List[float]],
    risk_free_rate: float,
) -> Union[LabeledPortfolioMetrics, List[LabeledPortfolioMetrics]]:
    """
    Step 2: The Constructor.
    Scales the Tangency Portfolio along the Capital Market Line (CML).
    Preserves universe context (tickers, returns, vols) in the output.
    """
    # Vectorized handling for List inputs
    if isinstance(target_volatility, list):
        return [
            target_portfolio(tangency_portfolio, tv, risk_free_rate)  # type: ignore
            for tv in target_volatility
        ]

    # --- Single Target Logic ---
    t_vol = tangency_portfolio["volatility"]

    # Edge case: If tangency has 0 vol, return 100% Cash
    if t_vol < 1e-8:
        return _create_cash_portfolio(tangency_portfolio, risk_free_rate)

    # Calculate Allocation Ratio (capped at 1.0 = no leverage)
    ratio = min(cast(float, target_volatility) / t_vol, 1.0)

    # Calculate New Weights
    cash_weight = 1.0 - ratio
    new_weights = tangency_portfolio["weights"] * ratio

    # Calculate New Metrics
    new_ret = (tangency_portfolio["expected_return"] * ratio) + (
        risk_free_rate * cash_weight
    )
    new_vol = t_vol * ratio

    # Sharpe remains constant along the CML
    new_sharpe = tangency_portfolio["sharpe_ratio"]
    if ratio < 1.0 and new_vol > 1e-8:
        new_sharpe = (new_ret - risk_free_rate) / new_vol

    return {
        "weights": new_weights,
        "expected_return": new_ret,
        "volatility": new_vol,
        "sharpe_ratio": new_sharpe,
        "cash_weight": cash_weight,
        "tickers": tangency_portfolio["tickers"],
        # Pass-through statistics for UI
        "asset_returns": tangency_portfolio["asset_returns"],
        "asset_vols": tangency_portfolio["asset_vols"],
    }


def generate_cml(
    tangency_portfolio: LabeledPortfolioMetrics,
    risk_free_rate: float,
    vol_step: float = 0.01,
    num_points: Optional[int] = None,
) -> List[LabeledPortfolioMetrics]:
    """
    Generate portfolios along the Capital Market Line.
    """
    max_vol = tangency_portfolio["volatility"]

    if num_points is not None:
        targets = np.linspace(0, max_vol, num_points).tolist()
    else:
        if vol_step <= 0:
            raise ValueError("vol_step must be positive")

        targets = np.arange(0, max_vol, vol_step).tolist()

        if not targets or not np.isclose(targets[-1], max_vol):
            targets.append(max_vol)

    result = target_portfolio(tangency_portfolio, targets, risk_free_rate)

    return cast(List[LabeledPortfolioMetrics], result)


def _create_cash_portfolio(
    base_portfolio: LabeledPortfolioMetrics, rf_rate: float
) -> LabeledPortfolioMetrics:
    """Helper to create a 100% cash portfolio, preserving asset context."""
    return {
        "weights": np.zeros_like(base_portfolio["weights"]),
        "expected_return": rf_rate,
        "volatility": 0.0,
        "sharpe_ratio": 0.0,
        "cash_weight": 1.0,
        "tickers": base_portfolio["tickers"],
        "asset_returns": base_portfolio["asset_returns"],
        "asset_vols": base_portfolio["asset_vols"],
    }
