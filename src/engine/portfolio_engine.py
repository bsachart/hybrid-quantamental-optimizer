"""
Portfolio Engine - High-level orchestration for portfolio construction.

Two-stage portfolio construction:
    1. optimize_portfolio: Finds the optimal risky mix (Tangency Portfolio).
    2. target_portfolio: Scales that mix to desired risk via Capital Market Line.
"""

from typing import List, Optional, Union

from .data_loader import load_universe, FileInput
from .optimizer import find_tangency_portfolio
from .portfolio_math import (
    LabeledPortfolioMetrics,
    build_labeled_portfolio_metrics,
    generate_cml_portfolios,
    scale_portfolio_to_target_volatility,
)
from .risk import RiskModel, calculate_covariance


def optimize_portfolio(
    price_source: FileInput,
    metric_source: FileInput,
    risk_free_rate: float,
    risk_model: RiskModel = RiskModel.FORWARD_LOOKING,
    annualization_factor: Optional[int] = None,
) -> LabeledPortfolioMetrics:
    """
    Find the Tangency Portfolio (Maximum Sharpe Ratio).

    Loads data, builds the risk model, and returns the optimal risky portfolio.

    Args:
        price_source: Path to CSV or Polars DataFrame (Prices).
        metric_source: Path to CSV or Polars DataFrame (Metrics).
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
    expected_returns = universe.metrics["expected_return"].to_numpy()
    bounds = list(
        zip(
            universe.metrics["min_weight"].to_numpy(),
            universe.metrics["max_weight"].to_numpy(),
        )
    )

    # 3. Calculate Risk (Covariance)
    implied_vols = None
    if risk_model == RiskModel.FORWARD_LOOKING:
        implied_vols = universe.metrics["implied_volatility"].to_numpy()

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

    return build_labeled_portfolio_metrics(
        raw_metrics=raw_metrics,
        tickers=tickers,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
    )


def target_portfolio(
    tangency_portfolio: LabeledPortfolioMetrics,
    target_volatility: Union[float, List[float]],
    risk_free_rate: float,
) -> Union[LabeledPortfolioMetrics, List[LabeledPortfolioMetrics]]:
    """Scale the tangency portfolio along the capital market line."""
    return scale_portfolio_to_target_volatility(
        tangency_portfolio=tangency_portfolio,
        target_volatility=target_volatility,
        risk_free_rate=risk_free_rate,
    )


def generate_cml(
    tangency_portfolio: LabeledPortfolioMetrics,
    risk_free_rate: float,
    vol_step: float = 0.01,
    num_points: Optional[int] = None,
) -> List[LabeledPortfolioMetrics]:
    """Generate labeled portfolios along the capital market line."""
    return generate_cml_portfolios(
        tangency_portfolio=tangency_portfolio,
        risk_free_rate=risk_free_rate,
        vol_step=vol_step,
        num_points=num_points,
    )
