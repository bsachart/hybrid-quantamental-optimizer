"""
Shared portfolio math helpers used by both engine entry points.
"""

from typing import List, Optional, Sequence, Union

import numpy as np

from .optimizer import PortfolioMetrics

ZERO_VOLATILITY_EPSILON = 1e-8


class LabeledPortfolioMetrics(PortfolioMetrics):
    """Portfolio metrics enriched with ticker and asset statistics."""

    tickers: List[str]
    asset_returns: List[float]
    asset_vols: List[float]


def build_labeled_portfolio_metrics(
    raw_metrics: PortfolioMetrics,
    tickers: Sequence[str],
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> LabeledPortfolioMetrics:
    """Attach ticker labels and asset statistics to optimizer output."""
    return {
        "weights": raw_metrics["weights"],
        "expected_return": raw_metrics["expected_return"],
        "volatility": raw_metrics["volatility"],
        "sharpe_ratio": raw_metrics["sharpe_ratio"],
        "cash_weight": raw_metrics["cash_weight"],
        "tickers": list(tickers),
        "asset_returns": expected_returns.tolist(),
        "asset_vols": np.sqrt(np.diag(cov_matrix)).tolist(),
    }


def scale_portfolio_to_target_volatility(
    tangency_portfolio: LabeledPortfolioMetrics,
    target_volatility: Union[float, List[float]],
    risk_free_rate: float,
) -> Union[LabeledPortfolioMetrics, List[LabeledPortfolioMetrics]]:
    """Scale a tangency portfolio along the capital market line."""
    if isinstance(target_volatility, list):
        return [
            _scale_single_target_volatility(tangency_portfolio, target, risk_free_rate)
            for target in target_volatility
        ]

    return _scale_single_target_volatility(
        tangency_portfolio, float(target_volatility), risk_free_rate
    )


def generate_target_volatility_series(
    max_volatility: float, vol_step: float = 0.01, num_points: Optional[int] = None
) -> List[float]:
    """Generate target volatilities from cash to the tangency point."""
    if num_points is not None:
        return np.linspace(0, max_volatility, num_points).tolist()

    if vol_step <= 0:
        raise ValueError("vol_step must be positive")

    targets = np.arange(0, max_volatility, vol_step).tolist()
    if not targets or not np.isclose(targets[-1], max_volatility):
        targets.append(max_volatility)
    return targets


def generate_cml_portfolios(
    tangency_portfolio: LabeledPortfolioMetrics,
    risk_free_rate: float,
    vol_step: float = 0.01,
    num_points: Optional[int] = None,
) -> List[LabeledPortfolioMetrics]:
    """Generate labeled portfolios from zero volatility to the tangency point."""
    targets = generate_target_volatility_series(
        tangency_portfolio["volatility"], vol_step=vol_step, num_points=num_points
    )
    return scale_portfolio_to_target_volatility(
        tangency_portfolio, targets, risk_free_rate
    )


def create_cash_portfolio(
    base_portfolio: LabeledPortfolioMetrics, risk_free_rate: float
) -> LabeledPortfolioMetrics:
    """Return a 100% cash portfolio while preserving asset context."""
    return {
        "weights": np.zeros_like(base_portfolio["weights"]),
        "expected_return": risk_free_rate,
        "volatility": 0.0,
        "sharpe_ratio": 0.0,
        "cash_weight": 1.0,
        "tickers": base_portfolio["tickers"],
        "asset_returns": base_portfolio["asset_returns"],
        "asset_vols": base_portfolio["asset_vols"],
    }


def _scale_single_target_volatility(
    tangency_portfolio: LabeledPortfolioMetrics,
    target_volatility: float,
    risk_free_rate: float,
) -> LabeledPortfolioMetrics:
    tangency_volatility = tangency_portfolio["volatility"]
    if tangency_volatility < ZERO_VOLATILITY_EPSILON:
        return create_cash_portfolio(tangency_portfolio, risk_free_rate)

    ratio = min(target_volatility / tangency_volatility, 1.0)
    if ratio <= ZERO_VOLATILITY_EPSILON:
        return create_cash_portfolio(tangency_portfolio, risk_free_rate)

    cash_weight = 1.0 - ratio
    scaled_volatility = tangency_volatility * ratio
    scaled_return = (
        tangency_portfolio["expected_return"] * ratio
        + risk_free_rate * cash_weight
    )

    return {
        "weights": tangency_portfolio["weights"] * ratio,
        "expected_return": scaled_return,
        "volatility": scaled_volatility,
        "sharpe_ratio": tangency_portfolio["sharpe_ratio"],
        "cash_weight": cash_weight,
        "tickers": tangency_portfolio["tickers"],
        "asset_returns": tangency_portfolio["asset_returns"],
        "asset_vols": tangency_portfolio["asset_vols"],
    }
