"""
Risk modeling: Covariance matrix construction.

Philosophy:
    - Deep Module: Hides linear algebra and scaling logic.
    - Explicit Interface: User explicitly defines the time-scaling factor.
    - Clean Execution: Suppresses internal numpy warnings for handled edge cases.
"""

import polars as pl
import numpy as np
from typing import Literal

RiskModel = Literal["historical", "forward-looking"]


def calculate_covariance(
    prices: pl.DataFrame,
    risk_model: RiskModel = "forward-looking",
    annualization_factor: int = 252,
    implied_vols: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate covariance matrix for portfolio optimization.

    Args:
        prices: DataFrame with 'date' column and ticker columns.
        risk_model:
            - 'historical': Uses past returns scaled by annualization_factor.
            - 'forward-looking': Uses historical correlation + implied_vols.
        annualization_factor: Number of periods per year to scale variance.
            - 252: Daily data (Equity default)
            - 365: Daily data (Crypto 24/7 markets)
            - 52:  Weekly data
            - 12:  Monthly data
            NOTE: Only affects 'historical' mode. 'forward-looking' uses
                  pre-annualized implied_vols.
        implied_vols: Annualized implied volatilities (Required for 'forward-looking').
                      These should already be in annual terms (e.g., 0.25 = 25% annual vol).
                      Must match the number of assets (columns - 1).

    Returns:
        n_assets x n_assets annualized covariance matrix.

    Raises:
        ValueError: If implied_vols missing or wrong shape for forward-looking model.
    """
    # 1. Prepare Data
    ticker_cols = [c for c in prices.columns if c != "date"]
    price_matrix = prices.select(ticker_cols).to_numpy()

    # Compute Log Returns: ln(P_t) - ln(P_{t-1})
    log_returns = np.diff(np.log(price_matrix), axis=0)

    # 2. Dispatch Logic
    if risk_model == "historical":
        cov = np.cov(log_returns, rowvar=False) * annualization_factor
        return np.atleast_2d(cov)

    elif risk_model == "forward-looking":
        if implied_vols is None:
            raise ValueError("implied_vols required for forward-looking risk model")

        if len(implied_vols) != log_returns.shape[1]:
            raise ValueError(
                f"Shape mismatch: {len(implied_vols)} implied_vols provided "
                f"for {log_returns.shape[1]} assets."
            )

        return _calculate_hybrid_covariance(log_returns, implied_vols)

    else:
        raise ValueError(f"Unknown risk_model: {risk_model}")


def _calculate_hybrid_covariance(
    log_returns: np.ndarray, implied_vols: np.ndarray
) -> np.ndarray:
    """
    Constructs Covariance using Historical Correlation + Implied Volatility.
    """
    # Calculate Historical Correlation
    # We use a context manager to silence division-by-zero warnings for
    # zero-variance assets, as we explicitly handle the resulting NaNs below.
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = np.corrcoef(log_returns, rowvar=False)

    # Handle NaN from zero-variance assets (constant prices)
    # NaN means no correlation structure, so we assume independence (correlation = 0)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Ensure diagonal is exactly 1.0 (sometimes numerical issues)
    np.fill_diagonal(corr_matrix, 1.0)

    # Apply Forward-Looking Volatility
    cov = implied_vols[:, None] * corr_matrix * implied_vols[None, :]
    return np.atleast_2d(cov)
