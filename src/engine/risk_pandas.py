"""
Risk modeling: Covariance matrix construction (Pandas version).
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Optional


class RiskModel(Enum):
    HISTORICAL = "historical"
    FORWARD_LOOKING = "forward-looking"


def calculate_covariance(
    prices: pd.DataFrame,
    risk_model: RiskModel,
    annualization_factor: Optional[int] = None,
    implied_vols: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate covariance matrix for portfolio optimization.

    Pandas version compatible with Pyodide/stlite.
    """
    # 1. Prepare Data
    ticker_cols = [c for c in prices.columns if c != "date"]
    price_matrix = prices[ticker_cols].values

    # Compute Log Returns
    log_returns = np.diff(np.log(price_matrix), axis=0)

    # 2. Dispatch Logic
    if risk_model is RiskModel.HISTORICAL:
        if annualization_factor is None:
            raise ValueError(
                "annualization_factor is required for HISTORICAL risk model"
            )

        cov = np.cov(log_returns, rowvar=False) * annualization_factor
        return np.atleast_2d(cov)

    elif risk_model is RiskModel.FORWARD_LOOKING:
        if implied_vols is None:
            raise ValueError("implied_vols is required for FORWARD_LOOKING risk model")

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
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = np.corrcoef(log_returns, rowvar=False)

    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    # Apply Forward-Looking Volatility
    cov = implied_vols[:, None] * corr_matrix * implied_vols[None, :]
    return np.atleast_2d(cov)
