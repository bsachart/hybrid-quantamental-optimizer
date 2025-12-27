"""
Centralized session state management for Streamlit app.

Philosophy:
    - Single source of truth for session state
    - Clear initialization and defaults
    - Type-safe accessors
"""

from typing import Optional, List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np


# Constants
DEFAULT_CONSTRAINT = "Long"
DEFAULT_IMPLIED_VOL = 30.0
DEFAULT_CUSTOM_RETURN = 8.0


def initialize_session_state():
    """
    Initialize all session state variables with defaults.
    Call this at the start of the Streamlit app.
    """
    defaults = {
        "prices": None,
        "metrics": None,
        "opt_results": None,
        "price_file_key": None,
        "metrics_file_key": None,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def create_default_metrics(tickers: List[str]) -> pd.DataFrame:
    """
    Create default metrics dataframe for a list of tickers.

    Args:
        tickers: List of ticker symbols

    Returns:
        DataFrame with default values for all metrics
    """
    return pd.DataFrame(
        {
            "Constraint": [DEFAULT_CONSTRAINT] * len(tickers),
            "Implied Volatility (%)": [DEFAULT_IMPLIED_VOL] * len(tickers),
            "Custom Return (%)": [DEFAULT_CUSTOM_RETURN] * len(tickers),
        },
        index=tickers,
    )


def save_optimization_results(
    tangency_return: float,
    tangency_vol: float,
    tangency_sharpe: float,
    tangency_weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf_rate: float,
    bounds: List[tuple],
    tickers: List[str],
    target_volatility: Optional[float] = None,
    frontier_points: Optional[List[Dict[str, Any]]] = None,
    random_portfolios: Optional[List[Dict[str, Any]]] = None,
):
    """
    Save optimization results to session state.

    Stores only serializable data, not objects.
    """
    st.session_state["opt_results"] = {
        "tangency_return": tangency_return,
        "tangency_vol": tangency_vol,
        "tangency_sharpe": tangency_sharpe,
        "tangency_weights": tangency_weights.tolist(),
        "expected_returns": expected_returns.tolist(),
        "cov_matrix": cov_matrix.tolist(),
        "rf_rate": rf_rate,
        "bounds": bounds,
        "tickers": tickers,
        "target_volatility": target_volatility,
        "frontier_points": frontier_points,
        "random_portfolios": random_portfolios,
    }


def get_optimization_results() -> Optional[Dict[str, Any]]:
    """
    Safely retrieve optimization results from session state.

    Returns:
        Optimization results dict or None if not available
    """
    return st.session_state.get("opt_results")


def clear_optimization_results():
    """Clear optimization results from session state."""
    if "opt_results" in st.session_state:
        del st.session_state["opt_results"]


def should_reoptimize() -> bool:
    """
    Check if optimization should be re-run based on state changes.

    This is a placeholder for future enhancement. Currently returns False.
    In a full implementation, you'd track input changes and invalidate results.
    """
    # Future: track hash of inputs and compare
    return False
