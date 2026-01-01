"""
Tests for results_display component logic.
"""

import pytest
import numpy as np
from src.stlite.components.results_display import (
    _safe_float,
    _create_allocation_df,
    _create_chart,
)

# --- Test Data Sanitization ---


def test_safe_float_basics():
    assert _safe_float(0.5) == 0.5
    assert _safe_float(1) == 1.0
    assert _safe_float("0.5") == 0.5
    assert _safe_float(None) == 0.0


def test_safe_float_numpy():
    assert _safe_float(np.float64(0.5)) == 0.5
    assert _safe_float(np.array(0.5)) == 0.5
    assert _safe_float(np.nan) == 0.0
    assert _safe_float(np.inf) == 0.0
    assert _safe_float(-np.inf) == 0.0


# --- Test Table Generation ---


def test_create_allocation_df_basic():
    final_pf = {"weights": [0.6, 0.4], "cash_weight": 0.0}
    context = {
        "tickers": ["A", "B"],
        "asset_returns": [0.1, 0.2],
        "asset_vols": [0.05, 0.1],
    }

    df = _create_allocation_df(final_pf, context)

    assert len(df) == 2
    assert "Asset" in df.columns
    assert "Weight" in df.columns
    assert df.iloc[0]["Asset"] == "A"
    assert df.iloc[0]["Weight"] == 0.6


def test_create_allocation_df_with_cash():
    final_pf = {"weights": [0.3, 0.2], "cash_weight": 0.5}
    context = {
        "tickers": ["A", "B"],
        "asset_returns": [0.1, 0.2],
        "asset_vols": [0.05, 0.1],
    }

    df = _create_allocation_df(final_pf, context)

    # 2 assets + 1 cash = 3 rows
    assert len(df) == 3
    # Sorted by weight desc? Cash 0.5 is max
    assert df.iloc[0]["Asset"] == "CASH"
    assert df.iloc[0]["Weight"] == 0.5


# --- Test Chart Generation ---


def test_create_chart_structure():
    """Verify chart is created with correct layers."""
    tangency = {
        "volatility": 0.2,
        "expected_return": 0.1,
        "tickers": ["A"],
        "asset_returns": [0.1],
        "asset_vols": [0.2],
    }
    final = {"volatility": 0.15, "expected_return": 0.075}
    cml_points = [
        {"volatility": 0.0, "expected_return": 0.02},
        {"volatility": 0.2, "expected_return": 0.1},
    ]
    rf = 0.02

    chart = _create_chart(tangency, final, cml_points, rf)

    # Convert to dict to verify structure
    spec = chart.to_dict()

    # Should have 5 layers: frontier, cml, assets, sharpe star, target diamond
    assert "layer" in spec
    assert len(spec["layer"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
