import numpy as np

from src.components.results_display import (
    _build_position_summary,
    _create_allocation_df,
    _create_results_csv,
)


def _sample_tangency():
    return {
        "weights": np.array([0.4, 0.6]),
        "expected_return": 0.12,
        "volatility": 0.20,
        "sharpe_ratio": 0.50,
        "cash_weight": 0.0,
        "tickers": ["A", "B"],
        "asset_returns": [0.08, 0.16],
        "asset_vols": [0.10, 0.30],
    }


def test_create_allocation_df_includes_borrowing_row():
    final_portfolio = {
        "weights": np.array([0.6, 0.9]),
        "cash_weight": -0.5,
    }

    alloc_df = _create_allocation_df(final_portfolio, _sample_tangency())

    borrowing_row = alloc_df[alloc_df["Asset"] == "Borrowing"].iloc[0]

    assert np.isclose(borrowing_row["Weight"], -0.5)


def test_create_results_csv_includes_borrowing_row():
    final_portfolio = {
        "weights": np.array([0.6, 0.9]),
        "expected_return": 0.155,
        "volatility": 0.30,
        "sharpe_ratio": 0.35,
        "cash_weight": -0.5,
    }

    csv_data = _create_results_csv(final_portfolio, _sample_tangency())

    assert "Borrowing,-0.5000,0.0000,0.0000" in csv_data


def test_build_position_summary_uses_borrowing_terms():
    summary = _build_position_summary(
        {
            "weights": np.array([0.6, 0.9]),
            "cash_weight": -0.5,
        }
    )

    assert "150.0% is invested in the tangency portfolio" in summary
    assert "50.0% is financed at the borrowing rate." in summary
