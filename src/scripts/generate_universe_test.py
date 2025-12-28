"""
Tests for generate_universe.py

Philosophy:
    - Test the public interface, not implementation details.
    - Cover both happy paths and error cases.
    - Mock external dependencies (yfinance).
"""

import pytest
import polars as pl
import pandas as pd
from unittest.mock import patch, MagicMock
from src.scripts.generate_universe import (
    generate_universe,
    AssetDefinition,
    _calculate_fundamental_cagr,
    _build_metrics,
    REINVESTMENT_RATE,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_yfinance_data():
    """Mock successful yfinance download response."""
    dates = pd.date_range("2023-01-01", periods=10, freq="W")
    data = pd.DataFrame(
        {
            ("AAPL", "Close"): [150.0] * 10,
            ("GOOG", "Close"): [100.0] * 10,
        },
        index=dates,
    )
    data.columns = pd.MultiIndex.from_tuples(data.columns)
    return data


@pytest.fixture
def mock_market_cap_info():
    """Mock yfinance market cap info."""
    return {
        "AAPL": {"marketCap": 3000000000000},  # $3T
        "GOOG": {"marketCap": 2000000000000},  # $2T
    }


@pytest.fixture
def temp_output(tmp_path):
    """Temporary output directory."""
    return tmp_path / "universe_test"


# ==============================================================================
# Unit Tests: CAGR Calculation
# ==============================================================================


def test_fundamental_cagr_calculation():
    """Test CAGR calculation logic matches expected math."""
    asset = AssetDefinition(
        ticker="TEST",
        implied_volatility=0.2,
        market_cap=100.0,
        current_sales=50.0,
        current_npm=0.20,
        organic_growth=0.10,
        terminal_npm=0.20,
        exit_pe=20.0,
        n_years=1,
    )

    # Manual verification:
    # 1. Growth Rate = Organic + (NPM * Reinvestment)
    #    Growth = 0.10 + (0.20 * 0.5) = 0.20 (20%)
    # 2. Projected Sales = Current Sales * (1 + Growth)
    #    Sales = 50.0 * 1.20 = 60.0
    # 3. Terminal Earnings = Sales * Terminal NPM
    #    Earnings = 60.0 * 0.20 = 12.0
    # 4. Terminal Value = Earnings * PE
    #    Value = 12.0 * 20.0 = 240.0
    # 5. CAGR = (Terminal Value / Market Cap)^(1/n) - 1
    #    CAGR = (240 / 100)^1 - 1 = 2.4 - 1 = 1.4 (140%)

    expected_cagr = 1.4
    result = _calculate_fundamental_cagr(asset)

    assert abs(result - expected_cagr) < 0.0001


def test_fundamental_cagr_npm_interpolation():
    """Test that NPM interpolation logic functions without error."""
    asset = AssetDefinition(
        ticker="TEST",
        implied_volatility=0.2,
        market_cap=100.0,
        current_sales=50.0,
        current_npm=0.10,
        organic_growth=0.0,
        terminal_npm=0.30,  # NPM increases over time
        exit_pe=20.0,
        n_years=2,
    )

    result = _calculate_fundamental_cagr(asset)
    # Since margins improve, we expect positive growth even with 0% organic
    assert result > 0


def test_fundamental_cagr_missing_inputs():
    """Test that missing fundamental inputs raise ValueError."""
    asset = AssetDefinition(
        ticker="TEST",
        implied_volatility=0.2,
        market_cap=100.0,
        current_sales=None,  # Missing
        current_npm=0.10,
        organic_growth=0.10,
        terminal_npm=0.15,
        exit_pe=20.0,
    )

    with pytest.raises(ValueError, match="missing fundamental inputs"):
        _calculate_fundamental_cagr(asset)


def test_fundamental_cagr_invalid_market_cap():
    """Test that invalid market cap raises ValueError."""
    asset = AssetDefinition(
        ticker="TEST",
        implied_volatility=0.2,
        market_cap=0.0,  # Invalid
        current_sales=50.0,
        current_npm=0.10,
        organic_growth=0.10,
        terminal_npm=0.15,
        exit_pe=20.0,
    )

    with pytest.raises(ValueError, match="invalid market cap"):
        _calculate_fundamental_cagr(asset)


# ==============================================================================
# Unit Tests: Metrics Building
# ==============================================================================


def test_build_metrics_explicit_returns():
    """Test metrics building with explicit returns."""
    assets = [
        AssetDefinition(
            ticker="AAPL",
            implied_volatility=0.25,
            expected_return=0.12,
            min_weight=0.0,
            max_weight=0.5,
        ),
        AssetDefinition(
            ticker="GOOG",
            implied_volatility=0.30,
            expected_return=0.15,
        ),
    ]

    df = _build_metrics(assets, default_min_weight=0.0, default_max_weight=1.0)

    assert df.shape == (2, 5)
    assert df["ticker"].to_list() == ["AAPL", "GOOG"]
    assert df["expected_return"].to_list() == [0.12, 0.15]
    assert df["min_weight"].to_list() == [0.0, 0.0]
    assert df["max_weight"].to_list() == [0.5, 1.0]


def test_build_metrics_validation_errors():
    """Test that validation errors are raised."""
    # Test negative volatility
    assets = [
        AssetDefinition(ticker="BAD", implied_volatility=-0.1, expected_return=0.1)
    ]
    with pytest.raises(ValueError, match="must be positive"):
        _build_metrics(assets, 0.0, 1.0)

    # Test min > max weight
    assets = [
        AssetDefinition(
            ticker="BAD",
            implied_volatility=0.2,
            expected_return=0.1,
            min_weight=0.6,
            max_weight=0.4,
        )
    ]
    with pytest.raises(ValueError, match="cannot exceed"):
        _build_metrics(assets, 0.0, 1.0)


# ==============================================================================
# Integration Tests: Full Pipeline
# ==============================================================================


@patch("src.scripts.generate_universe.yf.download")
def test_generate_universe_explicit_returns(
    mock_download, temp_output, mock_yfinance_data
):
    """Test full pipeline with explicit returns."""
    mock_download.return_value = mock_yfinance_data

    assets = [
        AssetDefinition(ticker="AAPL", implied_volatility=0.25, expected_return=0.12),
        AssetDefinition(ticker="GOOG", implied_volatility=0.30, expected_return=0.15),
    ]

    generate_universe(
        assets=assets,
        output_dir=str(temp_output),
        fetch_prices=True,
        auto_fetch_market_caps=False,
    )

    # Check files exist
    assert (temp_output / "universe.csv").exists()
    assert (temp_output / "metrics.csv").exists()

    # Validate output schema
    prices = pl.read_csv(temp_output / "universe.csv", try_parse_dates=True)
    assert "date" in prices.columns
    assert "AAPL" in prices.columns


@patch("src.scripts.generate_universe.yf.Tickers")
@patch("src.scripts.generate_universe.yf.download")
def test_generate_universe_auto_fetch_market_cap(
    mock_download, mock_tickers, temp_output, mock_yfinance_data, mock_market_cap_info
):
    """Test auto-fetching market caps logic."""
    mock_download.return_value = mock_yfinance_data

    # Mock Tickers object structure
    mock_ticker_objs = {}
    for ticker, info in mock_market_cap_info.items():
        mock_obj = MagicMock()
        mock_obj.info = info
        mock_ticker_objs[ticker] = mock_obj

    mock_tickers_instance = MagicMock()
    mock_tickers_instance.tickers = mock_ticker_objs
    mock_tickers.return_value = mock_tickers_instance

    assets = [
        AssetDefinition(
            ticker="AAPL",
            implied_volatility=0.25,
            # Missing market_cap, but fundamentals present
            current_sales=400.0,
            current_npm=0.25,
            organic_growth=0.10,
            terminal_npm=0.30,
            exit_pe=25.0,
        ),
    ]

    generate_universe(
        assets=assets,
        output_dir=str(temp_output),
        fetch_prices=True,
        auto_fetch_market_caps=True,
    )

    # Check that market cap was filled in the object
    assert assets[0].market_cap == 3000.0  # $3T / 1e9


def test_generate_universe_skip_prices(temp_output):
    """Test skipping price fetch."""
    assets = [
        AssetDefinition(ticker="AAPL", implied_volatility=0.25, expected_return=0.12)
    ]

    generate_universe(
        assets=assets,
        output_dir=str(temp_output),
        fetch_prices=False,
        auto_fetch_market_caps=False,
    )

    # Prices file should NOT exist
    assert not (temp_output / "universe.csv").exists()
    # Metrics file SHOULD exist
    assert (temp_output / "metrics.csv").exists()


def test_generate_universe_missing_market_cap_no_fetch(temp_output):
    """Test that missing market cap without fetch enabled raises error."""
    assets = [
        AssetDefinition(
            ticker="TEST",
            implied_volatility=0.25,
            # Missing market_cap and explicit return
            current_sales=100.0,
            current_npm=0.20,
            organic_growth=0.10,
            terminal_npm=0.25,
            exit_pe=20.0,
        )
    ]

    with pytest.raises(ValueError, match="missing market_cap"):
        generate_universe(
            assets=assets,
            output_dir=str(temp_output),
            fetch_prices=False,
            auto_fetch_market_caps=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
