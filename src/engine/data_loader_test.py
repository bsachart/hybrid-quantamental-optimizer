"""
Tests for data_loader module.

Test Philosophy (Ousterhout):
    - Test the interface, not the implementation
    - Cover common use cases and edge cases
    - Make test failures informative
"""

import pytest
import polars as pl
from io import StringIO
from src.engine.data_loader import (
    load_universe,
    Universe,
    DataValidationError,
)


# ==============================================================================
# Fixtures - Test Data
# ==============================================================================


@pytest.fixture
def valid_prices_csv():
    """Valid price data as CSV string."""
    return """date,AAPL,GOOG,MSFT
2023-01-31,150.23,105.44,280.50
2023-02-28,152.11,108.22,285.33
2023-03-31,155.00,110.00,290.00
2023-04-30,157.50,112.50,295.00
2023-05-31,160.00,115.00,300.00
2023-06-30,162.50,117.50,305.00
2023-07-31,165.00,120.00,310.00
2023-08-31,167.50,122.50,315.00
2023-09-30,170.00,125.00,320.00
2023-10-31,172.50,127.50,325.00
"""


@pytest.fixture
def valid_metrics_csv():
    """Valid metrics data as CSV string."""
    return """ticker,expected_return,implied_volatility,min_weight,max_weight
AAPL,0.12,0.25,0.0,1.0
GOOG,0.15,0.28,0.0,1.0
MSFT,0.10,0.20,0.0,0.5
"""


@pytest.fixture
def minimal_metrics_csv():
    """Metrics without optional weight columns."""
    return """ticker,expected_return,implied_volatility
AAPL,0.12,0.25
GOOG,0.15,0.28
MSFT,0.10,0.20
"""


# ==============================================================================
# Happy Path Tests
# ==============================================================================


def test_load_universe_basic(valid_prices_csv, valid_metrics_csv):
    """Test basic successful load."""
    universe = load_universe(StringIO(valid_prices_csv), StringIO(valid_metrics_csv))

    assert isinstance(universe, Universe)
    assert universe.tickers == ["AAPL", "GOOG", "MSFT"]
    assert universe.prices.shape == (10, 4)  # 10 rows, date + 3 tickers
    assert universe.metrics.shape == (3, 5)  # 3 tickers, 5 columns
    assert "date" in universe.prices.columns


def test_load_universe_with_defaults(valid_prices_csv, minimal_metrics_csv):
    """Test that min_weight and max_weight get defaults."""
    universe = load_universe(StringIO(valid_prices_csv), StringIO(minimal_metrics_csv))

    assert "min_weight" in universe.metrics.columns
    assert "max_weight" in universe.metrics.columns
    assert (universe.metrics["min_weight"] == 0.0).all()
    assert (universe.metrics["max_weight"] == 1.0).all()


def test_load_universe_partial_overlap(valid_prices_csv):
    """Test that only common tickers are included."""
    metrics_partial = """ticker,expected_return,implied_volatility
AAPL,0.12,0.25
TSLA,0.20,0.40
"""

    universe = load_universe(StringIO(valid_prices_csv), StringIO(metrics_partial))

    assert universe.tickers == ["AAPL"]
    assert universe.prices.shape[1] == 2  # date + AAPL
    assert universe.metrics.shape[0] == 1  # Only AAPL


def test_data_alignment_order():
    """Test that data is properly sorted and aligned."""
    prices = """date,B,A,C
2023-02-28,2.0,1.0,3.0
2023-01-31,2.1,1.1,3.1
2023-03-31,2.2,1.2,3.2
2023-05-31,2.3,1.3,3.3
2023-04-30,2.4,1.4,3.4
2023-06-30,2.5,1.5,3.5
2023-07-31,2.6,1.6,3.6
2023-08-31,2.7,1.7,3.7
2023-09-30,2.8,1.8,3.8
2023-10-31,2.9,1.9,3.9
"""

    metrics = """ticker,expected_return,implied_volatility
C,0.10,0.20
A,0.12,0.25
B,0.15,0.28
"""

    universe = load_universe(StringIO(prices), StringIO(metrics))

    # Verify sorting
    assert universe.tickers == ["A", "B", "C"]
    dates = universe.prices["date"].to_list()
    assert dates == sorted(dates)
    assert universe.metrics["ticker"].to_list() == ["A", "B", "C"]


# ==============================================================================
# Error Cases - Prices
# ==============================================================================


def test_prices_missing_date_column(valid_metrics_csv):
    """Test error when price data doesn't start with date."""
    bad_prices = """ticker,AAPL,GOOG
2023-01-31,150.23,105.44
"""

    with pytest.raises(DataValidationError, match="must start with 'date'"):
        load_universe(StringIO(bad_prices), StringIO(valid_metrics_csv))


def test_prices_empty():
    """Test error when price data is empty."""
    with pytest.raises(DataValidationError):
        load_universe(
            StringIO(""), StringIO("ticker,expected_return,implied_volatility\n")
        )


def test_prices_unparseable_dates(valid_metrics_csv):
    """Test error when dates can't be parsed."""
    bad_prices = """date,AAPL
not-a-date,150.23
"""

    with pytest.raises(Exception):
        load_universe(StringIO(bad_prices), StringIO(valid_metrics_csv))


# ==============================================================================
# Error Cases - Metrics
# ==============================================================================


def test_metrics_missing_required_columns(valid_prices_csv):
    """Test error when required columns are missing."""
    # Missing ticker
    with pytest.raises(DataValidationError, match="missing required columns"):
        bad = """symbol,expected_return,implied_volatility
AAPL,0.12,0.25
"""
        load_universe(StringIO(valid_prices_csv), StringIO(bad))

    # Missing expected_return
    with pytest.raises(DataValidationError, match="missing required columns"):
        bad = """ticker,implied_volatility
AAPL,0.25
"""
        load_universe(StringIO(valid_prices_csv), StringIO(bad))

    # Missing implied_volatility
    with pytest.raises(DataValidationError, match="missing required columns"):
        bad = """ticker,expected_return
AAPL,0.12
"""
        load_universe(StringIO(valid_prices_csv), StringIO(bad))


def test_metrics_invalid_volatility(valid_prices_csv):
    """Test error when implied volatility is non-positive."""
    negative = """ticker,expected_return,implied_volatility
AAPL,0.12,-0.25
"""
    zero = """ticker,expected_return,implied_volatility
AAPL,0.12,0.0
"""

    with pytest.raises(DataValidationError, match="must be positive"):
        load_universe(StringIO(valid_prices_csv), StringIO(negative))

    with pytest.raises(DataValidationError, match="must be positive"):
        load_universe(StringIO(valid_prices_csv), StringIO(zero))


def test_metrics_invalid_weight_bounds(valid_prices_csv):
    """Test error when min_weight > max_weight."""
    bad_metrics = """ticker,expected_return,implied_volatility,min_weight,max_weight
AAPL,0.12,0.25,0.5,0.3
"""

    with pytest.raises(DataValidationError, match="min_weight > max_weight"):
        load_universe(StringIO(valid_prices_csv), StringIO(bad_metrics))


# ==============================================================================
# Edge Cases
# ==============================================================================


def test_no_common_tickers():
    """Test error when price and metric tickers don't overlap."""
    prices = """date,AAPL,GOOG
2023-01-31,150.23,105.44
2023-02-28,152.11,108.22
2023-03-31,155.00,110.00
2023-04-30,157.50,112.50
2023-05-31,160.00,115.00
2023-06-30,162.50,117.50
2023-07-31,165.00,120.00
2023-08-31,167.50,122.50
2023-09-30,170.00,125.00
2023-10-31,172.50,127.50
"""

    metrics = """ticker,expected_return,implied_volatility
TSLA,0.20,0.40
NVDA,0.25,0.45
"""

    with pytest.raises(DataValidationError, match="No overlapping tickers"):
        load_universe(StringIO(prices), StringIO(metrics))


def test_single_ticker():
    """Test that single ticker works."""
    prices = """date,AAPL
2023-01-31,150.23
2023-02-28,152.11
2023-03-31,155.00
2023-04-30,157.50
2023-05-31,160.00
2023-06-30,162.50
2023-07-31,165.00
2023-08-31,167.50
2023-09-30,170.00
2023-10-31,172.50
"""

    metrics = """ticker,expected_return,implied_volatility
AAPL,0.12,0.25
"""

    universe = load_universe(StringIO(prices), StringIO(metrics))

    assert universe.tickers == ["AAPL"]
    assert universe.prices.shape == (10, 2)  # date + AAPL


def test_short_selling_constraints():
    """Test that negative min_weight (short selling) works."""
    prices = """date,AAPL
2023-01-31,150.23
2023-02-28,152.11
2023-03-31,155.00
2023-04-30,157.50
2023-05-31,160.00
2023-06-30,162.50
2023-07-31,165.00
2023-08-31,167.50
2023-09-30,170.00
2023-10-31,172.50
"""

    metrics = """ticker,expected_return,implied_volatility,min_weight,max_weight
AAPL,0.12,0.25,-0.5,1.0
"""

    universe = load_universe(StringIO(prices), StringIO(metrics))

    assert universe.metrics["min_weight"][0] == -0.5
    assert universe.metrics["max_weight"][0] == 1.0


def test_ticker_consistency():
    """Test that tickers list matches actual columns/rows."""
    prices = """date,A,B,C
2023-01-31,1,2,3
2023-02-28,1,2,3
2023-03-31,1,2,3
2023-04-30,1,2,3
2023-05-31,1,2,3
2023-06-30,1,2,3
2023-07-31,1,2,3
2023-08-31,1,2,3
2023-09-30,1,2,3
2023-10-31,1,2,3
"""

    metrics = """ticker,expected_return,implied_volatility
A,0.12,0.25
B,0.15,0.28
C,0.10,0.20
"""

    universe = load_universe(StringIO(prices), StringIO(metrics))

    # Verify alignment
    assert universe.tickers == [c for c in universe.prices.columns if c != "date"]
    assert universe.tickers == universe.metrics["ticker"].to_list()


def test_universe_immutability():
    """Test that Universe is immutable (frozen dataclass)."""
    prices = """date,AAPL
2023-01-31,150.23
2023-02-28,152.11
2023-03-31,155.00
2023-04-30,157.50
2023-05-31,160.00
2023-06-30,162.50
2023-07-31,165.00
2023-08-31,167.50
2023-09-30,170.00
2023-10-31,172.50
"""

    metrics = """ticker,expected_return,implied_volatility
AAPL,0.12,0.25
"""

    universe = load_universe(StringIO(prices), StringIO(metrics))

    with pytest.raises(Exception):  # FrozenInstanceError
        universe.tickers = ["MSFT"]
