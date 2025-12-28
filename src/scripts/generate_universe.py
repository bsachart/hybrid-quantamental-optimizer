"""
Universe Generator - Fetch market data and create portfolio inputs.

Philosophy (Ousterhout):
    - Deep Module: Simple interface, complex implementation hidden.
    - Information Hiding: CAGR calculation details are internal.
    - Define Errors Out of Existence: Validate early, fail fast.
    - Interface Compliance: Output matches src/engine/data_loader.py expectations.

Usage:
    python src/scripts/generate_universe.py
"""

import yfinance as yf
import polars as pl
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass


# ==============================================================================
# Constants
# ==============================================================================

# Assumption: 50% of Net Profit is reinvested into the business to drive future growth.
REINVESTMENT_RATE = 0.5


# ==============================================================================
# Public API
# ==============================================================================


@dataclass
class AssetDefinition:
    """
    Configuration for a single asset.

    Two modes for expected return:
    1. Explicit: Set expected_return directly (decimal, e.g., 0.12 = 12%).
    2. Fundamental: Provide fundamentals to calculate CAGR.
    """

    ticker: str
    implied_volatility: float  # Annualized (decimal, e.g., 0.25 = 25%)

    # Optional constraints (overrides global defaults)
    min_weight: Optional[float] = None
    max_weight: Optional[float] = None

    # Mode 1: Explicit Return
    expected_return: Optional[float] = None

    # Mode 2: Fundamental Inputs (all required if expected_return is None)
    market_cap: Optional[float] = None  # Billions
    current_sales: Optional[float] = None  # Billions
    current_npm: Optional[float] = None  # Net Profit Margin (decimal)
    organic_growth: Optional[float] = None  # Annual revenue growth (decimal)
    terminal_npm: Optional[float] = None  # Target margin at projection end
    exit_pe: Optional[float] = None  # Terminal P/E multiple
    n_years: int = 5  # Projection horizon


def generate_universe(
    assets: List[AssetDefinition],
    output_dir: str = "tmp",
    period: str = "2y",
    interval: str = "1wk",
    default_min_weight: float = 0.0,
    default_max_weight: float = 1.0,
    fetch_prices: bool = True,
    auto_fetch_market_caps: bool = True,
) -> None:
    """
    Generate portfolio universe files matching src/engine expectations.

    Creates two files:
    - universe.csv: Historical prices (date + ticker columns).
    - metrics.csv: Asset metrics (ticker, expected_return, implied_volatility, weights).

    Args:
        assets: List of asset configurations.
        output_dir: Output directory for CSV files.
        period: yfinance period (e.g., "1y", "2y", "5y").
        interval: yfinance interval (e.g., "1d", "1wk", "1mo").
        default_min_weight: Default minimum weight constraint.
        default_max_weight: Default maximum weight constraint.
        fetch_prices: Whether to download price history.
        auto_fetch_market_caps: Whether to auto-fetch missing market caps.

    Raises:
        ValueError: If assets list is empty or validation fails.
    """
    if not assets:
        raise ValueError("Asset list cannot be empty")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Auto-fetch missing market caps if enabled
    if auto_fetch_market_caps:
        _fill_missing_market_caps(assets)

    # Step 2: Fetch price history
    if fetch_prices:
        tickers = [a.ticker for a in assets]
        print(f"--> Fetching price data for {len(tickers)} assets...")
        prices_df = _fetch_historical_prices(tickers, period, interval)

        prices_path = output_path / "universe.csv"
        prices_df.write_csv(prices_path)
        print(f"    Prices saved: {prices_path} ({prices_df.height} rows)")
    else:
        print("--> Skipping price fetch (fetch_prices=False)")

    # Step 3: Build metrics
    print(f"--> Computing metrics...")
    metrics_df = _build_metrics(assets, default_min_weight, default_max_weight)

    metrics_path = output_path / "metrics.csv"
    metrics_df.write_csv(metrics_path)
    print(f"    Metrics saved: {metrics_path}")

    # Display preview
    print("\n=== Asset Metrics Preview ===")
    print(metrics_df.select(["ticker", "expected_return", "implied_volatility"]))


# ==============================================================================
# Internal Implementation (Information Hiding)
# ==============================================================================


def _fill_missing_market_caps(assets: List[AssetDefinition]) -> None:
    """
    Fetch missing market caps for assets using fundamental mode.
    Mutates AssetDefinition objects in place.
    """
    # Identify assets that need fundamentals but are missing market cap
    missing_caps = [
        a for a in assets if a.expected_return is None and a.market_cap is None
    ]

    if not missing_caps:
        return

    tickers = [a.ticker for a in missing_caps]
    print(f"--> Fetching missing market caps for: {', '.join(tickers)}...")

    fetched = _fetch_market_caps_batch(tickers)

    for asset in missing_caps:
        if asset.ticker in fetched:
            asset.market_cap = fetched[asset.ticker]
            print(f"    ✓ {asset.ticker}: ${asset.market_cap:.2f}B")
        else:
            print(f"    ✗ {asset.ticker}: Failed to fetch market cap")


def _fetch_market_caps_batch(tickers: List[str]) -> Dict[str, float]:
    """Fetch market caps in billions for multiple tickers."""
    results = {}

    try:
        # yfinance.Tickers is more efficient than looping .Ticker()
        tickers_obj = yf.Tickers(" ".join(tickers))
        for ticker in tickers:
            try:
                # Accessing .info triggers the specific API call for that ticker
                info = tickers_obj.tickers[ticker].info
                mc = info.get("marketCap")
                if mc and mc > 0:
                    results[ticker] = mc / 1e9  # Convert to billions
            except Exception:
                continue
    except Exception as e:
        print(f"    Warning: Batch fetch error - {e}")

    return results


def _fetch_historical_prices(
    tickers: List[str], period: str, interval: str
) -> pl.DataFrame:
    """
    Fetch historical adjusted close prices from Yahoo Finance.

    Returns:
        Polars DataFrame with 'date' column + ticker columns.
    """
    try:
        # Fetch data (auto_adjust handles splits/dividends)
        data = yf.download(
            tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        if data.empty:
            raise ValueError("No data returned from yfinance")

        # Extract Close prices for each ticker
        price_data = {}

        if len(tickers) == 1:
            # Single ticker: simpler DataFrame structure
            ticker = tickers[0]
            if "Close" in data.columns:
                price_data[ticker] = data["Close"]
            else:
                # Fallback to first column if structure is unexpected
                price_data[ticker] = data.iloc[:, 0]
        else:
            # Multiple tickers: extract from MultiIndex columns
            for ticker in tickers:
                try:
                    price_data[ticker] = data[ticker]["Close"]
                except (KeyError, TypeError):
                    print(f"    Warning: No data found for {ticker}")

        if not price_data:
            raise ValueError("Failed to extract any price data")

        # Convert to Polars DataFrame
        df_pd = pd.DataFrame(price_data)
        df_pd.index.name = "date"

        df_pl = pl.from_pandas(df_pd.reset_index()).with_columns(
            pl.col("date").cast(pl.Date)
        )

        return df_pl.sort("date")

    except Exception as e:
        raise ValueError(f"Price fetch failed: {e}")


def _build_metrics(
    assets: List[AssetDefinition],
    default_min_weight: float,
    default_max_weight: float,
) -> pl.DataFrame:
    """
    Build metrics DataFrame matching src/engine expectations.

    Schema: ticker, expected_return, implied_volatility, min_weight, max_weight
    """
    rows = []

    for asset in assets:
        # Determine expected return
        if asset.expected_return is not None:
            expected_return = asset.expected_return
        else:
            # Fundamental mode - validate and calculate
            if asset.market_cap is None:
                raise ValueError(
                    f"Asset '{asset.ticker}' missing market_cap. "
                    f"Provide manually or enable auto_fetch_market_caps=True"
                )
            expected_return = _calculate_fundamental_cagr(asset)

        # Resolve weight constraints
        min_w = asset.min_weight if asset.min_weight is not None else default_min_weight
        max_w = asset.max_weight if asset.max_weight is not None else default_max_weight

        rows.append(
            {
                "ticker": asset.ticker,
                "expected_return": expected_return,
                "implied_volatility": asset.implied_volatility,
                "min_weight": min_w,
                "max_weight": max_w,
            }
        )

    df = pl.DataFrame(rows)

    # Validate constraints (Fail Fast)
    if (df["implied_volatility"] <= 0).any():
        raise ValueError("All implied volatilities must be positive")

    if (df["min_weight"] > df["max_weight"]).any():
        raise ValueError("min_weight cannot exceed max_weight")

    return df


def _calculate_fundamental_cagr(asset: AssetDefinition) -> float:
    """
    Calculate CAGR from fundamental inputs.

    Logic:
    1. Project sales growth over n_years using organic growth + reinvestment.
    2. Linear interpolation of NPM from current to terminal.
    3. Calculate terminal value using exit P/E.
    4. Compute CAGR from current market cap to terminal value.

    Returns:
        Annualized CAGR (decimal).
    """
    # Validate all required fields are present
    required = {
        "current_sales": asset.current_sales,
        "current_npm": asset.current_npm,
        "organic_growth": asset.organic_growth,
        "terminal_npm": asset.terminal_npm,
        "exit_pe": asset.exit_pe,
    }

    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"Asset '{asset.ticker}' missing fundamental inputs: {missing}"
        )

    if asset.market_cap is None or asset.market_cap <= 0:
        raise ValueError(
            f"Asset '{asset.ticker}' has invalid market cap: {asset.market_cap}"
        )

    # Project sales with NPM interpolation
    sales = asset.current_sales
    for year in range(1, asset.n_years + 1):
        # Linear interpolation of NPM
        progress = year / asset.n_years
        current_npm = (
            asset.current_npm + (asset.terminal_npm - asset.current_npm) * progress
        )

        # Growth = Organic + Reinvestment
        # We assume 50% of profits are reinvested to fuel growth
        growth = asset.organic_growth + (current_npm * REINVESTMENT_RATE)
        sales *= 1 + growth

    # Terminal value calculation
    terminal_earnings = sales * asset.terminal_npm
    terminal_value = terminal_earnings * asset.exit_pe

    # CAGR calculation
    # Floor ratio at 0.01 to avoid complex numbers or negative bases with fractional powers
    ratio = max(terminal_value / asset.market_cap, 0.01)
    cagr = ratio ** (1 / asset.n_years) - 1

    return cagr


# ==============================================================================
# Example Usage
# ==============================================================================


if __name__ == "__main__":
    universe = [
        # Case A: Fundamental analysis with manual market cap
        AssetDefinition(
            ticker="GOOG",
            implied_volatility=0.25,
            market_cap=2000.0,
            current_sales=300.0,
            current_npm=0.25,
            organic_growth=0.10,
            terminal_npm=0.25,
            exit_pe=20.0,
        ),
        # Case B: Fundamental analysis with auto-fetch market cap
        AssetDefinition(
            ticker="NVO",
            implied_volatility=0.30,
            current_sales=42.0,
            current_npm=0.35,
            organic_growth=0.15,
            terminal_npm=0.35,
            exit_pe=20.0,
        ),
        # Case C: Explicit expected return
        AssetDefinition(
            ticker="SPY",
            implied_volatility=0.15,
            expected_return=0.08,
        ),
    ]

    generate_universe(
        assets=universe,
        output_dir="tmp",
        fetch_prices=True,
        auto_fetch_market_caps=True,
    )
