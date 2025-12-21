import yfinance as yf
import polars as pl
import pandas as pd
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class TickerDef:
    ticker: str
    implied_vol: float
    custom_return: float
    # alpha_delta removed as it's no longer required by the user
    constraint: Literal["Long", "Short", "Both"] = "Long"


def main():
    # 1. Configuration: Define your universe here using the TickerDef dataclass
    universe_definitions = [
        TickerDef(
            ticker=ticker,
            implied_vol=implied_vol,
            custom_return=custom_return,
            constraint=constraint,
        )
        for ticker, implied_vol, custom_return, constraint in [
            ("GOOG", 28.0, 8.0, "Long"),
            ("CROX", 35.0, 10.0, "Long"),
            ("NVO", 30.0, 8.0, "Long"),
            ("MPC", 26.0, 9.0, "Long"),
            ("VLO", 25.0, 9.0, "Long"),
            ("VIRT", 30.0, 8.0, "Long"),
            ("VTRS", 24.0, 7.0, "Long"),
            ("PFE", 21.0, 6.5, "Long"),
            ("ACGL", 26.0, 8.0, "Long"),
            ("DAC", 30.0, 8.0, "Long"),
            ("C", 24.0, 9.0, "Long"),
            ("COF", 30.0, 8.0, "Long"),
            ("BCC", 30.0, 9.0, "Long"),
            ("INTC", 35.0, 13.0, "Long"),
            ("SOFI", 50.0, 8.0, "Long"),
            ("PL", 30.0, 7.0, "Long"),
            ("VZ", 16.0, 6.0, "Long"),
            ("HPQ", 24.0, 7.5, "Long"),
        ]
    ]

    # Directories
    tmp_dir = "/home/bog/Documents/Efficient Portolio Construction/tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    universe_file = os.path.join(tmp_dir, "universe.csv")
    metrics_file = os.path.join(tmp_dir, "metrics.csv")

    # Extract tickers for fetching
    tickers = [t.ticker for t in universe_definitions]
    print(f"Fetching data for: {tickers}")

    # 2. Fetch Price History
    try:
        # User requested Open prices in previous turn
        data = yf.download(
            tickers, period="2y", interval="1wk", auto_adjust=True, progress=False
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Handle yfinance columns
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Using Open as per previous instruction
            prices = data["Open"]
        except KeyError:
            if "Open" in data.columns.levels[0]:
                prices = data.xs("Open", level=0, axis=1)
            else:
                prices = data
    else:
        prices = data["Open"] if "Open" in data.columns else data

    # Reset index to make Date a column
    prices = prices.reset_index()

    # Save Universe
    try:
        df_universe = pl.from_pandas(prices)
        print(f"Writing Universe to {universe_file}...")
        df_universe.write_csv(universe_file)
    except ImportError:
        print("Polars not found, using pandas to save.")
        prices.to_csv(universe_file, index=False)

    print("Universe generated successfully.")

    # 3. Generate Bulk Upload Metrics from Dataclass definitions

    metrics_data = {
        "Ticker": [t.ticker for t in universe_definitions],
        "Implied Volatility (%)": [t.implied_vol for t in universe_definitions],
        # "Alpha Delta (%)": REMOVED - data.py now handles missing this by defaulting to 0.0
        "Custom Return (%)": [t.custom_return for t in universe_definitions],
        "Constraint": [t.constraint for t in universe_definitions],
    }

    try:
        df_metrics = pl.DataFrame(metrics_data)
        print(f"Writing Metrics to {metrics_file}...")
        df_metrics.write_csv(metrics_file)
    except NameError:
        pd.DataFrame(metrics_data).to_csv(metrics_file, index=False)

    print("Metrics generated successfully.")


if __name__ == "__main__":
    main()
