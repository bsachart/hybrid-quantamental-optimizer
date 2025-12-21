"""
Module for generating Altair charts for the Portfolio Optimizer.

Philosophy:
    - "Deep Module": Hides the complexity of Altair configuration behind a simple interface.
    - "Information Hiding": The main app doesn't need to know about mark_circle, encode, etc.
    - "Define Errors Out of Existence": Accept raw data types and handle formatting internally.
"""

import altair as alt
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import numpy.typing as npt


def _format_top_holdings(weights: npt.NDArray[np.float64], tickers: List[str]) -> str:
    """Helper to format top holdings string."""
    sorted_idx = np.argsort(weights)[::-1]
    top_3 = sorted_idx[:3]
    
    holdings = []
    for i in top_3:
        if weights[i] > 0.01:
            if i < len(tickers):
                name = tickers[i]
            # Handle case where weights includes Cash at the end
            elif i == len(tickers):
                name = "CASH"
            else:
                continue
            holdings.append(f"{name}: {weights[i]:.0%}")
            
    return ", ".join(holdings)


def _create_base_chart_config() -> Dict[str, Any]:
    """Returns common chart configuration."""
    return {
        "axis_config": {
            "titleFontSize": 14,
            "labelFontSize": 12,
            "titlePadding": 10
        },
        "legend_domain": [
            "Efficient Frontier",
            "Assets",
            "Max Sharpe",
            "Feasible Set",
            "Capital Market Line",
            "Target Portfolio",
        ],
        "legend_range": ["#00E676", "#00BCD4", "#FF5252", "gray", "#FFC107", "#FFEB3B"],
    }


def _create_cloud_layer(random_df: pd.DataFrame, config: Dict) -> alt.Chart:
    """Creates the feasible set cloud layer."""
    return (
        alt.Chart(random_df)
        .mark_circle(size=30, opacity=0.15)
        .encode(
            x=alt.X(
                "Volatility:Q",
                scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(format="%", title="Volatility (Risk)", **config["axis_config"]),
            ),
            y=alt.Y(
                "Return:Q",
                scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(format="%", title="Expected Return", **config["axis_config"]),
            ),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(domain=config["legend_domain"], range=config["legend_range"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Return:Q", format=".2%"),
                alt.Tooltip("Volatility:Q", format=".2%"),
                alt.Tooltip("Top Holdings:N"),
            ],
        )
        .transform_calculate(Type="'Feasible Set'")
    )


def _create_frontier_layer(frontier_df: pd.DataFrame, config: Dict) -> alt.Chart:
    """Creates the efficient frontier line layer."""
    return (
        alt.Chart(frontier_df)
        .mark_line(size=5)
        .encode(
            x="Volatility:Q",
            y="Return:Q",
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(domain=config["legend_domain"], range=config["legend_range"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Return:Q", format=".2%"),
                alt.Tooltip("Volatility:Q", format=".2%"),
                alt.Tooltip("Top Holdings:N"),
            ],
        )
        .transform_calculate(Type="'Efficient Frontier'")
    )


def _create_assets_layer(
    assets_df: pd.DataFrame, 
    optimal_weights: np.ndarray, 
    tickers: List[str],
    config: Dict
) -> alt.Chart:
    """Creates the individual assets layer with labels."""
    # Determine which assets to label (>1% weight in optimal)
    tickers_to_label = set()
    for i, w in enumerate(optimal_weights):
        if w > 0.01 and i < len(tickers):
            tickers_to_label.add(tickers[i])
    
    assets_df["Label"] = assets_df["Ticker"].apply(
        lambda t: t if t in tickers_to_label else ""
    )
    
    points = (
        alt.Chart(assets_df)
        .mark_circle(size=150, opacity=1)
        .encode(
            x="Volatility:Q",
            y="Return:Q",
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(domain=config["legend_domain"], range=config["legend_range"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Return:Q", format=".2%"),
                alt.Tooltip("Volatility:Q", format=".2%"),
                alt.Tooltip("Top Holdings:N"),
            ],
        )
        .transform_calculate(Type="'Assets'")
    )
    
    labels = (
        alt.Chart(assets_df)
        .mark_text(
            align="left",
            dx=12,
            dy=-12,
            color="white",
            fontSize=13,
            fontWeight="bold"
        )
        .encode(x="Volatility:Q", y="Return:Q", text="Label")
    )
    
    return points + labels


def _create_optimal_layer(optimal_df: pd.DataFrame) -> alt.Chart:
    """Creates the optimal portfolio star marker."""
    return (
        alt.Chart(optimal_df)
        .mark_point(shape="star", size=300, filled=True, color="#FF0000")
        .encode(
            x="Volatility:Q",
            y="Return:Q",
            tooltip=[
                alt.Tooltip("Type"),
                alt.Tooltip("Return:Q", format=".2%"),
                alt.Tooltip("Volatility:Q", format=".2%"),
                alt.Tooltip("Top Holdings:N"),
            ],
        )
        .transform_calculate(Type_Label="'Max Sharpe'")
    )


def _create_cml_layer(cml_df: pd.DataFrame, config: Dict) -> alt.Chart:
    """Creates the Capital Market Line layer."""
    return (
        alt.Chart(cml_df)
        .mark_line(
            strokeWidth=2,
            strokeDash=[5, 5],
            opacity=0.8
        )
        .encode(
            x="Volatility:Q",
            y="Return:Q",
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(domain=config["legend_domain"], range=config["legend_range"]),
                legend=None,
            ),
        )
        .transform_calculate(Type="'Capital Market Line'")
    )


def plot_efficient_frontier(
    frontier_points: List[Dict[str, Any]],
    random_portfolios: List[Dict[str, Any]],
    optimal_portfolio: Dict[str, Any],
    tickers: List[str],
    asset_returns: npt.NDArray[np.float64],
    asset_vols: npt.NDArray[np.float64],
    rf_rate: float = 0.0,
    target_portfolio: Optional[Dict[str, Any]] = None,
) -> alt.Chart:
    """
    Generates the combined Efficient Frontier and Asset Allocation chart.

    Args:
        frontier_points: List of dicts with 'return', 'volatility', 'weights'.
        random_portfolios: List of dicts with 'return', 'volatility', 'weights'.
        optimal_portfolio: Dict with 'return', 'volatility', 'weights', 'sharpe'.
        tickers: List of asset ticker symbols.
        asset_returns: Array of expected returns for each asset.
        asset_vols: Array of volatilities for each asset.
        rf_rate: Risk-free rate for reference.
        target_portfolio: Optional dict with 'return', 'volatility', 'weights' for target allocation.

    Returns:
        An Altair Chart with all layers combined.
    """
    # Get chart configuration
    config = _create_base_chart_config()
    
    # --- Prepare DataFrames ---
    
    # Frontier data
    frontier_data = [
        {
            "Return": p["return"],
            "Volatility": p["volatility"],
            "Top Holdings": _format_top_holdings(p["weights"], tickers),
        }
        for p in frontier_points
    ]
    frontier_df = pd.DataFrame(frontier_data)
    
    # Random portfolios data
    random_data = [
        {
            "Return": p["return"],
            "Volatility": p["volatility"],
            "Top Holdings": _format_top_holdings(p["weights"], tickers),
        }
        for p in random_portfolios
    ]
    random_df = pd.DataFrame(random_data)
    
    # Assets data
    assets_df = pd.DataFrame({
        "Ticker": tickers,
        "Return": asset_returns,
        "Volatility": asset_vols,
        "Type": "Asset",
        "Top Holdings": [f"{t}: 100%" for t in tickers],
    })
    
    # Optimal portfolio data
    optimal_df = pd.DataFrame([{
        "Return": optimal_portfolio["return"],
        "Volatility": optimal_portfolio["volatility"],
        "Type": "Max Sharpe",
        "Ticker": "Optimal",
        "Top Holdings": _format_top_holdings(optimal_portfolio["weights"], tickers),
    }])
    # CML data
    cml_data = [
        # Point 1: Risk Free Rate (0 vol)
        {
            "Return": rf_rate,
            "Volatility": 0.0,
            "Top Holdings": "100% CASH",
            "Type": "Capital Market Line"
        },
        # Point 2: Tangency Portfolio
        {
            "Return": optimal_portfolio["return"],
            "Volatility": optimal_portfolio["volatility"],
            "Top Holdings": _format_top_holdings(optimal_portfolio["weights"], tickers),
            "Type": "Capital Market Line"
        }
    ]
    cml_df = pd.DataFrame(cml_data)
    
    # --- Create Chart Layers ---
    
    cloud_chart = _create_cloud_layer(random_df, config)
    frontier_chart = _create_frontier_layer(frontier_df, config)
    cml_chart = _create_cml_layer(cml_df, config)
    assets_chart = _create_assets_layer(
        assets_df, 
        optimal_portfolio["weights"], 
        tickers,
        config
    )
    optimal_chart = _create_optimal_layer(optimal_df)
    
    # Combine base layers
    combined = cloud_chart + frontier_chart + cml_chart + assets_chart + optimal_chart
    
    # Add target layer if provided
    if target_portfolio:
        target_df = pd.DataFrame([{
            "Return": target_portfolio["return"],
            "Volatility": target_portfolio["volatility"],
            "Type": "Target Portfolio",
            "Top Holdings": _format_top_holdings(
                target_portfolio.get("weights", np.array([])), 
                tickers
            ) if "weights" in target_portfolio else "Custom Allocation",
        }])
        target_chart = _create_target_layer(target_df)
        combined = combined + target_chart
    
    # Final composition
    return (
        combined
        .properties(
            width="container",
            height=500,
            title=alt.TitleParams(
                text="Efficient Frontier & Feasible Set",
                fontSize=20,
                anchor="start"
            ),
        )
        .interactive()
        .configure_view(strokeOpacity=0)
    )
