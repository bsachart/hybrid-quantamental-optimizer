"""
Results Display Component - Robust Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, List, Any


def render_results(
    tangency: Dict,
    final_portfolio: Dict,
    cml_points: List[Dict],
    rf_rate: float,
):
    """
    Render results with strict type handling for Pyodide.
    """
    # 1. Create Chart
    chart = _create_frontier_chart(
        tangency=tangency,
        final_portfolio=final_portfolio,
        cml_points=cml_points,
        rf_rate=rf_rate,
    )
    st.altair_chart(chart, use_container_width=True)

    # 2. Allocation Table
    st.markdown("---")
    st.subheader("📋 Final Allocation Breakdown")

    alloc_df = _create_allocation_table(
        final_portfolio=final_portfolio, universe_context=tangency
    )

    st.dataframe(
        alloc_df.style.format(
            {"Weight": "{:.2%}", "Expected Return": "{:.2%}", "Volatility": "{:.2%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    risky_pct = (1 - final_portfolio["cash_weight"]) * 100
    cash_pct = final_portfolio["cash_weight"] * 100

    st.info(
        f"💡 **Allocation Strategy**: {risky_pct:.1f}% in optimized risky portfolio, "
        f"{cash_pct:.1f}% in cash to achieve {final_portfolio['volatility']:.1%} target volatility."
    )


def _clean_val(val: Any) -> float:
    """
    Robust float conversion.
    """
    if val is None:
        return 0.0
    try:
        # Extract item if it's a 0-d numpy array
        if hasattr(val, "item"):
            val = val.item()

        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return 0.0
        return f
    except (ValueError, TypeError):
        return 0.0


def _force_float_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Explicitly cast DataFrame columns to float to ensure JSON serialization works.
    """
    if df.empty:
        return df
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def _create_frontier_chart(
    tangency: Dict,
    final_portfolio: Dict,
    cml_points: List[Dict],
    rf_rate: float,
) -> alt.Chart:
    """
    Generates a layered Altair chart.
    """

    # --- 1. Prepare DataFrames ---

    # Efficient Frontier
    frontier_data = [
        {
            "Volatility": _clean_val(p.get("volatility")),
            "Return": _clean_val(p.get("expected_return")),
            "Type": "Efficient Frontier",
        }
        for p in cml_points
    ]
    frontier_df = pd.DataFrame(frontier_data)
    frontier_df = _force_float_cols(frontier_df, ["Volatility", "Return"])

    # Capital Market Line (CML)
    t_vol = _clean_val(tangency.get("volatility"))
    t_ret = _clean_val(tangency.get("expected_return"))
    rf = _clean_val(rf_rate)

    cml_df = pd.DataFrame(
        [
            {"Volatility": 0.0, "Return": rf, "Type": "CML"},
            {"Volatility": t_vol, "Return": t_ret, "Type": "CML"},
        ]
    )
    cml_df = _force_float_cols(cml_df, ["Volatility", "Return"])

    # Assets (Scatter)
    asset_vols = tangency.get("asset_vols", [])
    asset_rets = tangency.get("asset_returns", [])
    tickers = tangency.get("tickers", [])

    # Defensive check on lengths
    n_assets = min(len(tickers), len(asset_vols), len(asset_rets))

    assets_data = []
    for i in range(n_assets):
        assets_data.append(
            {
                "Volatility": _clean_val(asset_vols[i]),
                "Return": _clean_val(asset_rets[i]),
                "Ticker": str(tickers[i]),
                "Type": "Assets",
            }
        )
    assets_df = pd.DataFrame(assets_data)
    assets_df = _force_float_cols(assets_df, ["Volatility", "Return"])

    # Special Points
    optimal_df = pd.DataFrame(
        [
            {
                "Volatility": t_vol,
                "Return": t_ret,
                "Type": "Max Sharpe",
                "Ticker": "Max Sharpe",
            }
        ]
    )
    optimal_df = _force_float_cols(optimal_df, ["Volatility", "Return"])

    target_vol = _clean_val(final_portfolio.get("volatility"))
    target_ret = _clean_val(final_portfolio.get("expected_return"))
    target_df = pd.DataFrame(
        [
            {
                "Volatility": target_vol,
                "Return": target_ret,
                "Type": "Target Portfolio",
                "Ticker": "Target",
            }
        ]
    )
    target_df = _force_float_cols(target_df, ["Volatility", "Return"])

    # --- 2. Build Layers ---

    # Common Scales
    domain = ["Efficient Frontier", "CML", "Assets", "Max Sharpe", "Target Portfolio"]
    range_ = ["#00E676", "#FFC107", "#00BCD4", "#FF5252", "#FFEB3B"]

    # Base Chart
    base = alt.Chart().encode(
        x=alt.X(
            "Volatility",
            type="quantitative",
            axis=alt.Axis(format="%", title="Volatility (Risk)"),
        ),
        y=alt.Y(
            "Return",
            type="quantitative",
            axis=alt.Axis(format="%", title="Expected Return"),
        ),
        color=alt.Color(
            "Type",
            scale=alt.Scale(domain=domain, range=range_),
            legend=alt.Legend(title="Legend"),
        ),
    )

    # Layer 1: CML (Dashed Line)
    cml_layer = base.mark_line(strokeDash=[5, 5], size=2).properties(data=cml_df)

    # Layer 2: Frontier (Solid Line)
    frontier_layer = base.mark_line(size=3).properties(data=frontier_df)

    # Layer 3: Assets (Points)
    assets_layer = (
        base.mark_circle(size=80, opacity=0.7)
        .encode(
            tooltip=[
                "Ticker",
                alt.Tooltip("Return", format=".2%"),
                alt.Tooltip("Volatility", format=".2%"),
            ]
        )
        .properties(data=assets_df)
    )

    # Layer 4: Optimal (Star)
    optimal_layer = (
        base.mark_point(shape="star", size=300, filled=True)
        .encode(
            tooltip=[
                "Type",
                alt.Tooltip("Return", format=".2%"),
                alt.Tooltip("Volatility", format=".2%"),
            ]
        )
        .properties(data=optimal_df)
    )

    # Layer 5: Target (Diamond)
    target_layer = (
        base.mark_point(shape="diamond", size=250, filled=True)
        .encode(
            tooltip=[
                "Type",
                alt.Tooltip("Return", format=".2%"),
                alt.Tooltip("Volatility", format=".2%"),
            ]
        )
        .properties(data=target_df)
    )

    # --- 3. Combine ---
    return (
        alt.layer(cml_layer, frontier_layer, assets_layer, optimal_layer, target_layer)
        .properties(title="Efficient Frontier & Capital Market Line", height=500)
        .interactive()
    )


def _create_allocation_table(
    final_portfolio: Dict, universe_context: Dict
) -> pd.DataFrame:
    """Create formatted allocation table."""
    tickers = universe_context.get("tickers", [])
    weights = final_portfolio.get("weights", [])
    returns = universe_context.get("asset_returns", [])
    vols = universe_context.get("asset_vols", [])

    # FIX: Robust check for empty weights (handles None, empty list, and numpy arrays)
    if weights is None or len(weights) == 0:
        return pd.DataFrame()

    rows = []

    # Cash row
    cash_w = _clean_val(final_portfolio.get("cash_weight"))
    if cash_w > 0.0001:
        rows.append(
            {
                "Asset": "CASH",
                "Weight": cash_w,
                "Expected Return": 0.0,
                "Volatility": 0.0,
            }
        )

    # Asset rows
    for i, ticker in enumerate(tickers):
        # Access weights safely
        w = _clean_val(weights[i])
        if abs(w) > 0.0001:
            r = _clean_val(returns[i]) if i < len(returns) else 0.0
            v = _clean_val(vols[i]) if i < len(vols) else 0.0
            rows.append(
                {
                    "Asset": str(ticker),
                    "Weight": w,
                    "Expected Return": r,
                    "Volatility": v,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Weight", ascending=False, key=abs)
    return df
