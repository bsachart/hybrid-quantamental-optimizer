"""
Results Display Component - Unified Data Model
"""

from html import escape
from textwrap import dedent
from typing import Dict, List, Any

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt


def render_results(
    tangency: Dict,
    final_portfolio: Dict,
    cml_points: List[Dict],
    lending_rate: float,
    borrowing_rate: float | None = None,
):
    alloc_df = _create_allocation_df(final_portfolio, tangency)
    position_summary = _build_position_summary(final_portfolio)

    st.markdown("#### Capital Market Line")
    st.caption(
        "The lending line starts at the lending rate. When borrowing applies, "
        "the borrowing line starts at the borrowing rate and meets the tangency portfolio."
    )
    chart = _create_chart(
        tangency,
        final_portfolio,
        cml_points,
        lending_rate,
        borrowing_rate=borrowing_rate,
    )
    st.altair_chart(chart, use_container_width=True, theme=None)

    csv_data = _create_results_csv(final_portfolio, tangency)
    st.download_button(
        label="Download results (CSV)",
        data=csv_data,
        file_name="portfolio_optimization_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("#### Final allocation")
    st.caption(
        "The final portfolio is the tangency portfolio scaled along the Capital Market Line."
    )
    st.markdown(
        f"""
        <div style="
            background: rgba(22, 108, 89, 0.08);
            border: 1px solid rgba(22, 108, 89, 0.14);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem;
            color: #355247;
            line-height: 1.6;
            margin-bottom: 0.75rem;
        ">
            <strong style="display:block; color:#1f3b32; margin-bottom:0.2rem;">Position summary</strong>
            {position_summary}
        </div>
        """,
        unsafe_allow_html=True,
    )
    allocation_chart = _create_allocation_chart(alloc_df)
    st.altair_chart(allocation_chart, use_container_width=True, theme=None)
    _render_allocation_table(alloc_df)


def _safe_float(val: Any) -> float:
    """Convert to native Python float, defaulting to 0.0 on error."""
    try:
        if val is None:
            return 0.0
        if hasattr(val, "item"):
            val = val.item()
        f = float(val)
        return f if np.isfinite(f) else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _create_chart(
    tangency: Dict,
    final_portfolio: Dict,
    cml_points: List[Dict],
    lending_rate: float,
    borrowing_rate: float | None = None,
) -> alt.Chart:
    """
    Create chart using a unified data model.

    Design: One DataFrame, two mark layers (lines + points).
    The 'Category' column drives color encoding and auto-generates the legend.
    """
    t_vol = _safe_float(tangency.get("volatility"))
    t_ret = _safe_float(tangency.get("expected_return"))

    lending_rows = []
    borrowing_rows = []
    for p in cml_points:
        row = {
            "x": _safe_float(p.get("volatility")),
            "y": _safe_float(p.get("expected_return")),
            "MarkType": "line",
            "Label": "",
        }
        if _safe_float(p.get("cash_weight")) < -1e-8:
            row["Category"] = "Borrowing Capital Market Line"
            borrowing_rows.append(row)
        else:
            row["Category"] = "Lending Capital Market Line"
            lending_rows.append(row)

    rows = lending_rows.copy()
    if borrowing_rows:
        borrowing_rows.insert(
            0,
            {
                "x": t_vol,
                "y": t_ret,
                "Category": "Borrowing Capital Market Line",
                "MarkType": "line",
                "Label": "",
            },
        )
        rows.extend(borrowing_rows)

    tickers = tangency.get("tickers", [])
    asset_rets = tangency.get("asset_returns", [])
    asset_vols = tangency.get("asset_vols", [])

    for i, ticker in enumerate(tickers):
        if i < len(asset_vols) and i < len(asset_rets):
            rows.append(
                {
                    "x": _safe_float(asset_vols[i]),
                    "y": _safe_float(asset_rets[i]),
                    "Category": "Assets",
                    "MarkType": "point",
                    "Label": str(ticker),
                }
            )

    rows.append(
        {
            "x": t_vol,
            "y": t_ret,
            "Category": "Tangency Portfolio",
            "MarkType": "point",
            "Label": "Tangency Portfolio",
        }
    )

    rows.append(
        {
            "x": _safe_float(final_portfolio.get("volatility")),
            "y": _safe_float(final_portfolio.get("expected_return")),
            "Category": "Final Portfolio",
            "MarkType": "point",
            "Label": "Final Portfolio",
        }
    )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[np.isfinite(df["x"]) & np.isfinite(df["y"])].copy()

    # Ensure no empty DataFrame
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": [0.0], "y": [0.0]})).mark_point()

    # --- CHART CONFIGURATION ---
    color_scale = alt.Scale(
        domain=[
            "Lending Capital Market Line",
            "Borrowing Capital Market Line",
            "Assets",
            "Tangency Portfolio",
            "Final Portfolio",
        ],
        range=["#166c59", "#c76b3f", "#6d8391", "#0f4d40", "#d7aa4b"],
    )

    x_axis = alt.X(
        "x:Q",
        title="Volatility (Risk)",
        scale=alt.Scale(nice=True, zero=True, domainMin=0, padding=12),
        axis=alt.Axis(
            format="%",
            labelColor="#6e6a63",
            titleColor="#22252b",
            gridColor="rgba(34, 37, 43, 0.12)",
            domainColor="rgba(34, 37, 43, 0.16)",
            tickColor="rgba(34, 37, 43, 0.16)",
        ),
    )
    y_axis = alt.Y(
        "y:Q",
        title="Expected Return",
        scale=alt.Scale(nice=True, zero=True, padding=12),
        axis=alt.Axis(
            format="%",
            labelColor="#6e6a63",
            titleColor="#22252b",
            gridColor="rgba(34, 37, 43, 0.12)",
            domainColor="rgba(34, 37, 43, 0.16)",
            tickColor="rgba(34, 37, 43, 0.16)",
        ),
    )

    df_lines = df[df["MarkType"] == "line"].copy()

    layers = []

    if len(df_lines) >= 2:
        cml = (
            alt.Chart(df_lines)
            .mark_line(size=3)
            .encode(
                x=x_axis,
                y=y_axis,
                color=alt.Color(
                    "Category:N",
                    scale=color_scale,
                    legend=alt.Legend(
                        title="Series",
                        orient="top",
                        direction="horizontal",
                        columns=2,
                        labelColor="#4d4b46",
                        titleColor="#22252b",
                    ),
                ),
                strokeDash=alt.StrokeDash(
                    "Category:N",
                    scale=alt.Scale(
                        domain=[
                            "Lending Capital Market Line",
                            "Borrowing Capital Market Line",
                        ],
                        range=[[1, 0], [6, 4]],
                    ),
                    legend=None,
                ),
                order="x:Q",
            )
        )
        layers.append(cml)

    df_points = df[df["MarkType"] == "point"].copy()

    # Assets (circles)
    df_assets = df_points[df_points["Category"] == "Assets"]
    if not df_assets.empty:
        assets = (
            alt.Chart(df_assets)
            .mark_circle(size=100, opacity=0.8)
            .encode(
                x=x_axis,
                y=y_axis,
                color=alt.Color("Category:N", scale=color_scale),
                tooltip=[
                    alt.Tooltip("Label:N", title="Ticker"),
                    alt.Tooltip("x:Q", format=".2%", title="Volatility"),
                    alt.Tooltip("y:Q", format=".2%", title="Return"),
                ],
            )
        )
        layers.append(assets)

    df_tangency = df_points[df_points["Category"] == "Tangency Portfolio"]
    if not df_tangency.empty:
        tangency_point = (
            alt.Chart(df_tangency)
            .mark_point(shape="cross", size=200, filled=True)
            .encode(
                x=x_axis,
                y=y_axis,
                color=alt.Color("Category:N", scale=color_scale),
                tooltip=[
                    alt.Tooltip("Label:N", title="Portfolio"),
                    alt.Tooltip("x:Q", format=".2%", title="Volatility"),
                    alt.Tooltip("y:Q", format=".2%", title="Return"),
                ],
            )
        )
        layers.append(tangency_point)

    df_target = df_points[df_points["Category"] == "Final Portfolio"]
    if not df_target.empty:
        target = (
            alt.Chart(df_target)
            .mark_point(shape="diamond", size=200, filled=True)
            .encode(
                x=x_axis,
                y=y_axis,
                color=alt.Color("Category:N", scale=color_scale),
                tooltip=[
                    alt.Tooltip("Label:N", title="Portfolio"),
                    alt.Tooltip("x:Q", format=".2%", title="Volatility"),
                    alt.Tooltip("y:Q", format=".2%", title="Return"),
                ],
            )
        )
        layers.append(target)

    if not layers:
        return alt.Chart(pd.DataFrame({"x": [0.0], "y": [0.0]})).mark_point()

    return (
        alt.layer(*layers)
        .properties(height=520)
        .configure(background="#fffdf9")
        .configure_view(stroke=None)
        .configure_axis(labelFont="Urbanist", titleFont="Urbanist")
        .configure_legend(labelFont="Urbanist", titleFont="Urbanist")
    )


def _create_allocation_chart(alloc_df: pd.DataFrame) -> alt.Chart:
    """Create a compact allocation bar chart."""
    if alloc_df.empty:
        return alt.Chart(pd.DataFrame({"Asset": [], "Weight": []})).mark_bar()

    chart_df = alloc_df.copy()
    chart_df = chart_df[np.isfinite(chart_df["Weight"])].copy()
    if chart_df.empty:
        return alt.Chart(pd.DataFrame({"Asset": [], "Weight": []})).mark_bar()
    chart_df["Category"] = chart_df["Asset"].apply(
        lambda asset: (
            "Cash"
            if asset == "Cash"
            else "Borrowing" if asset == "Borrowing" else "Risky assets"
        )
    )

    return (
        alt.Chart(chart_df)
        .mark_bar(size=24, cornerRadiusEnd=6)
        .encode(
            x=alt.X(
                "Weight:Q",
                title="Weight",
                stack=None,
                scale=alt.Scale(nice=True, zero=True),
                axis=alt.Axis(
                    format="%",
                    labelColor="#6e6a63",
                    titleColor="#22252b",
                    gridColor="rgba(34, 37, 43, 0.10)",
                ),
            ),
            y=alt.Y(
                "Asset:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelColor="#4d4b46"),
            ),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(
                    domain=["Risky assets", "Cash", "Borrowing"],
                    range=["#166c59", "#c76b3f", "#a44646"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Asset:N"),
                alt.Tooltip("Weight:Q", format=".2%"),
                alt.Tooltip("Expected Return:Q", format=".2%"),
                alt.Tooltip("Volatility:Q", format=".2%"),
            ],
        )
        .properties(height=max(160, 48 * len(chart_df)))
        .configure(background="#fffdf9")
        .configure_view(stroke=None)
        .configure_axis(labelFont="Urbanist", titleFont="Urbanist")
    )


def _create_allocation_df(
    final_portfolio: Dict, universe_context: Dict
) -> pd.DataFrame:
    """Create allocation table DataFrame."""
    tickers = universe_context.get("tickers", [])
    weights = final_portfolio.get("weights", [])
    returns = universe_context.get("asset_returns", [])
    vols = universe_context.get("asset_vols", [])

    if weights is None or len(weights) == 0:
        return pd.DataFrame()

    rows = []

    cash_w = _safe_float(final_portfolio.get("cash_weight"))
    if cash_w > 0.0001:
        rows.append(
            {
                "Asset": "Cash",
                "Weight": cash_w,
                "Expected Return": 0.0,
                "Volatility": 0.0,
            }
        )
    elif cash_w < -0.0001:
        rows.append(
            {
                "Asset": "Borrowing",
                "Weight": cash_w,
                "Expected Return": 0.0,
                "Volatility": 0.0,
            }
        )

    for i, ticker in enumerate(tickers):
        w = _safe_float(weights[i])
        if abs(w) > 0.0001:
            rows.append(
                {
                    "Asset": str(ticker),
                    "Weight": w,
                    "Expected Return": _safe_float(returns[i])
                    if i < len(returns)
                    else 0.0,
                    "Volatility": _safe_float(vols[i]) if i < len(vols) else 0.0,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Weight", ascending=False, key=abs)
    return df


def _create_results_csv(final_portfolio: Dict, tangency: Dict) -> str:
    """Generate a CSV summary of optimization results."""
    import io
    import csv

    tickers = tangency.get("tickers", [])
    rows = []

    # Header info
    rows.append(["Optimization Results"])
    rows.append(
        ["Target Volatility", f"{_safe_float(final_portfolio.get('volatility')):.2%}"]
    )
    rows.append([])

    # Portfolio metrics
    rows.append(["Portfolio Metrics"])
    rows.append(
        [
            "Expected Return",
            f"{_safe_float(final_portfolio.get('expected_return')):.2%}",
        ]
    )
    rows.append(["Volatility", f"{_safe_float(final_portfolio.get('volatility')):.2%}"])
    rows.append(
        ["Sharpe Ratio", f"{_safe_float(final_portfolio.get('sharpe_ratio')):.2f}"]
    )
    rows.append(
        ["Cash Weight", f"{_safe_float(final_portfolio.get('cash_weight')):.2%}"]
    )
    rows.append([])

    # Allocations
    rows.append(["Asset", "Weight", "Expected Return", "Volatility"])

    cash_w = _safe_float(final_portfolio.get("cash_weight"))
    if cash_w > 0.0001:
        rows.append(["Cash", f"{cash_w:.4f}", "0.0000", "0.0000"])
    elif cash_w < -0.0001:
        rows.append(["Borrowing", f"{cash_w:.4f}", "0.0000", "0.0000"])

    weights = final_portfolio.get("weights", [])
    asset_returns = tangency.get("asset_returns", [])
    asset_vols = tangency.get("asset_vols", [])

    for i, ticker in enumerate(tickers):
        w = _safe_float(weights[i]) if i < len(weights) else 0.0
        if abs(w) > 0.0001:
            ret = _safe_float(asset_returns[i]) if i < len(asset_returns) else 0.0
            vol = _safe_float(asset_vols[i]) if i < len(asset_vols) else 0.0
            rows.append([ticker, f"{w:.4f}", f"{ret:.4f}", f"{vol:.4f}"])

    # Convert to CSV string
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)
    return output.getvalue()


def _render_allocation_table(alloc_df: pd.DataFrame) -> None:
    """Render a light-themed allocation table."""
    if alloc_df.empty:
        st.info("No allocation rows to display.")
        return

    rows_html = "".join(
        dedent(
            f"""
            <tr>
                <td>{escape(str(row[0]))}</td>
                <td>{row[1]:.2%}</td>
                <td>{row[2]:.2%}</td>
                <td>{row[3]:.2%}</td>
            </tr>
            """
        ).strip()
        for row in alloc_df.itertuples(index=False, name=None)
    )

    table_html = dedent(
        f"""
        <style>
            .allocation-table-wrap {{
                overflow-x: auto;
            }}

            .allocation-table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                background: rgba(255, 255, 252, 0.96);
                border: 1px solid rgba(34, 37, 43, 0.10);
                border-radius: 18px;
                overflow: hidden;
                box-shadow: 0 18px 48px rgba(91, 70, 42, 0.06);
            }}

            .allocation-table thead tr {{
                background: #f3eee4;
                color: #22252b;
            }}

            .allocation-table th,
            .allocation-table td {{
                padding: 0.9rem 1rem;
            }}

            .allocation-table th {{
                font-weight: 700;
                text-align: left;
            }}

            .allocation-table th:not(:first-child),
            .allocation-table td:not(:first-child) {{
                text-align: right;
            }}

            .allocation-table tbody tr td {{
                color: #3d3a35;
                border-top: 1px solid rgba(34, 37, 43, 0.08);
            }}
        </style>
        <div class="allocation-table-wrap">
            <table class="allocation-table">
                <thead>
                    <tr>
                        <th>Asset</th>
                        <th>Weight</th>
                        <th>Expected Return</th>
                        <th>Volatility</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
    ).strip()

    st.html(table_html)


def _build_position_summary(final_portfolio: Dict) -> str:
    risky_weight = 0.0
    for weight in final_portfolio.get("weights", []):
        risky_weight += _safe_float(weight)

    risky_pct = risky_weight * 100.0
    cash_weight = _safe_float(final_portfolio.get("cash_weight"))

    if cash_weight > 0.0001:
        return (
            f"{risky_pct:.1f}% is invested in the tangency portfolio and "
            f"{cash_weight * 100.0:.1f}% stays at the lending rate."
        )

    if cash_weight < -0.0001:
        return (
            f"{risky_pct:.1f}% is invested in the tangency portfolio and "
            f"{abs(cash_weight) * 100.0:.1f}% is financed at the borrowing rate."
        )

    return f"{risky_pct:.1f}% is invested in the tangency portfolio with no cash position."
