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


_LIGHT_THEME = {
    "axis": "#5f6c76",
    "grid": "#d8e0e6",
    "paper": "#fffdf8",
    "accent": "#1b6c5c",
    "accent_alt": "#c76b3f",
    "asset": "#7b8794",
    "tangency": "#155548",
    "final": "#c79b37",
}


def _empty_xy_frame(*, series: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({"x": [], "y": []})
    if series is not None:
        frame["Series"] = pd.Series(dtype="object")
    return frame


def _build_cml_segment_frames(
    tangency: Dict,
    cml_points: List[Dict],
    lending_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build explicit lending and borrowing CML segments."""
    t_vol = _safe_float(tangency.get("volatility"))
    t_ret = _safe_float(tangency.get("expected_return"))

    borrowing_rows: list[dict[str, float | str]] = []
    lending_df = pd.DataFrame(
        [
            {
                "x": 0.0,
                "y": _safe_float(lending_rate),
                "Series": "Lending Capital Market Line",
            },
            {
                "x": t_vol,
                "y": t_ret,
                "Series": "Lending Capital Market Line",
            },
        ]
    )

    for point in cml_points:
        if _safe_float(point.get("cash_weight")) < -1e-8:
            borrowing_rows.append(
                {
                    "x": _safe_float(point.get("volatility")),
                    "y": _safe_float(point.get("expected_return")),
                    "Series": "Borrowing Capital Market Line",
                }
            )

    borrowing_df = pd.DataFrame(
        borrowing_rows,
        columns=["x", "y", "Series"],
    )
    if not borrowing_df.empty:
        borrowing_df = borrowing_df[
            np.isfinite(borrowing_df["x"]) & np.isfinite(borrowing_df["y"])
        ].copy()
        borrowing_df = borrowing_df.sort_values("x", kind="stable").reset_index(drop=True)
        borrowing_df = borrowing_df.tail(1).copy()
        borrowing_df = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "x": t_vol,
                            "y": t_ret,
                            "Series": "Borrowing Capital Market Line",
                        }
                    ]
                ),
                borrowing_df,
            ],
            ignore_index=True,
        )

    if lending_df.empty:
        lending_df = _empty_xy_frame(series="Lending Capital Market Line")
    if borrowing_df.empty:
        borrowing_df = _empty_xy_frame(series="Borrowing Capital Market Line")

    return lending_df, borrowing_df


def render_results(
    tangency: Dict,
    final_portfolio: Dict,
    cml_points: List[Dict],
    lending_rate: float,
    borrowing_rate: float | None = None,
):
    alloc_df = _create_allocation_df(final_portfolio, tangency, lending_rate, borrowing_rate)
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
    st.altair_chart(chart, width="stretch", theme="streamlit")

    csv_data = _create_results_csv(final_portfolio, tangency)
    st.download_button(
        label="Download results (CSV)",
        data=csv_data,
        file_name="portfolio_optimization_results.csv",
        mime="text/csv",
        width="stretch",
    )

    st.markdown("#### Final allocation")
    st.caption(
        "The final portfolio is the tangency portfolio scaled along the Capital Market Line."
    )
    st.markdown(
        f"""
        <div style="
            background: var(--accent-soft);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem;
            color: var(--ink);
            line-height: 1.6;
            margin-bottom: 0.75rem;
        ">
            <strong style="display:block; color:var(--ink); margin-bottom:0.2rem;">Position summary</strong>
            {position_summary}
        </div>
        """,
        unsafe_allow_html=True,
    )
    allocation_chart = _create_allocation_chart(alloc_df)
    st.altair_chart(allocation_chart, width="stretch", theme="streamlit")
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
    lending_df, borrowing_df = _build_cml_segment_frames(
        tangency,
        cml_points,
        lending_rate,
    )

    rows = []

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

    df = pd.DataFrame(rows, columns=["x", "y", "Category", "MarkType", "Label"])
    if not df.empty:
        df = df[np.isfinite(df["x"]) & np.isfinite(df["y"])].copy()

    if df.empty:
        df = pd.DataFrame({
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "Category": ["Tangency Portfolio", "Tangency Portfolio"],
            "MarkType": ["point", "point"],
            "Label": ["None", "None"]
        })

    color_scale = alt.Scale(
        domain=[
            "Lending Capital Market Line",
            "Borrowing Capital Market Line",
            "Assets",
            "Tangency Portfolio",
            "Final Portfolio",
        ],
        range=[
            _LIGHT_THEME["accent"],
            _LIGHT_THEME["accent_alt"],
            _LIGHT_THEME["asset"],
            _LIGHT_THEME["tangency"],
            _LIGHT_THEME["final"],
        ],
    )

    x_axis = alt.X(
        "x:Q",
        title="Volatility (Risk)",
        scale=alt.Scale(nice=True, zero=True, domainMin=0, padding=12),
        axis=alt.Axis(
            format="%",
            labelColor=_LIGHT_THEME["axis"],
            titleColor=_LIGHT_THEME["axis"],
            gridColor=_LIGHT_THEME["grid"],
        ),
    )
    y_axis = alt.Y(
        "y:Q",
        title="Expected Return",
        scale=alt.Scale(nice=True, zero=True, padding=12),
        axis=alt.Axis(
            format="%",
            labelColor=_LIGHT_THEME["axis"],
            titleColor=_LIGHT_THEME["axis"],
            gridColor=_LIGHT_THEME["grid"],
        ),
    )

    layers = []

    if len(lending_df) >= 2:
        lending_line = (
            alt.Chart(lending_df)
            .mark_line(size=3)
            .encode(
                x=x_axis,
                y=y_axis,
                color=alt.Color("Series:N", scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("x:Q", format=".2%", title="Volatility"),
                    alt.Tooltip("y:Q", format=".2%", title="Return"),
                ],
            )
        )
        layers.append(lending_line)

    if len(borrowing_df) >= 2:
        borrowing_line = (
            alt.Chart(borrowing_df)
            .mark_line(size=3, strokeDash=[6, 4], opacity=0.8)
            .encode(
                x=x_axis,
                y=y_axis,
                color=alt.Color("Series:N", scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("x:Q", format=".2%", title="Volatility"),
                    alt.Tooltip("y:Q", format=".2%", title="Return"),
                ],
            )
        )
        layers.append(borrowing_line)

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
            .mark_point(shape="circle", size=180, filled=True, stroke="black", strokeWidth=1.2)
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
            .mark_point(shape="cross", size=260, filled=True, strokeWidth=3)
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
        dummy_df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "Category": ["None", "None"], "Label": ["None", "None"]})
        dummy_layer = alt.Chart(dummy_df).mark_point(opacity=0).encode(x=x_axis, y=y_axis)
        layers.append(dummy_layer)

    return (
        alt.layer(*layers)
        .properties(height=520)
        .configure_view(stroke=None)
        .configure(background=_LIGHT_THEME["paper"])
        .configure_axis(labelFont="Urbanist", titleFont="Urbanist")
        .configure_legend(
            labelFont="Urbanist",
            titleFont="Urbanist",
            orient="top",
            direction="horizontal",
            columns=2,
        )
    )


def _create_allocation_chart(alloc_df: pd.DataFrame) -> alt.Chart:
    """Create a compact allocation bar chart."""
    if alloc_df.empty:
        chart_df = pd.DataFrame({"Asset": ["None", "None"], "Weight": [0.0, 1.0], "Category": ["Cash", "Cash"], "Expected Return": [0.0, 1.0], "Volatility": [0.0, 1.0]})
    else:
        chart_df = alloc_df.copy()
        chart_df = chart_df[np.isfinite(chart_df["Weight"])].copy()
        if chart_df.empty:
            chart_df = pd.DataFrame({"Asset": ["None", "None"], "Weight": [0.0, 1.0], "Category": ["Cash", "Cash"], "Expected Return": [0.0, 1.0], "Volatility": [0.0, 1.0]})
    chart_df = chart_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Weight"]).copy()
    if chart_df.empty:
        chart_df = pd.DataFrame({"Asset": ["None", "None"], "Weight": [0.0, 1.0], "Category": ["Cash", "Cash"], "Expected Return": [0.0, 1.0], "Volatility": [0.0, 1.0]})
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
                ),
            ),
            y=alt.Y(
                "Asset:N",
                sort="-x",
                title=None,
            ),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(
                    domain=["Risky assets", "Cash", "Borrowing"],
                    range=[
                        _LIGHT_THEME["accent"],
                        _LIGHT_THEME["asset"],
                        _LIGHT_THEME["accent_alt"],
                    ],
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
        .configure_view(stroke=None)
        .configure(background=_LIGHT_THEME["paper"])
        .configure_axis(labelFont="Urbanist", titleFont="Urbanist")
    )


def _create_allocation_df(
    final_portfolio: Dict, 
    universe_context: Dict,
    lending_rate: float = 0.0,
    borrowing_rate: float | None = None,
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
                "Expected Return": lending_rate,
                "Volatility": 0.0,
            }
        )
    elif cash_w < -0.0001:
        rows.append(
            {
                "Asset": "Borrowing",
                "Weight": cash_w,
                "Expected Return": borrowing_rate if borrowing_rate is not None else lending_rate,
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
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 18px;
                overflow: hidden;
                box-shadow: var(--shadow);
            }}

            .allocation-table thead tr {{
                background: var(--bg);
                color: var(--ink);
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
                color: var(--ink);
                border-top: 1px solid var(--line);
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
