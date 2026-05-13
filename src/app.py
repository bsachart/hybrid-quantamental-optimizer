import base64
import io
from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

try:
    from engine.portfolio_engine_pandas import (
        generate_cml,
        optimize_portfolio,
        target_portfolio,
    )
    from engine.risk_pandas import RiskModel
except ModuleNotFoundError:
    from src.engine.portfolio_engine_pandas import (
        generate_cml,
        optimize_portfolio,
        target_portfolio,
    )
    from src.engine.risk_pandas import RiskModel


APP_TITLE = "Hybrid Quantamental Optimizer"
PREVIEW_ROWS = 3

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title=APP_TITLE,
)
server = app.server


def parse_contents(contents: str) -> io.StringIO:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    return io.StringIO(decoded.decode("utf-8"))


def read_uploaded_csv(contents: str) -> tuple[str, pd.DataFrame]:
    stream = parse_contents(contents)
    text = stream.getvalue()
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [str(col).strip() for col in frame.columns]
    return text, frame


def _coerce_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _coerce_json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_coerce_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def serialize_for_store(payload: Any) -> Any:
    return _coerce_json_value(payload)


def restore_portfolio_payload(payload: dict[str, Any]) -> dict[str, Any]:
    restored = dict(payload)
    if "weights" in restored:
        restored["weights"] = np.asarray(restored["weights"], dtype=float)
    return restored


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        if hasattr(value, "item"):
            value = value.item()
        parsed = float(value)
        return parsed if np.isfinite(parsed) else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=title,
        paper_bgcolor="rgba(255, 251, 245, 0.94)",
        plot_bgcolor="rgba(255, 251, 245, 0.94)",
        font=dict(family="Urbanist", color="#1f2933"),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text="Solve a portfolio to populate this chart.",
                showarrow=False,
                font=dict(size=14, color="#5f6c76"),
            )
        ],
        margin=dict(l=40, r=40, t=70, b=40),
        height=420,
    )
    return fig


def render_header() -> html.Div:
    return html.Div(
        className="app-header",
        children=[
            html.Div("Portfolio Construction", className="app-kicker"),
            html.H1(APP_TITLE, className="app-title"),
            html.Div(
                "Move from raw market inputs to a target-risk allocation in one screen: "
                "upload the two source files, solve the risky portfolio, then tune the final cash mix.",
                className="app-copy",
            ),
        ],
    )


def render_upload_section(id_prefix: str, kicker: str, title: str, copy: str) -> html.Div:
    return html.Div(
        className="card-wrapper",
        children=[
            html.Div(kicker, className="card-kicker"),
            html.Div(title, className="card-title"),
            html.Div(copy, className="card-copy"),
            dcc.Upload(
                id=f"upload-{id_prefix}",
                children=html.Div(["Drag and drop or ", html.A("select a CSV")]),
                className="upload-area",
                multiple=False,
            ),
            html.Div(id=f"status-{id_prefix}", className="mt-3"),
        ],
    )


def render_metric_card(label: str, value: str) -> html.Div:
    return html.Div(
        className="metric-card",
        children=[
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
        ],
    )


def render_message(kind: str, title: str, copy: str) -> html.Div:
    return html.Div(
        className=f"status-card status--{kind}",
        children=[
            html.Div(title, className="status-title"),
            html.Div(copy, className="status-copy"),
        ],
    )


def render_preview_table(frame: pd.DataFrame) -> html.Div:
    preview = frame.head(PREVIEW_ROWS).fillna("").astype(str)
    return html.Div(
        className="preview-table-wrap",
        children=html.Table(
            className="preview-table",
            children=[
                html.Thead(html.Tr([html.Th(column) for column in preview.columns])),
                html.Tbody(
                    [
                        html.Tr([html.Td(row[column]) for column in preview.columns])
                        for _, row in preview.iterrows()
                    ]
                ),
            ],
        ),
    )


def build_upload_status(contents: str | None, filename: str | None, label: str) -> tuple[Any, Any]:
    if contents is None:
        return "", None

    try:
        text, frame = read_uploaded_csv(contents)
    except Exception as exc:
        return (
            render_message(
                "warning",
                f"{label} file needs attention",
                f"We couldn't read this CSV yet: {exc}",
            ),
            None,
        )

    rows, cols = frame.shape
    details = html.Div(
        className="file-ready",
        children=[
            html.Div(
                [
                    html.Strong(filename or f"{label.lower()} file"),
                    html.Span(f" • {rows} rows • {cols} columns", className="file-note"),
                ]
            ),
            html.Div(
                "Previewing the first few rows so you can confirm the structure before solving.",
                className="file-note",
            ),
            render_preview_table(frame),
        ],
    )

    store_payload = {
        "filename": filename,
        "text": text,
        "columns": frame.columns.tolist(),
        "rows": rows,
    }
    return details, store_payload


def create_cml_figure(
    tangency: dict[str, Any],
    final_portfolio: dict[str, Any],
    cml_points: list[dict[str, Any]],
    lending_rate: float,
) -> go.Figure:
    fig = go.Figure()

    tickers = tangency.get("tickers", [])
    asset_returns = tangency.get("asset_returns", [])
    asset_vols = tangency.get("asset_vols", [])
    tangency_vol = _safe_float(tangency.get("volatility"))
    tangency_ret = _safe_float(tangency.get("expected_return"))

    fig.add_trace(
        go.Scatter(
            x=asset_vols,
            y=asset_returns,
            mode="markers+text",
            name="Assets",
            text=tickers,
            textposition="top center",
            marker=dict(size=10, color="#7b8794", opacity=0.75),
            hovertemplate="<b>%{text}</b><br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, tangency_vol],
            y=[lending_rate, tangency_ret],
            mode="lines",
            name="Lending CML",
            line=dict(color="#1b6c5c", width=3),
            hovertemplate="Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra>Lending CML</extra>",
        )
    )

    borrowing_points = [
        point for point in cml_points if _safe_float(point.get("cash_weight")) < -1e-8
    ]
    if borrowing_points:
        borrow_vols = [tangency_vol] + [_safe_float(point["volatility"]) for point in borrowing_points]
        borrow_rets = [tangency_ret] + [_safe_float(point["expected_return"]) for point in borrowing_points]
        fig.add_trace(
            go.Scatter(
                x=borrow_vols,
                y=borrow_rets,
                mode="lines",
                name="Borrowing CML",
                line=dict(color="#c76b3f", width=3, dash="dash"),
                hovertemplate="Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra>Borrowing CML</extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[tangency_vol],
            y=[tangency_ret],
            mode="markers",
            name="Tangency Portfolio",
            marker=dict(size=15, color="#155548", symbol="circle", line=dict(width=2, color="#1f2933")),
            hovertemplate="Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra>Tangency Portfolio</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[_safe_float(final_portfolio.get("volatility"))],
            y=[_safe_float(final_portfolio.get("expected_return"))],
            mode="markers",
            name="Final Portfolio",
            marker=dict(size=18, color="#c79b37", symbol="x", line=dict(width=3)),
            hovertemplate="Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra>Final Portfolio</extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255, 251, 245, 0.94)",
        plot_bgcolor="rgba(255, 251, 245, 0.94)",
        font=dict(family="Urbanist", color="#1f2933"),
        xaxis=dict(title="Volatility (Risk)", tickformat=".1%", gridcolor="rgba(31, 41, 51, 0.12)"),
        yaxis=dict(title="Expected Return", tickformat=".1%", gridcolor="rgba(31, 41, 51, 0.12)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        height=520,
    )
    return fig


def create_allocation_figure(final_portfolio: dict[str, Any], tangency: dict[str, Any]) -> go.Figure:
    rows = []
    cash_weight = _safe_float(final_portfolio.get("cash_weight"))
    if abs(cash_weight) > 1e-4:
        rows.append(
            {
                "Asset": "Cash" if cash_weight > 0 else "Borrowing",
                "Weight": cash_weight,
                "Type": "Cash",
            }
        )

    tickers = tangency.get("tickers", [])
    weights = final_portfolio.get("weights", [])
    for index, ticker in enumerate(tickers):
        weight = _safe_float(weights[index]) if index < len(weights) else 0.0
        if abs(weight) > 1e-4:
            rows.append({"Asset": ticker, "Weight": weight, "Type": "Risky"})

    if not rows:
        return _empty_figure("Final portfolio weights")

    frame = pd.DataFrame(rows).sort_values("Weight", ascending=False, key=lambda s: s.abs())

    fig = go.Figure(
        go.Bar(
            x=frame["Weight"],
            y=frame["Asset"],
            orientation="h",
            marker_color=["#7b8794" if kind == "Cash" else "#1b6c5c" for kind in frame["Type"]],
            text=[_format_pct(weight) for weight in frame["Weight"]],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Weight: %{x:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255, 251, 245, 0.94)",
        plot_bgcolor="rgba(255, 251, 245, 0.94)",
        font=dict(family="Urbanist", color="#1f2933"),
        xaxis=dict(title="Weight", tickformat=".1%", gridcolor="rgba(31, 41, 51, 0.12)"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=40, t=40, b=40),
        height=max(220, 44 * len(frame)),
    )
    return fig


def render_allocation_table(
    final_portfolio: dict[str, Any],
    tangency: dict[str, Any],
    lending_rate: float,
    borrowing_rate: float,
) -> html.Div:
    rows = []
    cash_weight = _safe_float(final_portfolio.get("cash_weight"))
    if abs(cash_weight) > 1e-4:
        label = "Cash" if cash_weight > 0 else "Borrowing"
        rows.append(
            html.Tr(
                [
                    html.Td(label),
                    html.Td(_format_pct(cash_weight)),
                    html.Td(_format_pct(lending_rate if cash_weight > 0 else borrowing_rate)),
                    html.Td("0.00%"),
                ]
            )
        )

    tickers = tangency.get("tickers", [])
    weights = final_portfolio.get("weights", [])
    returns = tangency.get("asset_returns", [])
    vols = tangency.get("asset_vols", [])
    for index, ticker in enumerate(tickers):
        weight = _safe_float(weights[index]) if index < len(weights) else 0.0
        if abs(weight) > 1e-4:
            rows.append(
                html.Tr(
                    [
                        html.Td(ticker),
                        html.Td(_format_pct(weight)),
                        html.Td(_format_pct(_safe_float(returns[index]))),
                        html.Td(_format_pct(_safe_float(vols[index]))),
                    ]
                )
            )

    return html.Div(
        className="allocation-table-wrap",
        children=html.Table(
            className="allocation-table",
            children=[
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Asset"),
                            html.Th("Weight"),
                            html.Th("Expected Return"),
                            html.Th("Volatility"),
                        ]
                    )
                ),
                html.Tbody(rows),
            ],
        ),
    )


def render_tangency_metrics(tangency: dict[str, Any]) -> html.Div:
    return html.Div(
        className="card-wrapper",
        children=[
            html.Div("Step 3A", className="card-kicker"),
            html.Div("Tangency portfolio", className="card-title"),
            dbc.Row(
                [
                    dbc.Col(render_metric_card("Expected return", _format_pct(_safe_float(tangency["expected_return"])))),
                    dbc.Col(render_metric_card("Volatility", _format_pct(_safe_float(tangency["volatility"])))),
                    dbc.Col(render_metric_card("Sharpe ratio", f"{_safe_float(tangency['sharpe_ratio']):.2f}")),
                ]
            ),
        ],
    )


def render_final_metrics(
    final_portfolio: dict[str, Any],
    borrow_warning: str | None = None,
) -> html.Div:
    children = [
        html.Div("Step 3C", className="card-kicker"),
        html.Div("Final portfolio", className="card-title"),
    ]
    if borrow_warning:
        children.append(render_message("warning", "Borrowing rate adjusted", borrow_warning))
    children.append(
        dbc.Row(
            [
                dbc.Col(
                    render_metric_card(
                        "Expected return",
                        _format_pct(_safe_float(final_portfolio["expected_return"])),
                    )
                ),
                dbc.Col(
                    render_metric_card(
                        "Volatility",
                        _format_pct(_safe_float(final_portfolio["volatility"])),
                    )
                ),
                dbc.Col(
                    render_metric_card(
                        "Cash / borrowing",
                        _format_pct(_safe_float(final_portfolio["cash_weight"])),
                    )
                ),
            ]
        )
    )
    return html.Div(className="card-wrapper", children=children)


def _build_slider_marks(max_percent: float) -> dict[int, str]:
    upper = max(10, int(np.ceil(max_percent / 5.0) * 5))
    return {mark: f"{mark}%" for mark in range(0, upper + 1, 5)}


def solve_portfolio(
    prices_payload: dict[str, Any] | None,
    metrics_payload: dict[str, Any] | None,
    risk_free_rate_percent: float,
    risk_model_value: str,
) -> dict[str, Any]:
    if not prices_payload or not metrics_payload:
        raise ValueError("Upload both CSV files before solving.")

    risk_free_rate = risk_free_rate_percent / 100.0
    risk_model = RiskModel[risk_model_value]
    tangency = optimize_portfolio(
        price_source=io.StringIO(prices_payload["text"]),
        metric_source=io.StringIO(metrics_payload["text"]),
        risk_free_rate=risk_free_rate,
        risk_model=risk_model,
        annualization_factor=252 if risk_model == RiskModel.HISTORICAL else None,
    )

    target_volatility = _safe_float(tangency["volatility"])
    final_portfolio = target_portfolio(
        tangency_portfolio=tangency,
        target_volatility=target_volatility,
        risk_free_rate=risk_free_rate,
        borrowing_rate=risk_free_rate,
    )
    cml_points = generate_cml(
        tangency_portfolio=tangency,
        risk_free_rate=risk_free_rate,
        borrowing_rate=risk_free_rate,
        max_volatility=max(target_volatility * 2.0, 0.01),
        num_points=41,
    )

    return {
        "tangency": serialize_for_store(tangency),
        "final_portfolio": serialize_for_store(final_portfolio),
        "cml_points": serialize_for_store(cml_points),
        "target_volatility_percent": target_volatility * 100.0,
    }


def build_final_results(
    tangency: dict[str, Any],
    target_volatility_percent: float,
    borrowing_rate_percent: float,
    risk_free_rate_percent: float,
) -> dict[str, Any]:
    tangency_metrics = restore_portfolio_payload(tangency)
    risk_free_rate = risk_free_rate_percent / 100.0
    requested_borrow = borrowing_rate_percent / 100.0
    effective_borrow = max(requested_borrow, risk_free_rate)
    borrow_warning = None
    if effective_borrow != requested_borrow:
        borrow_warning = "Borrowing cannot price below lending in this workflow, so the lending rate is used instead."

    final_portfolio = target_portfolio(
        tangency_portfolio=tangency_metrics,
        target_volatility=target_volatility_percent / 100.0,
        risk_free_rate=risk_free_rate,
        borrowing_rate=effective_borrow,
    )
    cml_points = generate_cml(
        tangency_portfolio=tangency_metrics,
        risk_free_rate=risk_free_rate,
        borrowing_rate=effective_borrow,
        max_volatility=max(_safe_float(tangency_metrics["volatility"]) * 2.0, target_volatility_percent / 100.0, 0.01),
        num_points=41,
    )

    return {
        "final_store": serialize_for_store(final_portfolio),
        "cml_store": serialize_for_store(cml_points),
        "final_metrics": render_final_metrics(final_portfolio, borrow_warning),
        "cml_figure": create_cml_figure(tangency_metrics, final_portfolio, cml_points, risk_free_rate),
        "allocation_figure": create_allocation_figure(final_portfolio, tangency_metrics),
        "allocation_table": render_allocation_table(final_portfolio, tangency_metrics, risk_free_rate, effective_borrow),
    }


app.layout = dbc.Container(
    className="container",
    children=[
        render_header(),
        html.Div(
            className="section-wrap",
            children=[
                html.Div("Setup", className="section-kicker"),
                html.Div("Load inputs and solve the risky portfolio", className="section-title"),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    render_upload_section(
                        "prices",
                        "Step 1A",
                        "Price history",
                        "Upload one CSV with a date column followed by one column per ticker.",
                    ),
                    md=6,
                ),
                dbc.Col(
                    render_upload_section(
                        "metrics",
                        "Step 1B",
                        "Asset metrics",
                        "Upload expected return, implied volatility, and optional weight bounds.",
                    ),
                    md=6,
                ),
            ]
        ),
        html.Div(
            className="card-wrapper",
            children=[
                html.Div("Step 2", className="card-kicker"),
                html.Div("Assumptions and solve", className="card-title"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Risk-free rate (%)", className="metric-label"),
                                dcc.Slider(
                                    id="slider-rf",
                                    min=-5,
                                    max=10,
                                    step=0.25,
                                    value=4.0,
                                    marks={mark: f"{mark}%" for mark in range(-5, 11, 5)},
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Risk model", className="metric-label"),
                                dcc.RadioItems(
                                    id="radio-risk-model",
                                    options=[
                                        {"label": "Forward-looking", "value": "FORWARD_LOOKING"},
                                        {"label": "Historical", "value": "HISTORICAL"},
                                    ],
                                    value="FORWARD_LOOKING",
                                    inline=True,
                                    inputClassName="me-2",
                                    labelClassName="me-4",
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="mb-4",
                ),
                html.Button(
                    "Solve risky portfolio",
                    id="btn-solve",
                    className="primary-button",
                    disabled=True,
                ),
                dcc.Loading(
                    children=html.Div(
                        id="solve-message",
                        children=render_message(
                            "neutral",
                            "Waiting on files",
                            "Upload both CSVs to unlock the solve action.",
                        ),
                    )
                ),
            ],
        ),
        html.Div(
            id="results-container",
            style={"display": "none"},
            children=[
                html.Div(
                    className="section-wrap",
                    children=[
                        html.Div("Results", className="section-kicker"),
                        html.Div("Shape the final allocation", className="section-title"),
                    ],
                ),
                html.Div(id="tangency-metrics-container"),
                html.Div(
                    className="card-wrapper",
                    children=[
                        html.Div("Step 3B", className="card-kicker"),
                        html.Div("Capital Market Line controls", className="card-title"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Target portfolio volatility (%)", className="metric-label"),
                                        dcc.Slider(id="slider-target-vol", min=0, max=100, step=0.25, value=0, marks={}),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Borrowing rate (%)", className="metric-label"),
                                        dcc.Slider(id="slider-borrow", min=0, max=20, step=0.25, value=0),
                                    ],
                                    md=6,
                                ),
                            ]
                        ),
                    ],
                ),
                html.Div(id="final-metrics-container"),
                html.Div(
                    className="card-wrapper",
                    children=[
                        html.Div("Analysis", className="card-kicker"),
                        html.Div("Capital Market Line", className="card-title"),
                        dcc.Graph(id="graph-cml", figure=_empty_figure("Capital Market Line")),
                    ],
                ),
                html.Div(
                    className="card-wrapper",
                    children=[
                        html.Div("Allocation", className="card-kicker"),
                        html.Div("Final portfolio weights", className="card-title"),
                        dcc.Graph(id="graph-alloc", figure=_empty_figure("Final portfolio weights")),
                        html.Div(id="table-alloc-container"),
                    ],
                ),
            ],
        ),
        dcc.Store(id="store-prices"),
        dcc.Store(id="store-metrics"),
        dcc.Store(id="store-tangency"),
        dcc.Store(id="store-final"),
        dcc.Store(id="store-cml"),
    ],
)


@app.callback(
    [Output("status-prices", "children"), Output("store-prices", "data")],
    Input("upload-prices", "contents"),
    State("upload-prices", "filename"),
)
def update_prices_status(contents: str | None, filename: str | None) -> tuple[Any, Any]:
    return build_upload_status(contents, filename, "Price history")


@app.callback(
    [Output("status-metrics", "children"), Output("store-metrics", "data")],
    Input("upload-metrics", "contents"),
    State("upload-metrics", "filename"),
)
def update_metrics_status(contents: str | None, filename: str | None) -> tuple[Any, Any]:
    return build_upload_status(contents, filename, "Asset metrics")


@app.callback(
    [Output("btn-solve", "disabled"), Output("solve-message", "children")],
    [Input("store-prices", "data"), Input("store-metrics", "data")],
)
def toggle_solve_button(prices: Any, metrics: Any) -> tuple[bool, html.Div]:
    ready = prices is not None and metrics is not None
    if ready:
        return False, render_message(
            "ready",
            "Ready to solve",
            "Both files loaded successfully. Solve the tangency portfolio to unlock the interactive tuning stage.",
        )
    return True, render_message(
        "neutral",
        "Waiting on files",
        "Upload both CSVs to unlock the solve action.",
    )


@app.callback(
    [
        Output("store-tangency", "data"),
        Output("store-final", "data"),
        Output("store-cml", "data"),
        Output("results-container", "style"),
        Output("tangency-metrics-container", "children"),
        Output("final-metrics-container", "children"),
        Output("graph-cml", "figure"),
        Output("graph-alloc", "figure"),
        Output("table-alloc-container", "children"),
        Output("slider-target-vol", "min"),
        Output("slider-target-vol", "max"),
        Output("slider-target-vol", "value"),
        Output("slider-target-vol", "marks"),
        Output("slider-borrow", "min"),
        Output("slider-borrow", "max"),
        Output("slider-borrow", "value"),
        Output("solve-message", "children"),
    ],
    Input("btn-solve", "n_clicks"),
    [
        State("store-prices", "data"),
        State("store-metrics", "data"),
        State("slider-rf", "value"),
        State("radio-risk-model", "value"),
    ],
    prevent_initial_call=True,
)
def solve_and_render(
    n_clicks: int | None,
    prices_payload: dict[str, Any] | None,
    metrics_payload: dict[str, Any] | None,
    risk_free_rate_percent: float,
    risk_model_value: str,
):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    try:
        solve_result = solve_portfolio(
            prices_payload=prices_payload,
            metrics_payload=metrics_payload,
            risk_free_rate_percent=risk_free_rate_percent,
            risk_model_value=risk_model_value,
        )
        tangency = solve_result["tangency"]
        target_percent = solve_result["target_volatility_percent"]
        results = build_final_results(
            tangency=tangency,
            target_volatility_percent=target_percent,
            borrowing_rate_percent=risk_free_rate_percent,
            risk_free_rate_percent=risk_free_rate_percent,
        )
        slider_max = max(round(target_percent * 2.0, 2), 1.0)
        borrow_max = max(risk_free_rate_percent + 10.0, 5.0)

        return (
            tangency,
            results["final_store"],
            results["cml_store"],
            {"display": "block"},
            render_tangency_metrics(tangency),
            results["final_metrics"],
            results["cml_figure"],
            results["allocation_figure"],
            results["allocation_table"],
            0.0,
            slider_max,
            round(target_percent, 2),
            _build_slider_marks(slider_max),
            round(risk_free_rate_percent, 2),
            round(borrow_max, 2),
            round(risk_free_rate_percent, 2),
            render_message(
                "ready",
                "Portfolio solved",
                "The tangency portfolio is ready. Adjust target volatility and financing to explore the Capital Market Line.",
            ),
        )
    except Exception as exc:
        return (
            None,
            None,
            None,
            {"display": "block"},
            "",
            "",
            _empty_figure("Capital Market Line"),
            _empty_figure("Final portfolio weights"),
            render_message(
                "warning",
                "Solve failed",
                "We couldn't compute a portfolio from these files. Confirm the CSV structure and overlapping tickers, then try again.",
            ),
            0.0,
            100.0,
            0.0,
            {},
            0.0,
            20.0,
            0.0,
            render_message("warning", "Solve failed", str(exc)),
        )


@app.callback(
    [
        Output("store-final", "data", allow_duplicate=True),
        Output("store-cml", "data", allow_duplicate=True),
        Output("final-metrics-container", "children", allow_duplicate=True),
        Output("graph-cml", "figure", allow_duplicate=True),
        Output("graph-alloc", "figure", allow_duplicate=True),
        Output("table-alloc-container", "children", allow_duplicate=True),
    ],
    [
        Input("store-tangency", "data"),
        Input("slider-target-vol", "value"),
        Input("slider-borrow", "value"),
        Input("slider-rf", "value"),
    ],
    prevent_initial_call=True,
)
def update_final_metrics(
    tangency: dict[str, Any] | None,
    target_volatility_percent: float | None,
    borrowing_rate_percent: float | None,
    risk_free_rate_percent: float | None,
):
    if tangency is None or target_volatility_percent is None or borrowing_rate_percent is None or risk_free_rate_percent is None:
        raise dash.exceptions.PreventUpdate

    results = build_final_results(
        tangency=tangency,
        target_volatility_percent=target_volatility_percent,
        borrowing_rate_percent=borrowing_rate_percent,
        risk_free_rate_percent=risk_free_rate_percent,
    )
    return (
        results["final_store"],
        results["cml_store"],
        results["final_metrics"],
        results["cml_figure"],
        results["allocation_figure"],
        results["allocation_table"],
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
