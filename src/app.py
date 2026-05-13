import base64
import io
from typing import Dict, List, Optional, Tuple

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback_context

# Re-use existing engine logic
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

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Hybrid Quantamental Optimizer",
)

server = app.server

# --- Layout Components ---

def render_header():
    return html.Div(
        className="app-header",
        children=[
            html.Div("Portfolio Construction", className="app-kicker"),
            html.H1("Hybrid Quantamental Optimizer", className="app-title"),
            html.Div(
                "Move from raw market inputs to a target-risk allocation in one screen: "
                "upload the two source files, solve the risky portfolio, then tune the final cash mix.",
                className="app-copy",
            ),
        ],
    )

def render_upload_section(id_prefix: str, kicker: str, title: str, copy: str):
    return html.Div(
        className="card-wrapper",
        children=[
            html.Div(kicker, className="card-kicker"),
            html.Div(title, className="card-title"),
            html.Div(copy, className="card-copy"),
            dcc.Upload(
                id=f"upload-{id_prefix}",
                children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                className="upload-area",
                multiple=False,
            ),
            html.Div(id=f"status-{id_prefix}", className="mt-2"),
        ],
    )

app.layout = dbc.Container(
    className="container",
    children=[
        render_header(),
        
        # Setup Section
        html.Div(
            className="section-wrap",
            children=[
                html.Div("Setup", className="section-kicker"),
                html.Div("Load inputs and solve the risky portfolio", className="section-title"),
            ]
        ),
        
        dbc.Row([
            dbc.Col(
                render_upload_section(
                    "prices", 
                    "Step 1A", 
                    "Price history", 
                    "Upload one CSV with a date column followed by one column per ticker."
                ),
                md=6
            ),
            dbc.Col(
                render_upload_section(
                    "metrics", 
                    "Step 1B", 
                    "Asset metrics", 
                    "Upload the expected return, implied volatility, and optional weight bounds."
                ),
                md=6
            ),
        ]),
        
        html.Div(
            className="card-wrapper",
            children=[
                html.Div("Step 2", className="card-kicker"),
                html.Div("Assumptions and solve", className="card-title"),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Risk-free rate (%)", className="metric-label"),
                        dcc.Slider(
                            id="slider-rf",
                            min=-5, max=10, step=0.25, value=4.0,
                            marks={i: f"{i}%" for i in range(-5, 11, 5)},
                        ),
                    ], md=6),
                    dbc.Col([
                        html.Label("Risk model", className="metric-label"),
                        dcc.RadioItems(
                            id="radio-risk-model",
                            options=[
                                {"label": "Forward-Looking", "value": "FORWARD_LOOKING"},
                                {"label": "Historical", "value": "HISTORICAL"},
                            ],
                            value="FORWARD_LOOKING",
                            inline=True,
                            inputClassName="me-2",
                            labelClassName="me-4",
                        ),
                    ], md=6),
                ], className="mb-4"),
                
                html.Button(
                    "Solve risky portfolio",
                    id="btn-solve",
                    className="primary-button",
                    disabled=True,
                ),
                dcc.Loading(id="loading-solve", children=html.Div(id="solve-output")),
            ]
        ),
        
        # Results Section (initially hidden)
        html.Div(id="results-container", style={"display": "none"}),
        
        # Data storage
        dcc.Store(id="store-prices"),
        dcc.Store(id="store-metrics"),
        dcc.Store(id="store-tangency"),
    ]
)

# --- Helpers ---

def parse_contents(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return io.StringIO(decoded.decode("utf-8"))

def render_metric_card(label: str, value: str):
    return html.Div(
        className="metric-card",
        children=[
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
        ]
    )

def create_cml_figure(tangency, final_portfolio, cml_points, lending_rate):
    fig = go.Figure()

    # Asset points
    tickers = tangency.get("tickers", [])
    asset_rets = tangency.get("asset_returns", [])
    asset_vols = tangency.get("asset_vols", [])
    
    fig.add_trace(go.Scatter(
        x=asset_vols, y=asset_rets,
        mode="markers+text",
        name="Assets",
        text=tickers,
        textposition="top center",
        marker=dict(size=10, color="#7b8794", opacity=0.6)
    ))

    # CML Line (Lending)
    t_vol = tangency["volatility"]
    t_ret = tangency["expected_return"]
    
    fig.add_trace(go.Scatter(
        x=[0, t_vol], y=[lending_rate, t_ret],
        mode="lines",
        name="Lending CML",
        line=dict(color="#1b6c5c", width=3)
    ))

    # CML Line (Borrowing)
    if cml_points:
        b_vols = [p["volatility"] for p in cml_points if p["cash_weight"] < -1e-8]
        b_rets = [p["expected_return"] for p in cml_points if p["cash_weight"] < -1e-8]
        if b_vols:
            fig.add_trace(go.Scatter(
                x=[t_vol] + b_vols, y=[t_ret] + b_rets,
                mode="lines",
                name="Borrowing CML",
                line=dict(color="#c76b3f", width=3, dash="dash")
            ))

    # Tangency Point
    fig.add_trace(go.Scatter(
        x=[t_vol], y=[t_ret],
        mode="markers",
        name="Tangency Portfolio",
        marker=dict(size=15, color="#155548", symbol="circle", line=dict(width=2, color="black"))
    ))

    # Final Portfolio Point
    fig.add_trace(go.Scatter(
        x=[final_portfolio["volatility"]], y=[final_portfolio["expected_return"]],
        mode="markers",
        name="Final Portfolio",
        marker=dict(size=18, color="#c79b37", symbol="x", line=dict(width=3))
    ))

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

def create_allocation_figure(final_portfolio, tangency):
    tickers = tangency.get("tickers", [])
    weights = final_portfolio.get("weights", [])
    
    # Filter non-zero weights
    data = []
    cash_w = final_portfolio.get("cash_weight", 0)
    if abs(cash_w) > 0.0001:
        data.append({"Asset": "Cash" if cash_w > 0 else "Borrowing", "Weight": cash_w, "Type": "Cash"})
    
    for i, ticker in enumerate(tickers):
        w = weights[i]
        if abs(w) > 0.0001:
            data.append({"Asset": ticker, "Weight": w, "Type": "Risky"})
            
    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure()
        
    df = df.sort_values("Weight", ascending=False, key=abs)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Weight"],
        y=df["Asset"],
        orientation="h",
        marker_color=["#7b8794" if t == "Cash" else "#1b6c5c" for t in df["Type"]],
        text=[f"{w:.2%}" for w in df["Weight"]],
        textposition="auto",
    ))
    
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255, 251, 245, 0.94)",
        plot_bgcolor="rgba(255, 251, 245, 0.94)",
        font=dict(family="Urbanist", color="#1f2933"),
        xaxis=dict(title="Weight", tickformat=".1%", gridcolor="rgba(31, 41, 51, 0.12)"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=40, t=40, b=40),
        height=max(200, 40 * len(df)),
    )
    return fig

def render_allocation_table(final_portfolio, tangency, rf, borrow):
    tickers = tangency.get("tickers", [])
    weights = final_portfolio.get("weights", [])
    rets = tangency.get("asset_returns", [])
    vols = tangency.get("asset_vols", [])
    
    rows = []
    cash_w = final_portfolio.get("cash_weight", 0)
    if abs(cash_w) > 0.0001:
        label = "Cash" if cash_w > 0 else "Borrowing"
        rate = rf if cash_w > 0 else borrow
        rows.append(html.Tr([
            html.Td(label),
            html.Td(f"{cash_w:.2%}"),
            html.Td(f"{rate:.2%}"),
            html.Td("0.00%"),
        ]))
    
    for i, ticker in enumerate(tickers):
        w = weights[i]
        if abs(w) > 0.0001:
            rows.append(html.Tr([
                html.Td(ticker),
                html.Td(f"{w:.2%}"),
                html.Td(f"{rets[i]:.2%}"),
                html.Td(f"{vols[i]:.2%}"),
            ]))
            
    return html.Div(
        className="allocation-table-wrap",
        children=html.Table(
            className="allocation-table",
            children=[
                html.Thead(html.Tr([
                    html.Th("Asset"),
                    html.Th("Weight"),
                    html.Th("Expected Return"),
                    html.Th("Volatility"),
                ])),
                html.Tbody(rows)
            ]
        )
    )

# --- Callbacks ---

@app.callback(
    [Output("status-prices", "children"), Output("store-prices", "data")],
    Input("upload-prices", "contents"),
    State("upload-prices", "filename"),
)
def update_prices_status(contents, filename):
    if contents is None:
        return "", None
    return f"✅ {filename} loaded", contents

@app.callback(
    [Output("status-metrics", "children"), Output("store-metrics", "data")],
    Input("upload-metrics", "contents"),
    State("upload-metrics", "filename"),
)
def update_metrics_status(contents, filename):
    if contents is None:
        return "", None
    return f"✅ {filename} loaded", contents

@app.callback(
    Output("btn-solve", "disabled"),
    [Input("store-prices", "data"), Input("store-metrics", "data")],
)
def toggle_solve_button(prices, metrics):
    return prices is None or metrics is None

@app.callback(
    [
        Output("store-tangency", "data"),
        Output("results-container", "children"),
        Output("results-container", "style"),
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
def solve_and_render(n_clicks, prices_content, metrics_content, rf_pct, risk_model_val):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    
    try:
        rf = rf_pct / 100.0
        risk_model = RiskModel[risk_model_val]
        
        price_stream = parse_contents(prices_content)
        metric_stream = parse_contents(metrics_content)
        
        tangency = optimize_portfolio(
            price_source=price_stream,
            metric_source=metric_stream,
            risk_free_rate=rf,
            risk_model=risk_model,
            annualization_factor=252 if risk_model == RiskModel.HISTORICAL else None,
        )
        
        # Initial final portfolio is the tangency itself
        final_portfolio = target_portfolio(
            tangency_portfolio=tangency,
            target_volatility=tangency["volatility"],
            risk_free_rate=rf,
            borrowing_rate=rf,
        )
        
        cml_points = generate_cml(
            tangency_portfolio=tangency,
            risk_free_rate=rf,
            borrowing_rate=rf,
            max_volatility=tangency["volatility"] * 2.0,
            num_points=41,
        )
        
        results_layout = html.Div([
            html.Div(
                className="section-wrap",
                children=[
                    html.Div("Results", className="section-kicker"),
                    html.Div("Shape the final allocation", className="section-title"),
                ]
            ),
            
            # Step 3A: Tangency Info
            html.Div(className="card-wrapper", children=[
                html.Div("Step 3A", className="card-kicker"),
                html.Div("Tangency portfolio", className="card-title"),
                dbc.Row([
                    dbc.Col(render_metric_card("Expected return", f"{tangency['expected_return']:.2%}")),
                    dbc.Col(render_metric_card("Volatility", f"{tangency['volatility']:.2%}")),
                    dbc.Col(render_metric_card("Sharpe ratio", f"{tangency['sharpe_ratio']:.2f}")),
                ])
            ]),
            
            # Step 3B: Controls
            html.Div(className="card-wrapper", children=[
                html.Div("Step 3B", className="card-kicker"),
                html.Div("Capital Market Line controls", className="card-title"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Target portfolio volatility (%)", className="metric-label"),
                        dcc.Slider(
                            id="slider-target-vol",
                            min=0, max=tangency['volatility']*200, step=0.1,
                            value=tangency['volatility']*100,
                            marks={i: f"{i}%" for i in range(0, int(tangency['volatility']*200)+1, 10)},
                        ),
                    ], md=6),
                    dbc.Col([
                        html.Label("Borrowing rate (%)", className="metric-label"),
                        dcc.Slider(
                            id="slider-borrow",
                            min=rf_pct, max=rf_pct+10, step=0.25, value=rf_pct,
                        ),
                    ], md=6),
                ])
            ]),
            
            # Final Metrics
            html.Div(id="final-metrics-container"),
            
            # Visualization
            html.Div(className="card-wrapper", children=[
                html.Div("Analysis", className="card-kicker"),
                html.Div("Visual distribution", className="card-title"),
                dcc.Graph(id="graph-cml", figure=create_cml_figure(tangency, final_portfolio, cml_points, rf)),
            ]),
            
            # Allocation
            html.Div(className="card-wrapper", children=[
                html.Div("Allocation", className="card-kicker"),
                html.Div("Final portfolio weights", className="card-title"),
                dcc.Graph(id="graph-alloc", figure=create_allocation_figure(final_portfolio, tangency)),
                html.Div(id="table-alloc-container", children=render_allocation_table(final_portfolio, tangency, rf, rf)),
            ]),
        ])
        
        return tangency, results_layout, {"display": "block"}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, html.Div(f"Error: {str(e)}", className="status-card status--warning"), {"display": "block"}

@app.callback(
    [
        Output("final-metrics-container", "children"),
        Output("graph-cml", "figure"),
        Output("graph-alloc", "figure"),
        Output("table-alloc-container", "children"),
    ],
    [
        Input("slider-target-vol", "value"),
        Input("slider-borrow", "value"),
    ],
    [
        State("store-tangency", "data"),
        State("slider-rf", "value"),
    ],
    prevent_initial_call=True,
)
def update_final_metrics(target_vol_pct, borrow_pct, tangency, rf_pct):
    rf = rf_pct / 100.0
    borrow = borrow_pct / 100.0
    target_vol = target_vol_pct / 100.0
    
    final_portfolio = target_portfolio(
        tangency_portfolio=tangency,
        target_volatility=target_vol,
        risk_free_rate=rf,
        borrowing_rate=borrow,
    )
    
    cml_points = generate_cml(
        tangency_portfolio=tangency,
        risk_free_rate=rf,
        borrowing_rate=borrow,
        max_volatility=tangency["volatility"] * 2.0,
        num_points=41,
    )
    
    metrics = html.Div(className="card-wrapper", children=[
        html.Div("Step 3C", className="card-kicker"),
        html.Div("Final portfolio", className="card-title"),
        dbc.Row([
            dbc.Col(render_metric_card("Expected return", f"{final_portfolio['expected_return']:.2%}")),
            dbc.Col(render_metric_card("Volatility", f"{final_portfolio['volatility']:.2%}")),
            dbc.Col(render_metric_card("Financing", f"{final_portfolio['cash_weight']:.2%}")),
        ])
    ])
    
    fig_cml = create_cml_figure(tangency, final_portfolio, cml_points, rf)
    fig_alloc = create_allocation_figure(final_portfolio, tangency)
    table_alloc = render_allocation_table(final_portfolio, tangency, rf, borrow)
    
    return metrics, fig_cml, fig_alloc, table_alloc

if __name__ == "__main__":
    app.run(debug=True, port=8050)
