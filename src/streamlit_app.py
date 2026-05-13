"""Hybrid Quantamental Optimizer Streamlit application."""

from html import escape
from io import StringIO

import streamlit as st

try:
    from components.results_display import render_results
    from components.workflow_state import (
        build_run_summary,
        build_setup_status,
        build_workflow_stages,
    )
    from engine.portfolio_engine_pandas import (
        generate_cml,
        optimize_portfolio,
        target_portfolio,
    )
    from engine.risk_pandas import RiskModel
except ModuleNotFoundError:  # pragma: no cover - used by AppTest package-style imports
    from src.components.results_display import render_results
    from src.components.workflow_state import (
        build_run_summary,
        build_setup_status,
        build_workflow_stages,
    )
    from src.engine.portfolio_engine_pandas import (
        generate_cml,
        optimize_portfolio,
        target_portfolio,
    )
    from src.engine.risk_pandas import RiskModel


def _inject_styles() -> None:
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Gelasio:wght@500;600;700&family=Urbanist:wght@400;500;600;700&display=swap');

    :root {
        --bg: #f7f3ea;
        --bg-accent: #efe7d8;
        --panel: rgba(255, 251, 245, 0.94);
        --panel-strong: #fffdf8;
        --ink: #1f2933;
        --muted: #5f6c76;
        --line: rgba(31, 41, 51, 0.12);
        --accent: #1b6c5c;
        --accent-strong: #155548;
        --accent-soft: rgba(27, 108, 92, 0.1);
        --warning: #c76b3f;
        --warning-soft: rgba(199, 107, 63, 0.12);
        --neutral-soft: rgba(95, 108, 118, 0.1);
        --shadow: 0 18px 36px rgba(31, 41, 51, 0.08);
        --radius-lg: 20px;
        --radius-md: 16px;
        --space-card: 1.5rem;
    }

    html, body, [class*="stApp"] {
        font-family: "Urbanist", sans-serif;
        background: var(--bg) !important;
        color: var(--ink) !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
        background-color: var(--bg) !important;
        background-image:
            radial-gradient(circle at top left, rgba(255, 255, 255, 0.9), transparent 26%),
            linear-gradient(180deg, var(--bg), var(--bg-accent)) !important;
        color: var(--ink) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 243, 234, 0.92) !important;
    }

    section.main > div,
    [data-testid="stMain"] {
        background: transparent !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1120px !important;
        margin: 0 auto !important;
        padding-top: 2.5rem;
        padding-bottom: 5rem;
    }

    .app-header, .section-wrap { margin-bottom: 2rem; }

    .app-kicker, .section-kicker, .card-kicker, .stage-step, .metric-label {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.74rem;
        font-weight: 700;
        color: var(--accent) !important;
        margin-bottom: 0.5rem;
    }

    .app-title, .section-title, .card-title, .stage-title, .metric-value {
        font-family: "Gelasio", serif;
        color: var(--ink) !important;
        letter-spacing: -0.02em;
        margin-top: 0;
    }

    .app-title { font-size: clamp(2.7rem, 6vw, 3.8rem); line-height: 1.02; }
    .section-title { font-size: clamp(1.8rem, 4vw, 2.3rem); }
    .card-title, .stage-title { font-size: 1.35rem; }

    .app-copy, .section-copy, .card-copy, .stage-copy, .status-copy {
        color: var(--muted) !important;
        line-height: 1.65;
        font-size: 1rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow);
        padding: var(--space-card) !important;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        overflow: hidden;
    }

    .stage-card {
        background: var(--panel-strong);
        border: 1px solid var(--line);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        height: 100%;
    }

    .stage-card.stage--active {
        border-color: var(--accent);
        background-image: linear-gradient(180deg, var(--accent-soft), transparent);
    }

    .status-card {
        border-radius: var(--radius-md);
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        margin: 1rem 0;
        background: var(--panel-strong);
    }

    .status-title {
        color: var(--ink);
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .status-card.status--neutral { background: var(--neutral-soft); }
    .status-card.status--ready {
        background: var(--accent-soft);
        border-color: var(--accent);
    }
    .status-card.status--warning {
        background: var(--warning-soft);
        border-color: var(--warning);
    }

    div.stButton { margin: 1rem 0 0.5rem; }

    div.stButton > button[kind="primary"],
    [data-testid="stBaseButton-primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {
        border-radius: 999px !important;
        background: var(--accent) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
        width: 100% !important;
        max-width: 320px !important;
        margin: 0 auto !important;
        display: block !important;
        height: 3.35rem !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        border: 1px solid transparent !important;
        box-shadow: 0 12px 24px rgba(27, 108, 92, 0.22) !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background: var(--accent-strong) !important;
        border-color: var(--accent-strong) !important;
    }

    div.stButton > button[kind="primary"]:disabled,
    [data-testid="stBaseButton-primary"]:disabled,
    div.stButton > button:disabled {
        background: var(--panel-strong) !important;
        border: 1px solid var(--line) !important;
        color: var(--muted) !important;
        -webkit-text-fill-color: var(--muted) !important;
        opacity: 1 !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }

    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stRadio"] label,
    div[data-testid="stSlider"] label {
        color: var(--ink) !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploader"],
    [data-testid="stCodeBlock"] {
        border-radius: var(--radius-md);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: var(--panel-strong) !important;
        border: 1px solid var(--line) !important;
        color: var(--ink) !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: var(--ink) !important;
    }

    [data-testid="stFileUploaderDropzone"] button,
    div.stButton > button[kind="secondary"] {
        background: var(--panel-strong) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 999px !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover,
    div.stButton > button[kind="secondary"]:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],
    .st-emotion-cache-ue6h4q {
        color: var(--muted);
    }

    .file-ready,
    .inline-note,
    .metric-card {
        background: var(--panel-strong);
        border: 1px solid var(--line);
        border-radius: var(--radius-md);
    }

    .file-ready,
    .inline-note {
        padding: 0.9rem 1rem;
        color: var(--ink);
    }

    .file-note,
    .inline-note {
        color: var(--muted);
        line-height: 1.55;
    }

    .metric-card {
        padding: 1rem 1.1rem;
        height: 100%;
    }

    .metric-value {
        font-size: 1.9rem;
        line-height: 1.1;
        margin-top: 0.1rem;
    }

    .footer-note {
        color: var(--muted);
        text-align: center;
        font-size: 0.95rem;
        padding: 1rem 0 0;
    }

    .footer-note a {
        color: var(--accent);
        text-decoration: none;
    }

    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            padding-top: 1.5rem;
            padding-bottom: 4rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 1.15rem !important;
        }
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-kicker">Portfolio Construction</div>
            <h1 class="app-title">Hybrid Quantamental Optimizer</h1>
            <div class="app-copy">
                Move from raw market inputs to a target-risk allocation in one
                screen: upload the two source files, solve the risky portfolio,
                then tune the final cash mix.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-wrap">
            <div class="section-kicker">{escape(kicker)}</div>
            <div class="section-title">{escape(title)}</div>
            <div class="section-copy">{escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_workflow_stages(ready_to_solve: bool, optimization_complete: bool) -> None:
    stages = build_workflow_stages(
        ready_to_solve=ready_to_solve,
        optimization_complete=optimization_complete,
    )
    stage_cols = st.columns(len(stages), gap="medium")
    for index, (column, stage) in enumerate(zip(stage_cols, stages), start=1):
        with column:
            st.markdown(
                f"""
                <div class="stage-card stage--{escape(stage.status)}">
                    <div class="stage-step">Step {index}</div>
                    <div class="stage-title">{escape(stage.title)}</div>
                    <div class="stage-copy">{escape(stage.description)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_card_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="card-kicker">{escape(kicker)}</div>
        <div class="card-title">{escape(title)}</div>
        <div class="card-copy">{escape(copy)}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_control_label(label: str) -> None:
    st.markdown(
        f"<div class='control-label'>{escape(label)}</div>",
        unsafe_allow_html=True,
    )


def _render_status_card(title: str, body: str, tone: str) -> None:
    st.markdown(
        f"""
        <div class="status-card status--{escape(tone)}">
            <div class="status-title">{escape(title)}</div>
            <div class="status-copy">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_file_ready(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="file-ready">
            <strong>{escape(title)}</strong>
            <div class="file-note">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sample_disclosure(state_key: str) -> bool:
    visible = bool(st.session_state.get(state_key, False))
    label = "Hide sample format" if visible else "Show sample format"
    if st.button(label, key=f"{state_key}_button", type="secondary"):
        st.session_state[state_key] = not visible
        st.rerun()
    return bool(st.session_state.get(state_key, False))


def _risk_model_note(risk_model: RiskModel) -> str:
    if risk_model == RiskModel.HISTORICAL:
        return "Historical mode annualizes sample covariance using 252 trading days."
    return (
        "Forward-looking mode preserves historical correlation structure and uses "
        "implied volatility as the risk input."
    )


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def _financing_metric(cash_weight: float) -> tuple[str, str]:
    if cash_weight < -1e-8:
        return ("Borrowing", _fmt_pct(abs(cash_weight)))
    return ("Cash", _fmt_pct(max(cash_weight, 0.0)))


def _file_name(file_obj) -> str | None:
    return file_obj.name if file_obj is not None else None


st.set_page_config(
    page_title="Hybrid Quantamental Optimizer",
    page_icon="HQ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_inject_styles()

if "optimization_complete" not in st.session_state:
    st.session_state.optimization_complete = False
if "tangency_portfolio" not in st.session_state:
    st.session_state.tangency_portfolio = None
if "cml_points" not in st.session_state:
    st.session_state.cml_points = None
if "lending_rate" not in st.session_state:
    st.session_state.lending_rate = 0.04
if "borrowing_rate" not in st.session_state:
    st.session_state.borrowing_rate = 0.04
if "risk_model_label" not in st.session_state:
    st.session_state.risk_model_label = "Forward-Looking"
if "solve_error" not in st.session_state:
    st.session_state.solve_error = None
if "error_signature" not in st.session_state:
    st.session_state.error_signature = None
if "solved_signature" not in st.session_state:
    st.session_state.solved_signature = None

_render_header()
stage_placeholder = st.empty()

_render_section_header(
    "Setup",
    "Load inputs and solve the risky portfolio",
    "Upload the price history and asset metrics files, set the model assumptions, then run one solve to unlock the final allocation workflow.",
)

with st.container(border=True):
    _render_card_intro(
        "Step 1A",
        "Price history",
        "Upload one CSV with a date column followed by one column per ticker.",
    )
    if _render_sample_disclosure("show_price_sample"):
        st.code(
            """date,AAPL,GOOG,MSFT
2023-01-31,150.23,105.44,280.50
2023-02-28,152.11,108.22,285.33""",
            language="csv",
        )
    prices_file = st.file_uploader(
        "Choose prices.csv",
        type=["csv"],
        key="prices_upload",
    )
    if prices_file is not None:
        _render_file_ready(
            "Price file loaded",
            f"{prices_file.name} is ready for covariance construction.",
        )

with st.container(border=True):
    _render_card_intro(
        "Step 1B",
        "Asset metrics",
        "Upload the expected return, implied volatility, and optional weight bounds.",
    )
    if _render_sample_disclosure("show_metrics_sample"):
        st.code(
            """ticker,expected_return,implied_volatility,min_weight,max_weight
AAPL,0.12,0.25,0.0,1.0
GOOG,0.15,0.28,0.0,1.0""",
            language="csv",
        )
    metrics_file = st.file_uploader(
        "Choose metrics.csv",
        type=["csv"],
        key="metrics_upload",
    )
    if metrics_file is not None:
        _render_file_ready(
            "Metrics file loaded",
            f"{metrics_file.name} is ready for returns, volatility, and bounds.",
        )

with st.container(border=True):
    _render_card_intro(
        "Step 2",
        "Assumptions and solve",
        "Set the risk-free rate and risk model, then build the tangency portfolio.",
    )
    lending_rate_pct = st.slider(
        "Risk-free rate (%)",
        min_value=-5.0,
        max_value=10.0,
        value=float(st.session_state.lending_rate * 100.0),
        step=0.25,
        format="%.2f%%",
    )
    lending_rate = lending_rate_pct / 100.0

    risk_model_label = st.radio(
        "Risk model",
        ["Forward-Looking", "Historical"],
        index=0
        if st.session_state.risk_model_label == "Forward-Looking"
        else 1,
        horizontal=True,
    )

    risk_model = (
        RiskModel.FORWARD_LOOKING
        if risk_model_label == "Forward-Looking"
        else RiskModel.HISTORICAL
    )
    st.caption(_risk_model_note(risk_model))

prices_name = _file_name(prices_file)
metrics_name = _file_name(metrics_file)
files_ready = prices_file is not None and metrics_file is not None
solve_signature = (
    prices_name,
    metrics_name,
    lending_rate,
    risk_model_label,
)

if (
    st.session_state.optimization_complete
    and st.session_state.solved_signature != solve_signature
):
    st.session_state.optimization_complete = False
    st.session_state.tangency_portfolio = None
    st.session_state.cml_points = None

if st.session_state.error_signature is not None and (
    st.session_state.error_signature != solve_signature
):
    st.session_state.solve_error = None
    st.session_state.error_signature = None

solve_complete = bool(
    st.session_state.optimization_complete
    and st.session_state.tangency_portfolio is not None
)
setup_status = build_setup_status(
    prices_name,
    metrics_name,
    optimization_complete=solve_complete,
    error_message=st.session_state.solve_error,
)

with stage_placeholder.container():
    _render_workflow_stages(
        ready_to_solve=setup_status.ready_to_solve,
        optimization_complete=solve_complete,
    )

with st.container(border=True):
    _render_card_intro(
        "Status",
        "Current workflow state",
        "Upload both files and set your assumptions to enable the solver.",
    )
    if files_ready:
        st.markdown(
            f"<div class='inline-note'>{escape(build_run_summary(prices_name, metrics_name, lending_rate=lending_rate, borrowing_rate=lending_rate, risk_model_label=risk_model_label))}</div>",
            unsafe_allow_html=True,
        )
    _render_status_card(
        setup_status.title,
        setup_status.message,
        setup_status.tone,
    )
    optimize_button = False
    if not solve_complete:
        optimize_button = st.button(
            "Solve risky portfolio",
            type="primary",
            width="stretch",
            disabled=not setup_status.ready_to_solve,
            key="solve_button",
        )

if optimize_button:
    with st.spinner("Solving portfolio and generating the capital market line..."):
        try:
            prices_file.seek(0)
            metrics_file.seek(0)
            price_stream = StringIO(prices_file.getvalue().decode("utf-8"))
            metric_stream = StringIO(metrics_file.getvalue().decode("utf-8"))

            tangency = optimize_portfolio(
                price_source=price_stream,
                metric_source=metric_stream,
                risk_free_rate=lending_rate,
                risk_model=risk_model,
                annualization_factor=252
                if risk_model == RiskModel.HISTORICAL
                else None,
            )

            st.session_state.tangency_portfolio = tangency
            st.session_state.cml_points = None
            st.session_state.lending_rate = lending_rate
            st.session_state.borrowing_rate = max(
                st.session_state.borrowing_rate,
                lending_rate,
            )
            st.session_state.risk_model_label = risk_model_label
            st.session_state.solve_error = None
            st.session_state.error_signature = None
            st.session_state.solved_signature = solve_signature
            st.session_state.optimization_complete = True
            st.rerun()
        except Exception as exc:
            st.session_state.optimization_complete = False
            st.session_state.tangency_portfolio = None
            st.session_state.cml_points = None
            st.session_state.solve_error = str(exc)
            st.session_state.error_signature = solve_signature

if solve_complete and st.session_state.tangency_portfolio is not None:
    tangency = st.session_state.tangency_portfolio

    _render_section_header(
        "Results",
        "Shape the final allocation",
        "The risky portfolio is solved. Adjust the target risk level, then review the cash-adjusted allocation and visual analysis below.",
    )

    with st.container(border=True):
        _render_card_intro(
            "Step 3A",
            "Tangency portfolio",
            "This is the highest-Sharpe risky portfolio before any cash is blended in.",
        )
        tangency_a, tangency_b, tangency_c = st.columns(3, gap="large")
        with tangency_a:
            _render_metric_card(
                "Expected return",
                _fmt_pct(float(tangency["expected_return"])),
            )
        with tangency_b:
            _render_metric_card(
                "Volatility",
                _fmt_pct(float(tangency["volatility"])),
            )
        with tangency_c:
            _render_metric_card(
                "Sharpe ratio",
                f"{float(tangency['sharpe_ratio']):.2f}",
            )

    max_vol_pct = float(tangency["volatility"]) * 100.0
    default_target_pct = min(10.0, max_vol_pct)
    current_borrowing_pct = max(
        lending_rate_pct,
        float(st.session_state.borrowing_rate * 100.0),
    )

    with st.container(border=True):
        _render_card_intro(
            "Step 3B",
            "Capital Market Line controls",
            "Set the target volatility and rates that together determine where the portfolio sits on the Capital Market Line.",
        )
        tangency_vol_pct = float(tangency["volatility"]) * 100.0
        max_target_vol_pct = tangency_vol_pct * 2.0
        
        control_col_a, control_col_b = st.columns(2, gap="large")
        
        with control_col_a:
            target_vol_pct = st.slider(
                "Target portfolio volatility (%)",
                min_value=0.0,
                max_value=max_target_vol_pct,
                value=default_target_pct,
                step=0.1,
                format="%.1f%%",
                help="Below the tangency portfolio you are on the lending segment. Above it you move onto the borrowing segment.",
            )
            
        with control_col_b:
            current_borrowing_pct = float(st.session_state.borrowing_rate * 100.0)
            
            lending_rate_pct, borrowing_rate_pct = st.slider(
                "Rates (Lending & Borrowing) (%)",
                min_value=-5.0,
                max_value=15.0,
                value=(
                    float(st.session_state.lending_rate * 100.0),
                    float(max(st.session_state.lending_rate * 100.0, current_borrowing_pct))
                ),
                step=0.25,
                format="%.2f%%",
                help="The lower point is the lending rate (cash yield). The upper point is the borrowing rate (leverage cost).",
            )
        
        borrowing_rate = borrowing_rate_pct / 100.0
        st.session_state.lending_rate = lending_rate_pct / 100.0
        st.session_state.borrowing_rate = borrowing_rate
        if target_vol_pct <= tangency_vol_pct + 1e-8:
            _render_status_card(
                "Lending portfolio",
                "The final allocation holds cash at the lending rate alongside the tangency portfolio.",
                "ready",
            )
        else:
            _render_status_card(
                "Borrowing portfolio",
                "The final allocation uses leverage beyond the tangency portfolio, so the borrowing rate applies.",
                "warning",
            )

    final_portfolio = target_portfolio(
        tangency_portfolio=tangency,
        target_volatility=target_vol_pct / 100.0,
        risk_free_rate=st.session_state.lending_rate,
        borrowing_rate=borrowing_rate,
    )

    with st.container(border=True):
        _render_card_intro(
            "Step 3C",
            "Final portfolio",
            "These metrics reflect the tangency portfolio scaled along the lending or borrowing segment of the Capital Market Line.",
        )
        financing_label, financing_value = _financing_metric(
            float(final_portfolio["cash_weight"])
        )
        final_a, final_b, final_c = st.columns(3, gap="large")
        with final_a:
            _render_metric_card(
                "Expected return",
                _fmt_pct(float(final_portfolio["expected_return"])),
            )
        with final_b:
            _render_metric_card(
                "Volatility",
                _fmt_pct(float(final_portfolio["volatility"])),
            )
        with final_c:
            _render_metric_card(
                financing_label,
                financing_value,
            )


    cml_points = generate_cml(
        tangency_portfolio=tangency,
        risk_free_rate=st.session_state.lending_rate,
        borrowing_rate=borrowing_rate,
        max_volatility=tangency["volatility"] * 2.0,
        num_points=41,
    )

    render_results(
        tangency=tangency,
        final_portfolio=final_portfolio,
        cml_points=cml_points,
        lending_rate=st.session_state.lending_rate,
        borrowing_rate=borrowing_rate,
    )

else:
    with st.container(border=True):
        _render_card_intro(
            "Step 3",
            "Results unlock after solve",
            "Solve the risky portfolio to reveal the target-volatility control, final allocation, visual analysis, and export.",
        )

st.markdown(
    """
    <div class="footer-note">
        Built for deliberate portfolio construction.
        <a href="https://github.com/bsachart/hybrid-quantamental-optimizer" target="_blank">View source</a>
    </div>
    """,
    unsafe_allow_html=True,
)
