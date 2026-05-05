"""Hybrid Quantamental Optimizer Streamlit application."""

from html import escape
from io import StringIO

import streamlit as st

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


def _inject_styles() -> None:
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Gelasio:wght@500;600;700&family=Urbanist:wght@400;500;600;700&display=swap');

    :root {
        --bg: #f6f1e7;
        --panel: rgba(255, 255, 251, 0.94);
        --panel-strong: rgba(255, 255, 251, 0.98);
        --ink: #1f242a;
        --muted: #625d57;
        --line: rgba(31, 36, 42, 0.10);
        --accent: #1b6c5c;
        --accent-soft: rgba(27, 108, 92, 0.08);
        --warn: #af6a2d;
        --warn-soft: rgba(175, 106, 45, 0.10);
        --error: #a44646;
        --error-soft: rgba(164, 70, 70, 0.10);
        --shadow: 0 18px 40px rgba(78, 63, 39, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(27, 108, 92, 0.08), transparent 26%),
            linear-gradient(180deg, #f2ebdf 0%, var(--bg) 24%, #fbf8f3 100%);
        color: var(--ink);
        font-family: "Urbanist", "Segoe UI", sans-serif;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] * {
        font-family: "Urbanist", "Segoe UI", sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(246, 241, 231, 0.82);
        backdrop-filter: blur(10px);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 1.6rem;
        padding-bottom: 2.6rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel);
        border-radius: 22px;
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        padding: 1.1rem 1.15rem 1.2rem;
    }

    .app-header {
        background: var(--panel-strong);
        border: 1px solid var(--line);
        border-radius: 28px;
        box-shadow: var(--shadow);
        padding: 1.35rem 1.4rem 1.45rem;
        margin-bottom: 1.15rem;
    }

    .app-kicker,
    .section-kicker,
    .card-kicker,
    .stage-step,
    .metric-label {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.74rem;
        font-weight: 700;
    }

    .app-kicker,
    .section-kicker,
    .card-kicker,
    .stage-step {
        color: #936132;
    }

    .app-title,
    .section-title,
    .card-title,
    .metric-value {
        font-family: "Gelasio", "Georgia", serif;
        color: var(--ink);
        letter-spacing: -0.02em;
    }

    .app-title {
        font-size: clamp(2.3rem, 4vw, 3.6rem);
        line-height: 1.02;
        margin: 0.45rem 0 0.75rem;
        max-width: none;
    }

    .app-copy,
    .section-copy,
    .card-copy,
    .stage-copy,
    .status-copy,
    .file-note,
    .inline-note {
        color: var(--muted);
        line-height: 1.6;
    }

    .app-copy {
        max-width: 72ch;
        margin-bottom: 0;
    }

    .section-wrap {
        margin: 1.35rem 0 0.95rem;
    }

    .section-title {
        font-size: 1.7rem;
        line-height: 1.08;
        margin: 0.3rem 0 0.4rem;
    }

    .section-copy {
        max-width: 58ch;
        margin-bottom: 0;
    }

    .stage-card {
        background: var(--panel-strong);
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: var(--shadow);
        padding: 1rem 1rem 1.05rem;
        display: flex;
        flex-direction: column;
        min-height: 0;
    }

    .stage-card.stage--active {
        border-color: rgba(27, 108, 92, 0.32);
        background: linear-gradient(180deg, rgba(27, 108, 92, 0.08), rgba(255, 255, 251, 0.98));
    }

    .stage-card.stage--complete {
        border-color: rgba(27, 108, 92, 0.22);
        background: rgba(243, 250, 247, 0.96);
    }

    .stage-card.stage--pending {
        background: rgba(255, 255, 251, 0.82);
    }

    .stage-title,
    .card-title {
        font-size: 1.2rem;
        margin: 0.45rem 0 0.35rem;
    }

    .card-copy,
    .stage-copy {
        font-size: 0.94rem;
    }

    .card-copy {
        margin-bottom: 0.2rem;
    }

    .inline-note {
        background: rgba(243, 237, 227, 0.86);
        border: 1px solid rgba(31, 36, 42, 0.09);
        border-radius: 16px;
        padding: 0.85rem 0.95rem;
        font-size: 0.92rem;
        margin: 0.85rem 0;
    }

    .file-ready,
    .status-card {
        border-radius: 18px;
        padding: 0.95rem 1rem;
        border: 1px solid var(--line);
    }

    .file-ready {
        background: rgba(27, 108, 92, 0.07);
        border-color: rgba(27, 108, 92, 0.14);
        margin-top: 0.95rem;
    }

    .file-ready strong,
    .status-title {
        display: block;
        color: var(--ink);
        font-size: 0.96rem;
        margin-bottom: 0.2rem;
    }

    .status-card.status--neutral {
        background: rgba(243, 237, 227, 0.90);
    }

    .status-card.status--warning {
        background: var(--warn-soft);
        border-color: rgba(175, 106, 45, 0.18);
    }

    .status-card.status--ready,
    .status-card.status--success {
        background: var(--accent-soft);
        border-color: rgba(27, 108, 92, 0.18);
    }

    .status-card.status--error {
        background: var(--error-soft);
        border-color: rgba(164, 70, 70, 0.18);
    }

    .control-label {
        color: var(--ink);
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .metric-card {
        background: rgba(255, 255, 251, 0.96);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.95rem 1rem 1.05rem;
        min-height: 6.75rem;
    }

    .metric-label {
        color: #6c675f;
        margin-bottom: 0.8rem;
    }

    .metric-value {
        font-size: 2rem;
        line-height: 1.05;
    }

    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(248, 245, 239, 0.88) !important;
        background: rgba(248, 245, 239, 0.88) !important;
        border: 1.5px dashed rgba(27, 108, 92, 0.26) !important;
        border-radius: 18px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploadDropzone"] * {
        color: var(--muted) !important;
        fill: var(--muted) !important;
    }

    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    label[data-testid="stWidgetLabel"] {
        color: var(--ink) !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stSlider"] p,
    div[data-testid="stSlider"] span,
    div[data-testid="stSlider"] label,
    div[data-testid="stRadio"] p,
    div[data-testid="stRadio"] span,
    div[data-testid="stRadio"] label {
        color: var(--ink) !important;
        opacity: 1 !important;
    }

    div[data-testid="stFileUploader"] section small,
    div[data-testid="stFileUploader"] section span,
    div[data-testid="stFileUploader"] section p {
        color: var(--muted) !important;
        opacity: 1 !important;
    }

    button[kind="primary"] {
        border-radius: 999px;
        border: 0;
        background: linear-gradient(135deg, #1d6e5d, #0f5647);
        box-shadow: 0 16px 30px rgba(27, 108, 92, 0.20);
        min-height: 3.15rem;
        font-weight: 700;
    }

    button[kind="secondary"] {
        border-radius: 999px;
        background: #f8f6f1;
        color: var(--ink);
        border: 1px solid rgba(31, 36, 42, 0.12);
    }

    div[data-testid="stDownloadButton"] button,
    div[data-testid="stDownloadButton"] button:hover {
        background: #f8f6f1 !important;
        color: var(--ink) !important;
        border: 1px solid rgba(31, 36, 42, 0.12) !important;
        box-shadow: none !important;
    }

    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] * {
        color: var(--muted);
        opacity: 1;
    }

    div[data-testid="stCodeBlock"] pre,
    div[data-testid="stCodeBlock"] code,
    div[data-testid="stCode"] pre,
    div[data-testid="stCode"] code {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
    }

    .footer-note {
        color: var(--muted);
        text-align: center;
        font-size: 0.88rem;
        margin-top: 2rem;
    }

    .footer-note a {
        color: var(--accent);
        text-decoration: none;
        font-weight: 700;
    }

    @media (max-width: 980px) {
        .app-title {
            max-width: none;
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
        min_value=0.0,
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
    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    optimize_button = st.button(
        "Solve risky portfolio",
        type="primary",
        use_container_width=True,
        disabled=not setup_status.ready_to_solve,
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
        target_vol_pct = st.slider(
            "Target portfolio volatility (%)",
            min_value=0.0,
            max_value=max_target_vol_pct,
            value=default_target_pct,
            step=0.1,
            format="%.1f%%",
            help="Below the tangency portfolio you are on the lending segment. Above it you move onto the borrowing segment.",
        )
        rate_col_a, rate_col_b = st.columns(2, gap="large")
        with rate_col_a:
            st.slider(
                "Lending rate (%)",
                min_value=0.0,
                max_value=15.0,
                value=lending_rate_pct,
                step=0.25,
                format="%.2f%%",
                disabled=True,
                help="The lending rate is fixed from your assumptions in Step 2.",
            )
        with rate_col_b:
            borrowing_rate_pct = st.slider(
                "Borrowing rate (%)",
                min_value=lending_rate_pct,
                max_value=15.0,
                value=max(lending_rate_pct, current_borrowing_pct),
                step=0.25,
                format="%.2f%%",
                help="Equal lending and borrowing rates produce a straight Capital Market Line.",
            )
        borrowing_rate = borrowing_rate_pct / 100.0
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
    final_portfolio = target_portfolio(
        tangency_portfolio=tangency,
        target_volatility=target_vol_pct / 100.0,
        risk_free_rate=st.session_state.lending_rate,
        borrowing_rate=borrowing_rate,
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
