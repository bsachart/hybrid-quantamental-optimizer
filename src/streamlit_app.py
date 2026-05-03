"""
Hybrid Quantamental Optimizer - Streamlit application.
"""

from html import escape
from io import StringIO

import streamlit as st

from components.results_display import render_results
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
        --bg: #f7f4ee;
        --bg-alt: #f1ece3;
        --panel: rgba(255, 255, 252, 0.92);
        --panel-strong: rgba(255, 255, 252, 0.98);
        --ink: #22252b;
        --muted: #625d56;
        --accent: #166c59;
        --accent-soft: #deebe5;
        --signal: #c76b3f;
        --line: rgba(34, 37, 43, 0.10);
        --shadow: 0 18px 48px rgba(91, 70, 42, 0.08);
    }

    .stApp {
        background: linear-gradient(180deg, var(--bg-alt) 0%, var(--bg) 18%, #fbf9f5 100%);
        color: var(--ink);
        font-family: "Urbanist", "Segoe UI", sans-serif;
        font-optical-sizing: auto;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] * {
        font-family: "Urbanist", "Segoe UI", sans-serif;
        font-optical-sizing: auto;
    }

    .material-icons,
    .material-icons-round,
    .material-icons-sharp,
    .material-icons-outlined,
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-symbols-sharp {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-weight: normal;
        font-style: normal;
        letter-spacing: normal;
        text-transform: none;
        white-space: nowrap;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 244, 238, 0.88);
        backdrop-filter: blur(10px);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1600px;
        padding-top: 2.1rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3,
    .hero-title,
    .brief-title,
    .section-title {
        font-family: "Gelasio", "Georgia", serif;
        letter-spacing: -0.02em;
        color: var(--ink);
    }

    .hero-shell {
        background: var(--panel-strong);
        border: 1px solid rgba(34, 37, 43, 0.10);
        border-radius: 22px;
        box-shadow: var(--shadow);
        padding: 2.1rem 2.2rem;
        min-height: 100%;
    }

    .eyebrow {
        font-family: "Urbanist", "Segoe UI", sans-serif;
        color: var(--signal);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: clamp(2.8rem, 4vw, 4.2rem);
        line-height: 1.04;
        margin: 0;
        max-width: 11ch;
        text-wrap: balance;
    }

    .hero-copy {
        font-family: "Urbanist", "Segoe UI", sans-serif;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.6;
        max-width: 48ch;
        margin-top: 1rem;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 1.4rem;
    }

    .badge {
        font-family: "Urbanist", "Segoe UI", sans-serif;
        border-radius: 999px;
        background: rgba(22, 108, 89, 0.10);
        color: var(--accent);
        border: 1px solid rgba(22, 108, 89, 0.14);
        font-size: 0.84rem;
        font-weight: 600;
        padding: 0.4rem 0.8rem;
    }

    .brief-card {
        background: var(--panel);
        border: 1px solid rgba(34, 37, 43, 0.10);
        border-radius: 22px;
        box-shadow: var(--shadow);
        padding: 1.65rem 1.6rem;
        min-height: 100%;
    }

    .brief-title {
        color: var(--ink);
        font-family: "Gelasio", "Georgia", serif;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }

    .brief-line {
        border-top: 1px solid rgba(34, 37, 43, 0.10);
        padding-top: 0.95rem;
        margin-top: 0.95rem;
    }

    .brief-line strong {
        display: block;
        color: var(--ink);
        margin-bottom: 0.2rem;
        font-size: 0.96rem;
    }

    .brief-line span {
        font-family: "Urbanist", "Segoe UI", sans-serif;
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .section-wrap {
        margin-top: 2rem;
    }

    .section-kicker {
        font-family: "Urbanist", "Segoe UI", sans-serif;
        color: var(--signal);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .section-title {
        font-size: 1.8rem;
        line-height: 1.1;
        margin-bottom: 0.45rem;
    }

    .section-copy {
        font-family: "Urbanist", "Segoe UI", sans-serif;
        color: var(--muted);
        line-height: 1.65;
        max-width: 60ch;
        margin-bottom: 1rem;
    }

    .field-note, .status-note {
        font-family: "Urbanist", "Segoe UI", sans-serif;
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .panel-card, [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel);
        border-radius: 20px;
        border: 1px solid rgba(34, 37, 43, 0.10);
        box-shadow: var(--shadow);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0.95rem 1rem 1.05rem;
    }

    .panel-title {
        font-family: "Urbanist", "Segoe UI", sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 0.2rem;
    }

    .panel-copy {
        font-family: "Urbanist", "Segoe UI", sans-serif;
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.55;
        margin-bottom: 0.95rem;
    }

    .control-label {
        font-family: "Urbanist", "Segoe UI", sans-serif;
        color: var(--ink);
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .stat-card {
        background: rgba(255, 255, 252, 0.94);
        border: 1px solid rgba(34, 37, 43, 0.08);
        border-radius: 18px;
        padding: 1rem 1rem 1.1rem;
        min-height: 6.75rem;
    }

    .stat-label {
        font-family: "Urbanist", "Segoe UI", sans-serif;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .stat-value {
        color: var(--ink);
        font-family: "Gelasio", "Georgia", serif;
        font-size: 2.1rem;
        line-height: 1.05;
    }

    .signal-card {
        background: rgba(22, 108, 89, 0.06);
        border-radius: 18px;
        border: 1px solid rgba(22, 108, 89, 0.12);
        padding: 1rem 1.05rem;
    }

    .signal-card strong {
        display: block;
        color: var(--ink);
        font-size: 0.96rem;
        margin-bottom: 0.28rem;
    }

    .signal-card span {
        font-family: "Urbanist", "Segoe UI", sans-serif;
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.55;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(249, 247, 242, 0.88);
        border: 1.5px dashed rgba(22, 108, 89, 0.30);
        border-radius: 18px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    div[data-testid="stFileUploaderDropzone"] section small,
    div[data-testid="stFileUploaderDropzone"] section span,
    div[data-testid="stFileUploaderDropzone"] section p {
        color: var(--muted);
    }

    div[data-testid="stFileUploaderDropzone"] button {
        background: #f8f6f1;
        color: var(--ink);
        border: 1px solid rgba(34, 37, 43, 0.14);
        border-radius: 999px;
        box-shadow: none;
        font-weight: 600;
    }

    div[data-testid="stFileUploaderDropzone"] button:hover {
        background: #ede8de;
        border-color: rgba(34, 37, 43, 0.22);
    }

    div[data-testid="stFileUploaderDropzone"] button:disabled {
        background: #f3eee4;
        color: rgba(34, 37, 43, 0.45);
        border-color: rgba(34, 37, 43, 0.10);
        opacity: 1;
    }

    div[data-testid="stExpander"] {
        background: rgba(249, 247, 242, 0.82);
        border: 1px solid rgba(34, 37, 43, 0.09);
        border-radius: 14px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] details summary p {
        color: var(--ink);
        font-weight: 600;
    }

    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    label[data-testid="stWidgetLabel"] {
        color: var(--ink);
        font-weight: 600;
    }

    div[data-testid="stToggle"] label,
    div[data-testid="stToggle"] p {
        color: var(--muted);
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stRadio"] > div,
    div[data-testid="stSlider"] > div {
        background: transparent;
    }

    div[data-baseweb="select"] > div {
        border-radius: 14px;
        border-color: rgba(34, 37, 43, 0.12);
        background: rgba(249, 247, 242, 0.88);
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 252, 0.92);
        border: 1px solid rgba(34, 37, 43, 0.08);
        border-radius: 18px;
        padding: 0.95rem 1rem;
    }

    div[data-testid="metric-container"] {
        background: transparent;
        border: 0;
        padding: 0;
    }

    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] [data-testid="stMetricLabel"],
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] * {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"],
    div[data-testid="metric-container"] [data-testid="stMetricValue"] *,
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: var(--ink);
        font-family: "Gelasio", "Georgia", serif;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }

    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] * {
        color: var(--muted);
    }

    div[data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    .results-grid {
        margin-top: 0.4rem;
    }

    .results-divider {
        height: 1px;
        background: var(--line);
        margin: 1.1rem 0 1.4rem;
    }

    button[kind="primary"] {
        border-radius: 999px;
        border: 0;
        background: linear-gradient(135deg, #1d6e5d, #0f5647);
        box-shadow: 0 16px 30px rgba(22, 108, 89, 0.20);
        min-height: 3.15rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }

    button[kind="secondary"] {
        border-radius: 999px;
        background: #f8f6f1;
        color: var(--ink);
        border: 1px solid rgba(34, 37, 43, 0.14);
        box-shadow: none;
    }

    button[kind="secondary"]:hover {
        background: #ede8de;
        border-color: rgba(34, 37, 43, 0.22);
    }

    .results-shell {
        margin-top: 1rem;
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

    @media (max-width: 1100px) {
        .hero-title {
            max-width: 14ch;
        }
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    hero_col, brief_col = st.columns([1.9, 1.1], gap="medium")

    with hero_col:
        st.markdown(
            """
            <div class="hero-shell">
                <div class="eyebrow">Portfolio Construction</div>
                <h1 class="hero-title">Hybrid Quantamental Optimizer</h1>
                <div class="hero-copy">
                    Build a portfolio from return assumptions, volatility inputs,
                    and a target risk level. Upload the market data, solve the risky
                    mix, then decide how much to hold in cash.
                </div>
                <div class="badge-row">
                    <span class="badge">Expected returns</span>
                    <span class="badge">Forward volatility</span>
                    <span class="badge">Cash allocation</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with brief_col:
        st.markdown(
            """
            <div class="brief-card">
                <div class="brief-title">How it works</div>
                <div class="brief-line">
                    <strong>1. Load the universe</strong>
                    <span>Bring in price history and one metrics file with return,
                    volatility, and weight bounds.</span>
                </div>
                <div class="brief-line">
                    <strong>2. Solve the risky mix</strong>
                    <span>Find the tangency portfolio for the chosen risk model
                    and constraints.</span>
                </div>
                <div class="brief-line">
                    <strong>3. Set target risk</strong>
                    <span>Choose the final volatility target and let the app blend
                    risky assets with cash.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_section(kicker: str, title: str, copy: str) -> None:
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


def _render_signal_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="signal-card">
            <strong>{escape(title)}</strong>
            <span>{escape(body)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_panel_intro(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="panel-title">{escape(title)}</div>
        <div class="panel-copy">{escape(body)}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_control_label(label: str) -> None:
    st.markdown(
        f"<div class='control-label'>{escape(label)}</div>",
        unsafe_allow_html=True,
    )


def _render_stat_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{escape(label)}</div>
            <div class="stat-value">{escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _risk_model_note(risk_model: RiskModel) -> str:
    if risk_model == RiskModel.HISTORICAL:
        return "Historical mode annualizes sample covariance using 252 trading days."
    return "Forward-looking mode keeps the correlation structure from history and uses implied volatility as the risk input."


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}"


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
if "risk_free_rate" not in st.session_state:
    st.session_state.risk_free_rate = 0.04

_render_hero()

_render_section(
    "Inputs",
    "Upload inputs",
    "Add a price history file and a metrics file with expected return, volatility, and optional weight bounds.",
)

prices_col, metrics_col = st.columns(2, gap="medium")

with prices_col:
    with st.container(border=True):
        _render_panel_intro(
            "Price history",
            "Use one date column followed by one column per ticker.",
        )
        if st.toggle("Show sample format", key="show_price_sample"):
            st.code(
                """date,AAPL,GOOG,MSFT
2023-01-31,150.23,105.44,280.50
2023-02-28,152.11,108.22,285.33""",
                language="csv",
            )
        prices_file = st.file_uploader(
            "Upload prices.csv",
            type=["csv"],
            key="prices_upload",
            label_visibility="collapsed",
        )
        if prices_file is not None:
            _render_signal_card(
                "Price file loaded",
                f"{prices_file.name} is ready for covariance construction.",
            )

with metrics_col:
    with st.container(border=True):
        _render_panel_intro(
            "Asset metrics",
            "Include ticker, expected return, implied volatility, and optional weight bounds.",
        )
        if st.toggle("Show sample format", key="show_metrics_sample"):
            st.code(
                """ticker,expected_return,implied_volatility,min_weight,max_weight
AAPL,0.12,0.25,0.0,1.0
GOOG,0.15,0.28,0.0,1.0""",
                language="csv",
            )
        metrics_file = st.file_uploader(
            "Upload metrics.csv",
            type=["csv"],
            key="metrics_upload",
            label_visibility="collapsed",
        )
        if metrics_file is not None:
            _render_signal_card(
                "Metrics file loaded",
                f"{metrics_file.name} is ready for return, volatility, and bounds.",
            )

files_ready = prices_file is not None and metrics_file is not None

_render_section(
    "Model setup",
    "Choose assumptions",
    "Set the risk-free rate, pick the risk model, and solve for the tangency portfolio.",
)

controls_col, status_col = st.columns([1.45, 0.9], gap="medium")

with controls_col:
    with st.container(border=True):
        _render_panel_intro(
            "Optimization inputs",
            "These settings drive the tangency portfolio before any cash mix is applied.",
        )
        control_a, control_b = st.columns(2, gap="large")
        with control_a:
            _render_control_label("Risk-free rate (%)")
            risk_free_rate = (
                st.slider(
                    "Risk-free rate (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=4.0,
                    step=0.25,
                    format="%.2f%%",
                    disabled=not files_ready,
                    label_visibility="collapsed",
                )
                / 100.0
            )

        with control_b:
            _render_control_label("Risk model")
            risk_model_label = st.radio(
                "Risk model",
                ["Forward-Looking", "Historical"],
                index=0,
                horizontal=True,
                disabled=not files_ready,
                label_visibility="collapsed",
            )
            risk_model = (
                RiskModel.FORWARD_LOOKING
                if risk_model_label == "Forward-Looking"
                else RiskModel.HISTORICAL
            )

        st.caption(_risk_model_note(risk_model))
        optimize_button = st.button(
            "Solve portfolio",
            type="primary",
            use_container_width=True,
            disabled=not files_ready,
        )

with status_col:
    with st.container(border=True):
        if files_ready:
            _render_panel_intro("Ready", "Both files are loaded and the model can be solved.")
            _render_signal_card(
                "Current run",
                f"{prices_file.name} + {metrics_file.name} with a risk-free rate of {_fmt_pct(risk_free_rate)}.",
            )
        else:
            _render_panel_intro(
                "Waiting for files",
                "Load both CSV files to enable the optimization controls.",
            )
            _render_signal_card(
                "Needed to continue",
                "Add one price history CSV and one asset metrics CSV.",
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
                risk_free_rate=risk_free_rate,
                risk_model=risk_model,
                annualization_factor=252
                if risk_model == RiskModel.HISTORICAL
                else None,
            )
            cml_points = generate_cml(
                tangency_portfolio=tangency,
                risk_free_rate=risk_free_rate,
                num_points=30,
            )

            st.session_state.tangency_portfolio = tangency
            st.session_state.cml_points = cml_points
            st.session_state.risk_free_rate = risk_free_rate
            st.session_state.optimization_complete = True
            st.success("Portfolio solved. Adjust the target volatility to shape the final allocation.")
            st.rerun()
        except Exception as exc:
            st.session_state.optimization_complete = False
            st.error(f"Optimization failed: {exc}")

if st.session_state.optimization_complete and st.session_state.tangency_portfolio is not None:
    tangency = st.session_state.tangency_portfolio

    _render_section(
        "Results",
        "Results",
        "Review the tangency portfolio, then adjust the target volatility to decide the final cash mix.",
    )

    with st.container(border=True):
        st.markdown(
            "<div class='panel-title'>Tangency portfolio</div><div class='panel-copy'>This is the highest-Sharpe risky portfolio before any cash is blended in.</div>",
            unsafe_allow_html=True,
        )
        headline_a, headline_b, headline_c = st.columns(3, gap="large")
        with headline_a:
            _render_stat_card("Expected return", _fmt_pct(float(tangency["expected_return"])))
        with headline_b:
            _render_stat_card("Volatility", _fmt_pct(float(tangency["volatility"])))
        with headline_c:
            _render_stat_card("Sharpe ratio", f"{float(tangency['sharpe_ratio']):.2f}")

    max_vol_pct = float(tangency["volatility"]) * 100.0
    default_target_pct = min(10.0, max_vol_pct)

    with st.container(border=True):
        st.markdown(
            "<div class='panel-title'>Target volatility</div><div class='panel-copy'>Move from all cash toward the tangency portfolio until the final risk level matches your target.</div>",
            unsafe_allow_html=True,
        )
        _render_control_label("Target portfolio volatility (%)")
        target_vol_pct = st.slider(
            "Target portfolio volatility (%)",
            min_value=0.0,
            max_value=max_vol_pct,
            value=default_target_pct,
            step=0.1,
            format="%.1f%%",
            help="Slide from all cash toward the tangency portfolio.",
            label_visibility="collapsed",
        )

    final_portfolio = target_portfolio(
        tangency_portfolio=tangency,
        target_volatility=target_vol_pct / 100.0,
        risk_free_rate=st.session_state.risk_free_rate,
    )

    with st.container(border=True):
        st.markdown(
            "<div class='panel-title'>Final portfolio</div><div class='panel-copy'>These numbers reflect the risky portfolio scaled with cash to hit the selected volatility.</div>",
            unsafe_allow_html=True,
        )
        final_a, final_b, final_c, final_d = st.columns(4, gap="large")
        with final_a:
            _render_stat_card("Expected return", _fmt_pct(float(final_portfolio["expected_return"])))
        with final_b:
            _render_stat_card("Volatility", _fmt_pct(float(final_portfolio["volatility"])))
        with final_c:
            _render_stat_card("Sharpe ratio", f"{float(final_portfolio['sharpe_ratio']):.2f}")
        with final_d:
            _render_stat_card("Cash allocation", _fmt_pct(float(final_portfolio["cash_weight"])))

    render_results(
        tangency=tangency,
        final_portfolio=final_portfolio,
        cml_points=st.session_state.cml_points,
        rf_rate=st.session_state.risk_free_rate,
    )

else:
    with st.container(border=True):
        _render_panel_intro(
            "No allocation yet",
            "Solve the model to see the tangency portfolio, the target volatility slider, and the final allocation.",
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
