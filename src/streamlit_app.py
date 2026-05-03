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
    :root {
        --bg: #f3ecdf;
        --bg-glow: #efe2ce;
        --panel: rgba(255, 250, 242, 0.9);
        --panel-strong: rgba(255, 248, 238, 0.98);
        --ink: #22252b;
        --muted: #6e6a63;
        --accent: #166c59;
        --accent-soft: #dbe9e3;
        --signal: #c76b3f;
        --line: rgba(34, 37, 43, 0.12);
        --shadow: 0 18px 60px rgba(91, 70, 42, 0.09);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(199, 107, 63, 0.10), transparent 30%),
            radial-gradient(circle at top right, rgba(22, 108, 89, 0.12), transparent 28%),
            linear-gradient(180deg, var(--bg-glow) 0%, var(--bg) 36%, #f7f1e6 100%);
        color: var(--ink);
        font-family: "Avenir Next", "Segoe UI", sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(243, 236, 223, 0.72);
        backdrop-filter: blur(10px);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1240px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
        letter-spacing: -0.03em;
        color: var(--ink);
    }

    .hero-shell {
        background: linear-gradient(145deg, rgba(255, 250, 242, 0.96), rgba(247, 239, 227, 0.90));
        border: 1px solid rgba(34, 37, 43, 0.08);
        border-radius: 32px;
        box-shadow: var(--shadow);
        padding: 2rem 2.2rem;
        min-height: 100%;
    }

    .eyebrow {
        color: var(--signal);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: clamp(2.7rem, 5vw, 4.6rem);
        line-height: 0.94;
        margin: 0;
        max-width: 9ch;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.65;
        max-width: 52ch;
        margin-top: 1rem;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 1.4rem;
    }

    .badge {
        border-radius: 999px;
        background: rgba(22, 108, 89, 0.10);
        color: var(--accent);
        border: 1px solid rgba(22, 108, 89, 0.14);
        font-size: 0.88rem;
        font-weight: 600;
        padding: 0.45rem 0.85rem;
    }

    .brief-card {
        background: linear-gradient(180deg, rgba(255, 248, 238, 0.98), rgba(247, 239, 227, 0.92));
        border: 1px solid rgba(34, 37, 43, 0.08);
        border-radius: 28px;
        box-shadow: var(--shadow);
        padding: 1.65rem 1.6rem;
        min-height: 100%;
    }

    .brief-title {
        color: var(--ink);
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
        font-size: 1.35rem;
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
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .section-wrap {
        margin-top: 2rem;
    }

    .section-kicker {
        color: var(--signal);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .section-title {
        font-size: 2rem;
        line-height: 1.04;
        margin-bottom: 0.45rem;
    }

    .section-copy {
        color: var(--muted);
        line-height: 1.65;
        max-width: 60ch;
        margin-bottom: 1rem;
    }

    .field-note, .status-note {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .signal-card {
        background: var(--panel);
        border-radius: 24px;
        border: 1px solid rgba(34, 37, 43, 0.08);
        box-shadow: var(--shadow);
        padding: 1.35rem 1.45rem;
    }

    .signal-card strong {
        display: block;
        color: var(--ink);
        font-size: 0.96rem;
        margin-bottom: 0.28rem;
    }

    .signal-card span {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.55;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 250, 242, 0.86);
        border: 1.5px dashed rgba(22, 108, 89, 0.30);
        border-radius: 24px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    div[data-testid="stExpander"] {
        background: rgba(255, 250, 242, 0.78);
        border: 1px solid rgba(34, 37, 43, 0.09);
        border-radius: 18px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] details summary p {
        color: var(--ink);
        font-weight: 600;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stRadio"] > div,
    div[data-testid="stSlider"] > div {
        background: transparent;
    }

    div[data-baseweb="select"] > div {
        border-radius: 16px;
        border-color: rgba(34, 37, 43, 0.12);
        background: rgba(255, 250, 242, 0.82);
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 250, 242, 0.92);
        border: 1px solid rgba(34, 37, 43, 0.08);
        border-radius: 22px;
        box-shadow: var(--shadow);
        padding: 0.95rem 1rem;
    }

    div[data-testid="metric-container"] {
        background: transparent;
        border: 0;
        padding: 0;
    }

    div[data-testid="metric-container"] label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
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
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    hero_col, brief_col = st.columns([1.8, 1], gap="large")

    with hero_col:
        st.markdown(
            """
            <div class="hero-shell">
                <div class="eyebrow">Forward-Looking Portfolio Construction</div>
                <h1 class="hero-title">Hybrid Quantamental Optimizer</h1>
                <div class="hero-copy">
                    Build an allocation from forecast return assumptions,
                    forward-looking volatility, and a deliberate capital-market-line
                    decision instead of relying on a purely backward-looking portfolio.
                </div>
                <div class="badge-row">
                    <span class="badge">Expected return engine</span>
                    <span class="badge">Forward volatility input</span>
                    <span class="badge">Cash-mix targeting</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with brief_col:
        st.markdown(
            """
            <div class="brief-card">
                <div class="brief-title">What this workflow does</div>
                <div class="brief-line">
                    <strong>1. Align the universe</strong>
                    <span>Load price history and asset assumptions into one clean
                    optimization set.</span>
                </div>
                <div class="brief-line">
                    <strong>2. Solve the risky mix</strong>
                    <span>Find the tangency portfolio under your chosen risk model
                    and weight bounds.</span>
                </div>
                <div class="brief-line">
                    <strong>3. Choose the final risk posture</strong>
                    <span>Dial the target volatility and let the app blend risky
                    assets with cash.</span>
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


def _risk_model_note(risk_model: RiskModel) -> str:
    if risk_model == RiskModel.HISTORICAL:
        return "Historical mode annualizes sample covariance using 252 trading days."
    return "Forward-looking mode keeps historical correlation structure and swaps in implied volatility."


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
    "Assemble the market packet",
    "Upload one time-series file for price history and one cross-sectional file for expected return, volatility, and allocation bounds.",
)

prices_col, metrics_col = st.columns(2, gap="large")

with prices_col:
    st.markdown("**Price history**")
    st.markdown(
        "<div class='field-note'>One <code>date</code> column followed by one column per ticker.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Preview expected format", expanded=False):
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
            "Loaded price history",
            f"{prices_file.name} is ready for alignment and covariance construction.",
        )

with metrics_col:
    st.markdown("**Asset metrics**")
    st.markdown(
        "<div class='field-note'>Include <code>ticker</code>, <code>expected_return</code>, <code>implied_volatility</code>, and optional weight bounds.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Preview expected format", expanded=False):
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
            "Loaded asset assumptions",
            f"{metrics_file.name} is ready for expected return, volatility, and weight constraints.",
        )

files_ready = prices_file is not None and metrics_file is not None

_render_section(
    "Model setup",
    "Set the optimization stance",
    "Choose the risk-free rate, decide how covariance should be built, and then solve for the risky mix.",
)

controls_col, status_col = st.columns([1.7, 1], gap="large")

with controls_col:
    control_a, control_b = st.columns(2, gap="large")
    with control_a:
        risk_free_rate = (
            st.slider(
                "Risk-free rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=4.0,
                step=0.25,
                format="%.2f%%",
                disabled=not files_ready,
            )
            / 100.0
        )

    with control_b:
        risk_model_label = st.radio(
            "Risk model",
            ["Forward-Looking", "Historical"],
            index=0,
            horizontal=True,
            disabled=not files_ready,
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
    if files_ready:
        _render_signal_card(
            "Ready to solve",
            f"Using {prices_file.name} with {metrics_file.name}. The current rate assumption is {_fmt_pct(risk_free_rate)}.",
        )
    else:
        _render_signal_card(
            "Waiting for both files",
            "Upload both the price history and the asset metrics packet to unlock optimization controls.",
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
        "Read the portfolio story",
        "Start with the risky tangency mix, then decide how much cash to blend in by dialing the target volatility.",
    )

    headline_a, headline_b, headline_c = st.columns(3, gap="large")
    with headline_a:
        st.metric("Tangency return", _fmt_pct(float(tangency["expected_return"])))
    with headline_b:
        st.metric("Tangency volatility", _fmt_pct(float(tangency["volatility"])))
    with headline_c:
        st.metric("Tangency Sharpe", f"{float(tangency['sharpe_ratio']):.2f}")

    max_vol_pct = float(tangency["volatility"]) * 100.0
    default_target_pct = min(10.0, max_vol_pct)

    st.markdown("<div class='results-shell'>", unsafe_allow_html=True)
    target_vol_pct = st.slider(
        "Target portfolio volatility (%)",
        min_value=0.0,
        max_value=max_vol_pct,
        value=default_target_pct,
        step=0.1,
        format="%.1f%%",
        help="Slide from all-cash toward the tangency portfolio.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    final_portfolio = target_portfolio(
        tangency_portfolio=tangency,
        target_volatility=target_vol_pct / 100.0,
        risk_free_rate=st.session_state.risk_free_rate,
    )

    final_a, final_b, final_c, final_d = st.columns(4, gap="large")
    with final_a:
        st.metric("Expected return", _fmt_pct(float(final_portfolio["expected_return"])))
    with final_b:
        st.metric("Volatility", _fmt_pct(float(final_portfolio["volatility"])))
    with final_c:
        st.metric("Sharpe ratio", f"{float(final_portfolio['sharpe_ratio']):.2f}")
    with final_d:
        st.metric("Cash allocation", _fmt_pct(float(final_portfolio["cash_weight"])))

    render_results(
        tangency=tangency,
        final_portfolio=final_portfolio,
        cml_points=st.session_state.cml_points,
        rf_rate=st.session_state.risk_free_rate,
    )

else:
    _render_signal_card(
        "No allocation yet",
        "Once both files are loaded and the portfolio is solved, the tangency mix, target slider, and final allocation story will appear here.",
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
