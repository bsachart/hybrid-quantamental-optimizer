"""
Streamlit UI for the Hybrid Quantamental Optimizer.
Workflow:
1. Configure Assets (Prices + Constraints)
2. Generate Returns (Market + Alpha)
3. Calibrate Risk (IV + Blending)
4. Optimize
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Prioritize project root in path
sys.path.insert(0, os.getcwd())

from src.ui.state_manager import (
    initialize_session_state,
    create_default_metrics,
    save_optimization_results,
    get_optimization_results,
    clear_optimization_results,
)
from src.core.data import (
    load_and_validate_prices,
    load_and_validate_asset_metrics,
    align_tickers,
    infer_frequency,
    fetch_prices_from_yfinance,
)
from src.core.returns import calculate_view_returns
from src.core.risk import calculate_hybrid_covariance, calculate_covariance_matrix
from src.optimization import optimizer
from src.ui import charts

st.set_page_config(
    page_title="Hybrid Optimizer", layout="wide", initial_sidebar_state="collapsed"
)

# --- Custom CSS for Premium Look ---
st.markdown(
    """
<style>
    .block-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    h1 {
        background: linear-gradient(90deg, #00C853, #00E5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Hybrid Quantamental Optimizer")
st.markdown("> *Modern Portfolio Theory evolved for the fundamental investor.*")

# --- Initialize Session State ---
initialize_session_state()

# --- Step 1: Universe ---
st.header("Step 1: Universe")

if st.session_state["prices"] is None:
    st.info("👋 **Welcome!** Define your asset universe to begin.")

data_mode = st.radio(
    "Data Entry Mode", ["Manual Ticker Entry", "Upload CSV"], horizontal=True
)

if data_mode == "Manual Ticker Entry":
    tickers_input = st.text_input(
        "Enter Ticker Symbols (comma separated)", "AAPL, GOOG, MSFT, NVDA"
    )

    c1, c2 = st.columns(2)
    lookback_period = c1.selectbox(
        "Lookback Period", ["1y", "2y", "5y", "10y"], index=1
    )
    data_interval = c2.selectbox("Frequency", ["Daily", "Weekly", "Monthly"], index=2)

    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}

    if st.button("Fetch Market Data"):
        try:
            ticker_list = [
                t.strip().upper() for t in tickers_input.split(",") if t.strip()
            ]
            with st.spinner("Downloading price history..."):
                prices = fetch_prices_from_yfinance(
                    ticker_list,
                    period=lookback_period,
                    interval=interval_map[data_interval],
                )
                st.session_state["prices"] = prices
                st.success(f"Successfully fetched {len(prices.columns)} assets.")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
else:  # Upload CSV
    with st.expander("ℹ️ CSV Format Guide"):
        st.markdown("""
        **Expected Format**:
        - **Column 1**: Date (YYYY-MM-DD)
        - **Other Columns**: Ticker names with price history.
        """)
    price_file = st.file_uploader("Upload Price History (CSV)", type=["csv"])
    if price_file:
        file_key = f"price_{price_file.name}_{price_file.size}"
        if st.session_state.get("price_file_key") != file_key:
            try:
                price_file.seek(0)
                st.session_state["prices"] = load_and_validate_prices(price_file)
                st.session_state["price_file_key"] = file_key
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

if st.session_state["prices"] is not None:
    with st.expander("📊 View Asset Correlations"):
        corr = st.session_state["prices"].pct_change().corr()
        st.dataframe(
            corr.style.background_gradient(
                cmap="RdYlGn", axis=None, vmin=-1, vmax=1
            ).format("{:.2f}"),
            width="stretch",
        )

st.markdown("---")
st.subheader("Bulk Upload Metrics")
with st.expander("ℹ️ Metrics CSV Format Guide"):
    st.markdown("""
    **Expected Format**:
    - **Ticker**: (Index) e.g., AAPL
    - **Implied Volatility**: e.g., 0.25
    - **Custom Return**: e.g., 0.10
    - **Constraint**: 'Long', 'Short', or 'Both'
    """)

metrics_file = st.file_uploader("Upload Asset Metrics (CSV)", type=["csv"])
if metrics_file:
    file_key = f"metrics_{metrics_file.name}_{metrics_file.size}"
    if st.session_state.get("metrics_file_key") != file_key:
        try:
            metrics_file.seek(0)
            metrics_df = load_and_validate_asset_metrics(metrics_file)
            if st.session_state["prices"] is not None:
                common = st.session_state["prices"].columns.intersection(
                    metrics_df.index
                )
                if not common.empty:
                    st.session_state["metrics"] = metrics_df.loc[common]
                    st.session_state["metrics_file_key"] = file_key
                    st.success(f"Loaded metrics for {len(common)} assets.")
                    st.rerun()
                else:
                    st.warning("No common tickers between Price data and Metrics CSV.")
            else:
                st.session_state["metrics"] = metrics_df
                st.session_state["metrics_file_key"] = file_key
                st.success(f"Loaded metrics for {len(metrics_df)} assets.")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state["prices"] is not None:
    tickers = st.session_state["prices"].columns.tolist()

    if (
        st.session_state["metrics"] is None
        or list(st.session_state["metrics"].index) != tickers
    ):
        st.session_state["metrics"] = create_default_metrics(tickers)

    # Handled in Steps 2 & 3

    # --- Step 2: Generate Returns ---
    st.markdown("---")
    st.header("Step 2: Generate Returns")

    col_ret1, col_ret2 = st.columns([2, 1])
    with col_ret1:
        ret_method = st.radio(
            "Return Generation Method",
            ["Custom", "Fundamental"],
            horizontal=True,
            label_visibility="collapsed",
            index=1,
        )

    if ret_method == "Custom":
        st.info("💡 **Logic**: Provide raw expected returns below.")
        custom_editor = st.data_editor(
            st.session_state["metrics"][["Custom Return (%)", "Constraint"]],
            column_config={
                "Constraint": st.column_config.SelectboxColumn(
                    "Constraint", options=["Long", "Short", "Both"], required=True
                ),
                "Custom Return (%)": st.column_config.NumberColumn(
                    "Custom Return (%)", format="%.2f"
                ),
            },
            width="stretch",
            key="custom_ret_editor",
        )
        st.session_state["metrics"].update(custom_editor)
        expected_returns = (
            st.session_state["metrics"]["Custom Return (%)"].values / 100.0
        )

    else:  # Fundamental Implied CAGR
        st.warning("⚠️ **Fundamental Mode** (In Development)")
        st.markdown("""
        We are building a model to derive returns from:
        - **Revenue Growth** & **Margin Expansion**
        - **Current Valuation (P/E)** vs **Terminal P/E**
        
        *Expected release: Q1 2026. Defaulting to 8% for now.*
        """)
        expected_returns = np.full(len(tickers), 0.08)

    # --- Step 3: Calibrate Risk ---
    st.markdown("---")
    st.header("Step 3: Calibrate Risk")

    st.markdown("**Risk Calibration**")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        market_iv = st.slider(
            "Market Implied Volatility (VIX)", 0.0, 1.0, 0.20, 0.01, format="%.2f"
        )
    with col_r2:
        blend_w = st.slider("IV Blending Weight (w)", 0.0, 1.0, 0.5, 0.05)

    st.info(
        f"Using {blend_w:.0%} Asset Implied Vol / {1-blend_w:.0%} Market Implied Vol (Reference Index: MSCI World, S&P 500...)"
    )

    st.markdown("**Asset-Specific Implied Volatility**")
    st.caption("ℹ️ *Blended Vol is auto-calculated based on Asset IV and Market IV.*")

    # Pre-calculate components for the table
    detected_freq = infer_frequency(st.session_state["prices"])
    hist_cov = calculate_covariance_matrix(st.session_state["prices"], detected_freq)
    hist_vols = np.sqrt(np.diag(hist_cov))

    # Build a temp display dataframe
    iv_display_df = st.session_state["metrics"][["Implied Volatility (%)"]].copy()
    iv_display_df["Blended Vol (%)"] = blend_w * iv_display_df[
        "Implied Volatility (%)"
    ] + (1 - blend_w) * (market_iv * 100.0)

    iv_editor = st.data_editor(
        iv_display_df,
        column_config={
            "Implied Volatility (%)": st.column_config.NumberColumn(
                "Asset IV (%)", format="%.2f", min_value=1.0
            ),
            "Blended Vol (%)": st.column_config.NumberColumn(
                "Blended Vol (%)", format="%.2f", disabled=True
            ),
        },
        width="stretch",
        key="iv_editor",
    )
    # Update state only from the editable column
    st.session_state["metrics"]["Implied Volatility (%)"] = iv_editor[
        "Implied Volatility (%)"
    ]

    hybrid_cov = calculate_hybrid_covariance(
        st.session_state["prices"],
        st.session_state["metrics"]["Implied Volatility (%)"].values / 100.0,
        anchor_vols=market_iv,
        w_iv=blend_w,
        frequency=detected_freq,
    )

    max_val = hybrid_cov.abs().max().max()
    with st.expander("View Hybrid Covariance Matrix"):
        st.dataframe(
            hybrid_cov.style.background_gradient(
                cmap="RdYlGn", axis=None, vmin=-max_val, vmax=max_val
            ).format("{:.4f}"),
            width="stretch",
        )

    # --- Step 4: Optimize ---
    st.markdown("---")
    st.header("Step 4: Optimize")

    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        rf_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.25) / 100.0
    with col_o2:
        max_long = st.slider("Global Max Long (%)", 0.0, 100.0, 100.0, 0.5) / 100.0
    with col_o3:
        max_short = st.slider("Global Max Short (%)", 0.0, 100.0, 20.0, 0.5) / 100.0

    st.info(
        "🎯 Clicking optimize will generate the Tangency Portfolio and Efficient Frontier."
    )
    # Map directional constraints to bounds, applying global caps
    asset_bounds = []
    for constraint in st.session_state["metrics"]["Constraint"]:
        if constraint == "Long":
            asset_bounds.append((0.0, max_long))
        elif constraint == "Short":
            asset_bounds.append((-max_short, 0.0))
        else:
            asset_bounds.append((-max_short, max_long))

    if st.button("Run Optimization", type="primary"):
        # Step 1: Find tangency portfolio (risky assets only)
        opt = optimizer.PortfolioOptimizer(
            expected_returns=expected_returns,
            cov_matrix=hybrid_cov.values,
            risk_free_rate=rf_rate,
        )

        tangency = opt.maximize_sharpe(bounds=asset_bounds)

        if not tangency.success:
            st.error(f"Optimization failed: {tangency.message}")
        else:
            # Compute Efficient Frontier & Random Portfolios ONCE
            with st.spinner("Calculating Efficient Frontier..."):
                frontier = opt.efficient_frontier(bounds=asset_bounds, num_points=50)
                frontier_dicts = [
                    {
                        "return": p.return_,
                        "volatility": p.volatility,
                        "weights": p.weights,
                    }
                    for p in frontier
                ]

                random_ps = opt.random_portfolios(num_portfolios=1000)
                random_dicts = [
                    {
                        "return": p.return_,
                        "volatility": p.volatility,
                        "weights": p.weights,
                    }
                    for p in random_ps
                ]

            # Save optimization results to session state (data only, no objects)
            save_optimization_results(
                tangency_return=tangency.return_,
                tangency_vol=tangency.volatility,
                tangency_sharpe=tangency.sharpe_ratio,
                tangency_weights=tangency.weights,
                expected_returns=expected_returns,
                cov_matrix=hybrid_cov.values,
                rf_rate=rf_rate,
                bounds=asset_bounds,
                tickers=tickers,
                frontier_points=frontier_dicts,
                random_portfolios=random_dicts,
            )
            # Note: st.rerun() not needed - Streamlit reruns automatically on state change

    # --- Display Results if Available ---
    opt_results = get_optimization_results()

    if opt_results is not None:
        # Reconstruct objects from saved data
        tickers = opt_results["tickers"]
        tangency_weights = np.array(opt_results["tangency_weights"])
        expected_returns = np.array(opt_results["expected_returns"])
        cov_matrix = np.array(opt_results["cov_matrix"])
        rf_rate = opt_results["rf_rate"]
        asset_bounds = opt_results["bounds"]

        # Create a temporary PortfolioMetrics object for display
        from src.optimization.optimizer import PortfolioMetrics

        tangency = PortfolioMetrics(
            return_=opt_results["tangency_return"],
            volatility=opt_results["tangency_vol"],
            sharpe_ratio=opt_results["tangency_sharpe"],
            weights=tangency_weights,
            success=True,
            message="",
        )

        # Recreate optimizer for cash allocation
        opt = optimizer.PortfolioOptimizer(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=rf_rate,
        )

        # Display metrics for tangency portfolio
        with st.expander("🎯 Tangency Portfolio (100% Risky Assets)", expanded=False):
            t1, t2, t3 = st.columns(3)
            t1.metric("Expected Return", f"{tangency.return_:.2%}")
            t2.metric("Volatility", f"{tangency.volatility:.2%}")
            t3.metric("Sharpe Ratio", f"{tangency.sharpe_ratio:.2f}")

            st.caption("Weights (Risky Assets Only):")
            tangency_df = pd.DataFrame({"Weight": tangency.weights}, index=tickers)
            st.dataframe(
                tangency_df.style.format("{:.2%}").background_gradient(
                    cmap="RdYlGn", subset=["Weight"], vmin=-1.0, vmax=1.0
                ),
                width="stretch",
            )

        # Prepare cached chart data
        frontier_dicts = opt_results.get("frontier_points", [])
        random_dicts = opt_results.get("random_portfolios", [])

        st.markdown("---")
        st.subheader("Final Allocation Strategy (Capital Market Line)")

        # Dynamic CML Slider
        max_vol = tangency.volatility

        # Default to 15% or max_vol if lower
        default_target = min(0.15, max_vol)

        target_volatility = st.slider(
            "Target Risk (Volatility) on CML",
            min_value=0.0,
            max_value=float(max_vol),
            value=float(default_target),
            step=0.005,
            format="%.1f%%",
            key="cml_vol_slider",
            help="Navigate the Capital Market Line: 0% = Risk Free Asset, Max = Tangency Portfolio",
        )

        # Validate target volatility
        if (
            target_volatility > tangency.volatility * 1.05
        ):  # Allow slight overshoot due to float precision
            st.warning(
                f"⚠️ Target volatility ({target_volatility:.1%}) is higher than the tangency portfolio's volatility ({tangency.volatility:.1%}). "
                f"This will result in a cash allocation of 0% and the portfolio will be the tangency portfolio."
            )

        st.info(f"📊 Target Portfolio Volatility: {target_volatility:.2%}")

        # Step 2: Allocate between tangency portfolio and cash
        allocation = opt.allocate_with_cash(
            tangency_portfolio=tangency,
            target_volatility=target_volatility,
            asset_names=tickers,
        )

        # Display final portfolio metrics (with cash)
        st.subheader("Final Portfolio (with Cash Allocation)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Return", f"{allocation['final_return']:.2%}")
        m2.metric("Volatility", f"{allocation['final_volatility']:.2%}")
        m3.metric("Sharpe Ratio", f"{allocation['final_sharpe']:.2f}")
        m4.metric("Cash Allocation", f"{allocation['cash_fraction']:.2%}")

        # Prepare data for charts (Using Cached Frontier)
        # frontier and random_ps are already loaded as dicts above

        opt_dict = {
            "return": tangency.return_,
            "volatility": tangency.volatility,
            "weights": tangency.weights,
            "sharpe": tangency.sharpe_ratio,
        }

        target_dict = {
            "return": allocation["final_return"],
            "volatility": allocation["final_volatility"],
            "weights": allocation["final_weights"],
        }

        asset_vols = np.sqrt(np.diag(cov_matrix))

        chart = charts.plot_efficient_frontier(
            frontier_points=frontier_dicts,
            random_portfolios=random_dicts,
            optimal_portfolio=opt_dict,
            tickers=tickers,
            asset_returns=expected_returns,
            asset_vols=asset_vols,
            rf_rate=rf_rate,
        )

        st.altair_chart(chart, width="stretch")

        # Allocation Table (with cash) - Use more efficient approach
        st.subheader("Final Allocation Table")

        # Create ticker to index mapping (O(1) lookups)
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        # Build allocation data efficiently
        alloc_data = [
            {
                "Asset": "CASH",
                "Weight": allocation["cash_fraction"],
                "Expected Return": rf_rate,
                "Volatility": 0.0,
            }
        ]

        # Add risky assets
        for ticker in tickers:
            idx = ticker_to_idx[ticker]
            weight = allocation["allocation_table"][ticker]
            if abs(weight) > 0.0001:  # Only show non-trivial weights
                alloc_data.append(
                    {
                        "Asset": ticker,
                        "Weight": weight,
                        "Expected Return": expected_returns[idx],
                        "Volatility": asset_vols[idx],
                    }
                )

        alloc_df = pd.DataFrame(alloc_data).set_index("Asset")
        alloc_df = alloc_df.sort_values("Weight", ascending=False)

        st.dataframe(
            alloc_df.style.format(
                {
                    "Weight": "{:.2%}",
                    "Expected Return": "{:.2%}",
                    "Volatility": "{:.2%}",
                }
            ).background_gradient(
                cmap="RdYlGn", subset=["Weight"], vmin=-1.0, vmax=1.0
            ),
            width="stretch",
        )

        st.caption(
            f"💡 **Allocation Strategy**: {allocation['risky_fraction']:.1%} in optimized "
            f"risky portfolio, {allocation['cash_fraction']:.1%} in cash to achieve "
            f"{target_volatility:.1%} target volatility."
        )

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "Hybrid Quantamental Optimizer &copy; 2025 | Built for Precision and Insight"
    "</div>",
    unsafe_allow_html=True,
)
