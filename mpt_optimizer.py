"""
Modern Portfolio Theory (MPT) Optimizer
-----------------------------------------
Proves: Linear Algebra (covariance/matrix ops), Monte Carlo simulation,
and constrained optimization (SciPy SLSQP) applied to the Markowitz
Efficient Frontier.

Run locally:
    pip install streamlit yfinance numpy pandas scipy plotly
    streamlit run mpt_optimizer.py

Deploy free & public:
    1. Push this file + requirements.txt to a public GitHub repo.
    2. Go to https://share.streamlit.io -> "New app" -> point to the repo.
    3. Streamlit Community Cloud builds and hosts it at a public URL for free.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="MPT Optimizer", layout="wide")
st.title("📈 Modern Portfolio Theory (MPT) Optimizer")
st.caption("Markowitz Efficient Frontier · Monte Carlo Simulation · SLSQP Optimization")

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Configuration")
default_tickers = "AAPL, MSFT, JPM, JNJ, XOM"
tickers_input = st.sidebar.text_input("Tickers (comma-separated, exactly 5 recommended)", default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

years_back = st.sidebar.slider("Years of history", 1, 10, 3)
n_portfolios = st.sidebar.number_input("Monte Carlo simulations", 1000, 200000, 50000, step=1000)
risk_free_rate = st.sidebar.number_input("Risk-free rate (annual, decimal)", 0.0, 0.20, 0.065, step=0.005)
allow_short = st.sidebar.checkbox("Allow short selling (weights can be negative)", value=False)

run_btn = st.sidebar.button("Run Optimization", type="primary")

# ---------------- Core functions ----------------
@st.cache_data(ttl=3600)
def fetch_prices(tickers, years):
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data.dropna()

def portfolio_perf(weights, mean_rets, cov):
    ret = np.dot(weights, mean_rets)
    vol = np.sqrt(weights.T @ cov @ weights)
    return ret, vol

def neg_sharpe(weights, mean_rets, cov, rf):
    ret, vol = portfolio_perf(weights, mean_rets, cov)
    return -(ret - rf) / vol

def vol_objective(weights, mean_rets, cov):
    return portfolio_perf(weights, mean_rets, cov)[1]

def optimize_max_sharpe(mean_rets, cov, rf, bounds):
    n = len(mean_rets)
    x0 = np.repeat(1 / n, n)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(neg_sharpe, x0, args=(mean_rets, cov, rf), method="SLSQP",
                    bounds=bounds, constraints=cons)
    return res.x

def optimize_min_vol(mean_rets, cov, bounds, target_return=None):
    n = len(mean_rets)
    x0 = np.repeat(1 / n, n)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if target_return is not None:
        cons.append({"type": "eq",
                      "fun": lambda w: portfolio_perf(w, mean_rets, cov)[0] - target_return})
    res = minimize(vol_objective, x0, args=(mean_rets, cov), method="SLSQP",
                    bounds=bounds, constraints=cons)
    return res

# ---------------- Main pipeline ----------------
if run_btn:
    if len(tickers) < 2:
        st.error("Enter at least 2 tickers.")
        st.stop()

    with st.spinner("Downloading price history..."):
        prices = fetch_prices(tickers, years_back)

    if prices.empty or prices.shape[1] < 2:
        st.error("Could not fetch enough data for these tickers.")
        st.stop()

    assets = list(prices.columns)
    n = len(assets)

    log_returns = np.log(prices / prices.shift(1)).dropna()
    mean_rets = (log_returns.mean() * 252).values
    cov = (log_returns.cov() * 252).values

    st.subheader("1. Historical Statistics")
    stats_df = pd.DataFrame({
        "Annualized Return": mean_rets,
        "Annualized Volatility": np.sqrt(np.diag(cov))
    }, index=assets)
    st.dataframe(stats_df.style.format("{:.2%}"))

    st.subheader("2. Covariance Matrix (Annualized)")
    st.dataframe(pd.DataFrame(cov, index=assets, columns=assets).style.format("{:.5f}"))

    bounds = tuple((None, 1.0) if allow_short else (0.0, 1.0) for _ in range(n))

    # ---- Monte Carlo simulation ----
    with st.spinner(f"Running {n_portfolios:,} Monte Carlo simulations..."):
        rng = np.random.default_rng(42)
        if allow_short:
            raw = rng.normal(size=(n_portfolios, n))
            weights_mc = raw / raw.sum(axis=1, keepdims=True)
        else:
            weights_mc = rng.dirichlet(np.ones(n), size=n_portfolios)

        rets_mc = weights_mc @ mean_rets
        vols_mc = np.sqrt(np.einsum("ij,jk,ik->i", weights_mc, cov, weights_mc))
        sharpe_mc = (rets_mc - risk_free_rate) / vols_mc

    # ---- SLSQP: Max Sharpe & Min Vol ----
    w_max_sharpe = optimize_max_sharpe(mean_rets, cov, risk_free_rate, bounds)
    ret_ms, vol_ms = portfolio_perf(w_max_sharpe, mean_rets, cov)
    sharpe_ms = (ret_ms - risk_free_rate) / vol_ms

    min_vol_res = optimize_min_vol(mean_rets, cov, bounds)
    w_min_vol = min_vol_res.x
    ret_mv, vol_mv = portfolio_perf(w_min_vol, mean_rets, cov)

    # ---- Efficient frontier curve via SLSQP ----
    target_returns = np.linspace(mean_rets.min(), mean_rets.max(), 60)
    frontier_vols = []
    for tr in target_returns:
        res = optimize_min_vol(mean_rets, cov, bounds, target_return=tr)
        frontier_vols.append(res.fun if res.success else np.nan)

    st.subheader("3. Optimal Portfolios (SLSQP)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏆 Maximum Sharpe Ratio Portfolio**")
        w_df = pd.DataFrame({"Weight": w_max_sharpe}, index=assets)
        st.dataframe(w_df.style.format("{:.2%}"))
        st.metric("Expected Return", f"{ret_ms:.2%}")
        st.metric("Volatility", f"{vol_ms:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe_ms:.3f}")
    with col2:
        st.markdown("**🛡️ Minimum Volatility Portfolio**")
        w_df2 = pd.DataFrame({"Weight": w_min_vol}, index=assets)
        st.dataframe(w_df2.style.format("{:.2%}"))
        st.metric("Expected Return", f"{ret_mv:.2%}")
        st.metric("Volatility", f"{vol_mv:.2%}")

    # ---- Plot ----
    st.subheader("4. Efficient Frontier")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vols_mc, y=rets_mc, mode="markers",
        marker=dict(size=3, color=sharpe_mc, colorscale="Viridis",
                    colorbar=dict(title="Sharpe"), showscale=True),
        name="Monte Carlo Portfolios", opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=frontier_vols, y=target_returns, mode="lines",
        line=dict(color="red", width=3), name="Efficient Frontier (SLSQP)"
    ))
    fig.add_trace(go.Scatter(
        x=[vol_ms], y=[ret_ms], mode="markers",
        marker=dict(color="gold", size=16, symbol="star", line=dict(color="black", width=1)),
        name="Max Sharpe Portfolio"
    ))
    fig.add_trace(go.Scatter(
        x=[vol_mv], y=[ret_mv], mode="markers",
        marker=dict(color="blue", size=14, symbol="diamond", line=dict(color="black", width=1)),
        name="Min Volatility Portfolio"
    ))
    fig.update_layout(
        xaxis_title="Annualized Volatility (Risk)",
        yaxis_title="Annualized Expected Return",
        template="plotly_white", height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Gold star = tangency portfolio (max Sharpe). Blue diamond = global minimum-variance portfolio. "
        "Red curve = theoretical efficient frontier solved via constrained SLSQP optimization for each target return."
    )
else:
    st.info("Set your 5 tickers in the sidebar and click **Run Optimization**.")
