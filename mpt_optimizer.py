"""
Modern Portfolio Theory (MPT) Optimizer
-----------------------------------------
Proves: Linear Algebra (covariance/matrix ops), Monte Carlo simulation,
and constrained optimization (SciPy SLSQP) applied to the Markowitz
Efficient Frontier.

Supports a large universe: Top 40 US stocks by market cap + Top 20 India
(NSE) stocks by market cap, or any custom ticker list.

Run locally:
    pip install streamlit yfinance numpy pandas scipy plotly
    streamlit run mpt_optimizer.py

Deploy free & public:
    1. Push this file + requirements.txt to a public GitHub repo.
    2. Go to https://share.streamlit.io -> "New app" -> point to the repo.
    3. Streamlit Community Cloud builds and hosts it at a public URL for free.

Note on yfinance reliability:
    yfinance scrapes Yahoo Finance (no official API), so it is occasionally
    rate-limited (HTTP 429 "Too Many Requests"), especially for large batch
    requests or on shared cloud IPs like Streamlit Community Cloud. This file
    mitigates that with: batched chunked downloads, retry-with-backoff, a
    per-ticker fallback, and 24h caching so repeated runs don't re-hit the API.
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="MPT Optimizer", layout="wide")
st.title("\U0001F4C8 Modern Portfolio Theory (MPT) Optimizer")
st.caption("Markowitz Efficient Frontier \u00b7 Monte Carlo Simulation \u00b7 SLSQP Optimization \u00b7 US + India Universe")

# ---------------- Static universes ----------------
US_TOP40 = {
    "NVDA": "NVIDIA", "AAPL": "Apple", "GOOGL": "Alphabet", "MSFT": "Microsoft",
    "AMZN": "Amazon", "AVGO": "Broadcom", "META": "Meta Platforms", "TSLA": "Tesla",
    "MU": "Micron Technology", "LLY": "Eli Lilly", "BRK-B": "Berkshire Hathaway",
    "WMT": "Walmart", "AMD": "Advanced Micro Devices", "JPM": "JPMorgan Chase",
    "ORCL": "Oracle", "XOM": "Exxon Mobil", "V": "Visa", "INTC": "Intel",
    "JNJ": "Johnson & Johnson", "CSCO": "Cisco Systems", "MA": "Mastercard",
    "HD": "Home Depot", "PG": "Procter & Gamble", "COST": "Costco",
    "ABBV": "AbbVie", "NFLX": "Netflix", "BAC": "Bank of America", "KO": "Coca-Cola",
    "PEP": "PepsiCo", "CVX": "Chevron", "TMO": "Thermo Fisher", "MRK": "Merck",
    "ADBE": "Adobe", "CRM": "Salesforce", "ACN": "Accenture", "LIN": "Linde",
    "MCD": "McDonald's", "ABT": "Abbott Labs", "WFC": "Wells Fargo", "DIS": "Disney",
}

INDIA_TOP20 = {
    "RELIANCE.NS": "Reliance Industries", "BHARTIARTL.NS": "Bharti Airtel",
    "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India", "TCS.NS": "Tata Consultancy Services",
    "BAJFINANCE.NS": "Bajaj Finance", "LT.NS": "Larsen & Toubro",
    "LICI.NS": "Life Insurance Corp of India", "HINDUNILVR.NS": "Hindustan Unilever",
    "INFY.NS": "Infosys", "ITC.NS": "ITC", "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "AXISBANK.NS": "Axis Bank", "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharma", "HCLTECH.NS": "HCL Technologies",
    "ADANIENT.NS": "Adani Enterprises", "ULTRACEMCO.NS": "UltraTech Cement",
    "NTPC.NS": "NTPC",
}

FULL_UNIVERSE = {**US_TOP40, **INDIA_TOP20}

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Configuration")

universe_choice = st.sidebar.radio(
    "Ticker universe",
    ["Top 40 US + Top 20 India (60 total)", "Top 40 US only", "Top 20 India only", "Custom list"],
)

if universe_choice == "Top 40 US + Top 20 India (60 total)":
    candidate_map = FULL_UNIVERSE
elif universe_choice == "Top 40 US only":
    candidate_map = US_TOP40
elif universe_choice == "Top 20 India only":
    candidate_map = INDIA_TOP20
else:
    candidate_map = None

if candidate_map is not None:
    options = [f"{tk} \u2014 {name}" for tk, name in candidate_map.items()]
    default_sel = options  # select all by default
    selected_labels = st.sidebar.multiselect(
        "Select assets (defaults to full list; deselect to narrow down)",
        options, default=default_sel
    )
    tickers = [lbl.split(" \u2014 ")[0] for lbl in selected_labels]
else:
    custom_input = st.sidebar.text_area(
        "Enter tickers, comma-separated (use .NS suffix for NSE-listed India stocks, e.g. INFY.NS)",
        "AAPL, MSFT, JPM, JNJ, XOM"
    )
    tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]

st.sidebar.caption(f"**{len(tickers)} assets selected**")

years_back = st.sidebar.slider("Years of history", 1, 10, 3)
n_portfolios = st.sidebar.number_input("Monte Carlo simulations", 1000, 200000, 50000, step=1000)
risk_free_rate = st.sidebar.number_input("Risk-free rate (annual, decimal)", 0.0, 0.20, 0.065, step=0.005)
allow_short = st.sidebar.checkbox("Allow short selling (weights can be negative)", value=False)
max_weight = st.sidebar.slider("Max weight per asset (concentration cap)", 0.05, 1.0, 1.0, step=0.05)

run_btn = st.sidebar.button("Run Optimization", type="primary")

# ---------------- Data fetching (robust to Yahoo rate limits) ----------------
CHUNK_SIZE = 15          # keep batch requests small to reduce 429s
MAX_RETRIES = 3
BASE_BACKOFF = 2.0        # seconds, doubles each retry

def _extract_close(df, tk):
    if df is None or df.empty:
        return None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if (tk, "Close") in df.columns:
                return df[(tk, "Close")]
            lvl0 = df.columns.get_level_values(0)
            if tk in lvl0:
                sub = df[tk]
                if "Close" in sub.columns:
                    return sub["Close"]
        else:
            if "Close" in df.columns:
                return df["Close"]
    except Exception:
        return None
    return None

def _download_chunk(chunk, start, end):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            data = yf.download(chunk, start=start, end=end, auto_adjust=True,
                                progress=False, group_by="ticker", threads=True)
            if data is not None and not data.empty:
                return data
        except Exception as e:
            last_err = e
        time.sleep(BASE_BACKOFF * (2 ** attempt))
    return None

def _download_single(tk, start, end):
    for attempt in range(MAX_RETRIES):
        try:
            hist = yf.Ticker(tk).history(start=start, end=end, auto_adjust=True)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                s = hist["Close"].copy()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                return s
        except Exception:
            pass
        time.sleep(BASE_BACKOFF * (2 ** attempt))
    return None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_prices(tickers, years):
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)

    series_map = {}
    remaining = list(tickers)

    for i in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[i:i + CHUNK_SIZE]
        data = _download_chunk(chunk, start, end)
        if data is None:
            continue
        for tk in chunk:
            s = _extract_close(data, tk) if len(chunk) > 1 else (
                data["Close"] if "Close" in data.columns else None
            )
            if s is not None and not s.dropna().empty:
                s = s.copy()
                s.index = pd.to_datetime(s.index).tz_localize(None) if s.index.tz is not None else s.index
                series_map[tk] = s
        time.sleep(1.0)

    missing = [t for t in tickers if t not in series_map]
    for tk in missing:
        s = _download_single(tk, start, end)
        if s is not None and not s.dropna().empty:
            series_map[tk] = s
        time.sleep(0.5)

    if not series_map:
        return pd.DataFrame()

    prices = pd.DataFrame(series_map)
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill().dropna(how="any")
    return prices

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
                    bounds=bounds, constraints=cons, options={"maxiter": 500})
    return res.x

def optimize_min_vol(mean_rets, cov, bounds, target_return=None):
    n = len(mean_rets)
    x0 = np.repeat(1 / n, n)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if target_return is not None:
        cons.append({"type": "eq",
                      "fun": lambda w: portfolio_perf(w, mean_rets, cov)[0] - target_return})
    res = minimize(vol_objective, x0, args=(mean_rets, cov), method="SLSQP",
                    bounds=bounds, constraints=cons, options={"maxiter": 500})
    return res

# ---------------- Main pipeline ----------------
if run_btn:
    if len(tickers) < 2:
        st.error("Select at least 2 tickers.")
        st.stop()

    with st.spinner(f"Downloading price history for {len(tickers)} tickers (chunked, with retries)..."):
        prices = fetch_prices(tuple(tickers), years_back)

    dropped = set(tickers) - set(prices.columns if not prices.empty else [])
    if dropped:
        st.warning(
            f"Could not fetch data for: {', '.join(sorted(dropped))} \u2014 excluded from optimization. "
            "Yahoo Finance sometimes rate-limits large batch requests (HTTP 429), especially on shared cloud "
            "IPs. Click **Run Optimization** again in 30\u201360 seconds, or reduce the number of selected tickers."
        )

    if prices.empty or prices.shape[1] < 2:
        st.error(
            "Could not fetch enough price data. This is almost always Yahoo Finance temporarily rate-limiting "
            "requests, not a bug in the app. Wait about a minute and click **Run Optimization** again, try a "
            "smaller ticker selection, or reduce Monte Carlo simulations to retry faster."
        )
        st.stop()

    assets = list(prices.columns)
    n = len(assets)

    log_returns = np.log(prices / prices.shift(1)).dropna()
    mean_rets = (log_returns.mean() * 252).values
    cov = (log_returns.cov() * 252).values
    corr = log_returns.corr().values

    st.subheader(f"1. Historical Statistics ({n} assets)")
    name_lookup = {**US_TOP40, **INDIA_TOP20}
    stats_df = pd.DataFrame({
        "Name": [name_lookup.get(a, a) for a in assets],
        "Annualized Return": mean_rets,
        "Annualized Volatility": np.sqrt(np.diag(cov))
    }, index=assets)
    st.dataframe(
        stats_df.style.format({"Annualized Return": "{:.2%}", "Annualized Volatility": "{:.2%}"}),
        height=min(35 * n + 40, 500)
    )

    st.subheader("2. Correlation Matrix (Heatmap)")
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr, x=assets, y=assets, colorscale="RdBu", zmid=0,
        colorbar=dict(title="Corr")
    ))
    fig_corr.update_layout(height=min(28 * n + 100, 800), template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption(
        "Full annualized covariance is used internally for optimization; this heatmap shows "
        "pairwise correlation for readability at this asset count."
    )

    bounds = tuple((None, max_weight) if allow_short else (0.0, max_weight) for _ in range(n))

    with st.spinner(f"Running {n_portfolios:,} Monte Carlo simulations across {n} assets..."):
        rng = np.random.default_rng(42)
        if allow_short:
            raw = rng.normal(size=(n_portfolios, n))
            weights_mc = raw / raw.sum(axis=1, keepdims=True)
        else:
            weights_mc = rng.dirichlet(np.ones(n), size=n_portfolios)

        rets_mc = weights_mc @ mean_rets
        vols_mc = np.sqrt(np.einsum("ij,jk,ik->i", weights_mc, cov, weights_mc))
        sharpe_mc = (rets_mc - risk_free_rate) / vols_mc

    with st.spinner("Solving SLSQP optimizations..."):
        w_max_sharpe = optimize_max_sharpe(mean_rets, cov, risk_free_rate, bounds)
        ret_ms, vol_ms = portfolio_perf(w_max_sharpe, mean_rets, cov)
        sharpe_ms = (ret_ms - risk_free_rate) / vol_ms

        min_vol_res = optimize_min_vol(mean_rets, cov, bounds)
        w_min_vol = min_vol_res.x
        ret_mv, vol_mv = portfolio_perf(w_min_vol, mean_rets, cov)

        target_returns = np.linspace(mean_rets.min(), mean_rets.max(), 50)
        frontier_vols = []
        for tr in target_returns:
            res = optimize_min_vol(mean_rets, cov, bounds, target_return=tr)
            frontier_vols.append(res.fun if res.success else np.nan)

    st.subheader("3. Optimal Portfolios (SLSQP)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**\U0001F3C6 Maximum Sharpe Ratio Portfolio**")
        w_df = pd.DataFrame({"Weight": w_max_sharpe}, index=assets).sort_values("Weight", ascending=False)
        w_df_top = w_df[w_df["Weight"] > 0.001]
        st.dataframe(w_df_top.style.format("{:.2%}"), height=min(35 * len(w_df_top) + 40, 400))
        st.metric("Expected Return", f"{ret_ms:.2%}")
        st.metric("Volatility", f"{vol_ms:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe_ms:.3f}")
    with col2:
        st.markdown("**\U0001F6E1\uFE0F Minimum Volatility Portfolio**")
        w_df2 = pd.DataFrame({"Weight": w_min_vol}, index=assets).sort_values("Weight", ascending=False)
        w_df2_top = w_df2[w_df2["Weight"] > 0.001]
        st.dataframe(w_df2_top.style.format("{:.2%}"), height=min(35 * len(w_df2_top) + 40, 400))
        st.metric("Expected Return", f"{ret_mv:.2%}")
        st.metric("Volatility", f"{vol_mv:.2%}")

    st.subheader("4. Efficient Frontier")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vols_mc, y=rets_mc, mode="markers",
        marker=dict(size=3, color=sharpe_mc, colorscale="Viridis",
                    colorbar=dict(title="Sharpe"), showscale=True),
        name="Monte Carlo Portfolios", opacity=0.5
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
        "Red curve = theoretical efficient frontier solved via constrained SLSQP optimization for each target return. "
        f"Universe: {n} assets \u2014 US large caps priced in USD, India (NSE) stocks priced in INR; "
        "returns/volatility are computed independently per currency and not FX-adjusted."
    )
else:
    st.info("Choose a universe and tickers in the sidebar, then click **Run Optimization**.")
    st.markdown(
        "This app ships with two built-in universes: the **Top 40 US stocks** by market cap "
        "(NVDA, AAPL, MSFT, GOOGL, AMZN...) and the **Top 20 India (NSE) stocks** by market cap "
        "(RELIANCE.NS, HDFCBANK.NS, TCS.NS...), or you can combine both for a 60-asset universe, "
        "or type in any custom ticker list."
    )
