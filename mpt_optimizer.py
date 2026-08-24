"""
Modern Portfolio Theory (MPT) Optimizer
-----------------------------------------
Proves: Linear Algebra (covariance/matrix ops), Monte Carlo simulation,
and constrained optimization (SciPy SLSQP) applied to the Markowitz
Efficient Frontier.

Universe: Top 100 US stocks, Top 100 India (NSE) stocks, Top 50 US mutual
funds, and Top 50 India-listed ETFs (substituted for Indian mutual funds,
since Yahoo Finance/yfinance does not carry AMFI mutual fund NAV data).

A hard cap of 100 assets per optimization run is enforced regardless of
how many are selected, to keep Monte Carlo + SLSQP runtime and Yahoo
Finance API load manageable.

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
st.caption("Markowitz Efficient Frontier \u00b7 Monte Carlo Simulation \u00b7 SLSQP Optimization \u00b7 US + India, Stocks + Funds")

MAX_ASSETS = 100  # hard cap per optimization run

# ---------------- Static universes ----------------
US_TOP100 = {
    "NVDA": "NVIDIA", "AAPL": "Apple", "GOOGL": "Alphabet", "MSFT": "Microsoft",
    "AMZN": "Amazon.com", "AVGO": "Broadcom", "TSLA": "Tesla", "META": "Meta Platforms",
    "LLY": "Eli Lilly", "MU": "Micron Technology", "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan Chase",
    "WMT": "Walmart", "AMD": "Advanced Micro Devices", "V": "Visa", "XOM": "Exxon Mobil",
    "JNJ": "Johnson & Johnson", "MA": "Mastercard", "INTC": "Intel", "ABBV": "AbbVie",
    "CSCO": "Cisco Systems", "PLTR": "Palantir Technologies", "BAC": "Bank of America", "ORCL": "Oracle",
    "COST": "Costco Wholesale", "CVX": "Chevron", "KO": "Coca-Cola", "AMAT": "Applied Materials",
    "CAT": "Caterpillar", "MRK": "Merck & Co", "GE": "GE Aerospace", "UNH": "UnitedHealth Group",
    "MS": "Morgan Stanley", "PG": "Procter & Gamble", "HD": "Home Depot", "NFLX": "Netflix",
    "DELL": "Dell Technologies", "GS": "Goldman Sachs Group", "PM": "Philip Morris International", "PANW": "Palo Alto Networks",
    "RTX": "RTX Corp", "WFC": "Wells Fargo", "TXN": "Texas Instruments", "KLAC": "KLA Corp",
    "ANET": "Arista Networks", "AMGN": "Amgen", "SNDK": "SanDisk", "TMO": "Thermo Fisher Scientific",
    "AXP": "American Express", "LIN": "Linde", "IBM": "IBM", "C": "Citigroup",
    "MRVL": "Marvell Technology", "VZ": "Verizon Communications", "TMUS": "T-Mobile US", "PEP": "PepsiCo",
    "CRWD": "CrowdStrike Holdings", "ABT": "Abbott Labs", "SCHW": "Charles Schwab", "APH": "Amphenol",
    "MCD": "McDonald's", "DIS": "Walt Disney", "UNP": "Union Pacific", "SCCO": "Southern Copper",
    "ADI": "Analog Devices", "GILD": "Gilead Sciences", "BLK": "BlackRock", "DE": "Deere & Co",
    "NEE": "NextEra Energy", "T": "AT&T", "WELL": "Welltower", "CRM": "Salesforce",
    "BA": "Boeing", "QCOM": "Qualcomm", "WDC": "Western Digital", "LRCX": "Lam Research",
    "LOW": "Lowe's", "SPGI": "S&P Global", "BKNG": "Booking Holdings", "ADBE": "Adobe",
    "NOW": "ServiceNow", "ISRG": "Intuitive Surgical", "SYK": "Stryker", "TJX": "TJX Companies",
    "VRTX": "Vertex Pharmaceuticals", "PGR": "Progressive Corp", "BSX": "Boston Scientific", "ETN": "Eaton Corp",
    "MMC": "Marsh & McLennan", "PYPL": "PayPal Holdings", "ADP": "Automatic Data Processing", "CB": "Chubb",
    "MDT": "Medtronic", "CI": "Cigna Group", "SO": "Southern Co", "REGN": "Regeneron Pharmaceuticals",
    "DUK": "Duke Energy", "ELV": "Elevance Health", "ICE": "Intercontinental Exchange", "APD": "Air Products",
}

INDIA_TOP100 = {
    "RELIANCE.NS": "Reliance Industries", "BHARTIARTL.NS": "Bharti Airtel", "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank", "SBIN.NS": "State Bank of India", "TCS.NS": "Tata Consultancy Services",
    "BAJFINANCE.NS": "Bajaj Finance", "LT.NS": "Larsen & Toubro", "HINDUNILVR.NS": "Hindustan Unilever",
    "SUNPHARMA.NS": "Sun Pharmaceutical", "INFY.NS": "Infosys", "TITAN.NS": "Titan Company",
    "ADANIENT.NS": "Adani Enterprises", "MARUTI.NS": "Maruti Suzuki", "M&M.NS": "Mahindra & Mahindra",
    "KOTAKBANK.NS": "Kotak Mahindra Bank", "ADANIPOWER.NS": "Adani Power", "ADANIPORTS.NS": "Adani Ports & SEZ",
    "AXISBANK.NS": "Axis Bank", "HCLTECH.NS": "HCL Technologies", "ITC.NS": "ITC", "NTPC.NS": "NTPC",
    "ULTRACEMCO.NS": "UltraTech Cement", "LICI.NS": "Life Insurance Corp of India", "WIPRO.NS": "Wipro",
    "ONGC.NS": "Oil & Natural Gas Corp", "POWERGRID.NS": "Power Grid Corp", "BAJAJFINSV.NS": "Bajaj Finserv",
    "ASIANPAINT.NS": "Asian Paints", "NESTLEIND.NS": "Nestle India", "COALINDIA.NS": "Coal India",
    "TATAMOTORS.NS": "Tata Motors", "JSWSTEEL.NS": "JSW Steel", "TATASTEEL.NS": "Tata Steel", "GRASIM.NS": "Grasim Industries",
    "TECHM.NS": "Tech Mahindra", "HINDALCO.NS": "Hindalco Industries", "BAJAJ-AUTO.NS": "Bajaj Auto",
    "DRREDDY.NS": "Dr. Reddy's Labs", "CIPLA.NS": "Cipla", "EICHERMOT.NS": "Eicher Motors", "APOLLOHOSP.NS": "Apollo Hospitals",
    "SBILIFE.NS": "SBI Life Insurance", "HDFCLIFE.NS": "HDFC Life Insurance", "DIVISLAB.NS": "Divi's Laboratories",
    "BRITANNIA.NS": "Britannia Industries", "INDUSINDBK.NS": "IndusInd Bank", "VEDL.NS": "Vedanta",
    "SHREECEM.NS": "Shree Cement", "PIDILITIND.NS": "Pidilite Industries", "HAVELLS.NS": "Havells India",
    "DABUR.NS": "Dabur India", "GODREJCP.NS": "Godrej Consumer Products", "TATACONSUM.NS": "Tata Consumer Products",
    "AMBUJACEM.NS": "Ambuja Cements", "BANKBARODA.NS": "Bank of Baroda", "PNB.NS": "Punjab National Bank",
    "CANBK.NS": "Canara Bank", "IOC.NS": "Indian Oil Corp", "BPCL.NS": "Bharat Petroleum",
    "GAIL.NS": "GAIL India", "SIEMENS.NS": "Siemens Ltd", "ABB.NS": "ABB India", "CGPOWER.NS": "CG Power & Industrial",
    "SOLARINDS.NS": "Solar Industries India", "HAL.NS": "Hindustan Aeronautics", "BEL.NS": "Bharat Electronics",
    "MAZDOCK.NS": "Mazagon Dock Shipbuilders", "IRFC.NS": "Indian Railway Finance Corp", "ZOMATO.NS": "Eternal (Zomato)",
    "PAYTM.NS": "One 97 Communications", "NYKAA.NS": "FSN E-Commerce (Nykaa)", "DMART.NS": "Avenue Supermarts",
    "TRENT.NS": "Trent Ltd", "PGHH.NS": "Procter & Gamble Hygiene", "COLPAL.NS": "Colgate-Palmolive India",
    "MCDOWELL-N.NS": "United Spirits", "VBL.NS": "Varun Beverages", "PAGEIND.NS": "Page Industries",
    "MOTHERSON.NS": "Samvardhana Motherson", "BOSCHLTD.NS": "Bosch Ltd", "TVSMOTOR.NS": "TVS Motor Co",
    "HEROMOTOCO.NS": "Hero MotoCorp", "BALKRISIND.NS": "Balkrishna Industries", "MRF.NS": "MRF Ltd",
    "LUPIN.NS": "Lupin Ltd", "AUROPHARMA.NS": "Aurobindo Pharma", "TORNTPHARM.NS": "Torrent Pharmaceuticals",
    "ALKEM.NS": "Alkem Laboratories", "BIOCON.NS": "Biocon Ltd", "MPHASIS.NS": "Mphasis Ltd",
    "LTIM.NS": "LTIMindtree", "PERSISTENT.NS": "Persistent Systems", "COFORGE.NS": "Coforge Ltd",
    "OBEROIRLTY.NS": "Oberoi Realty", "DLF.NS": "DLF Ltd", "GODREJPROP.NS": "Godrej Properties",
    "INDIGO.NS": "InterGlobe Aviation", "JUBLFOOD.NS": "Jubilant FoodWorks", "POLYCAB.NS": "Polycab India",
    "ASTRAL.NS": "Astral Ltd", "SRF.NS": "SRF Ltd", "UPL.NS": "UPL Ltd", "PIIND.NS": "PI Industries",
}

US_FUNDS_50 = {
    "VTSAX": "Vanguard Total Stock Market Index", "VFIAX": "Vanguard 500 Index", "FXAIX": "Fidelity 500 Index", "VTIAX": "Vanguard Total International Stock Index",
    "VBTLX": "Vanguard Total Bond Market Index", "SPAXX": "Fidelity Government Money Market", "VIGAX": "Vanguard Growth Index", "FCNTX": "Fidelity Contrafund",
    "FZCXX": "Fidelity Government Cash Reserves", "VINIX": "Vanguard Institutional Index", "VTWAX": "Vanguard Total World Stock Index", "VWENX": "Vanguard Wellington",
    "VBIAX": "Vanguard Balanced Index", "VIMAX": "Vanguard Mid-Cap Index", "FCTDX": "Strategic Advisers Fidelity US Total Stock", "VVIAX": "Vanguard Value Index",
    "DODGX": "Dodge & Cox Stock", "VSMAX": "Vanguard Small-Cap Index", "VTABX": "Vanguard International Bond Index", "TRBCX": "T. Rowe Price Blue Chip Growth",
    "AGTHX": "American Funds Growth Fund of America", "AIVSX": "American Funds Investment Co of America", "AWSHX": "American Funds Washington Mutual", "ANCFX": "American Funds Fundamental Investors",
    "ANWPX": "American Funds New Perspective", "CWGIX": "American Funds Capital World Growth & Income", "PTTRX": "PIMCO Total Return", "VWNDX": "Vanguard Windsor",
    "VWELX": "Vanguard Wellington Investor", "VGSLX": "Vanguard Real Estate Index", "VEXAX": "Vanguard Extended Market Index", "VTMFX": "Vanguard Tax-Managed Balanced",
    "VHGEX": "Vanguard Global Equity", "VEMAX": "Vanguard Emerging Markets Stock Index", "VDIGX": "Vanguard Dividend Growth", "VWUSX": "Vanguard US Growth",
    "VPMAX": "Vanguard PRIMECAP", "FDGRX": "Fidelity Growth Company", "FBGRX": "Fidelity Blue Chip Growth", "FLPSX": "Fidelity Low-Priced Stock",
    "FDVLX": "Fidelity Value", "FMAGX": "Fidelity Magellan", "FSKAX": "Fidelity Total Market Index", "FTIHX": "Fidelity Total International Index",
    "FSPGX": "Fidelity Large Cap Growth Index", "FXNAX": "Fidelity US Bond Index", "JABAX": "Janus Henderson Balanced", "JAENX": "Janus Henderson Enterprise",
    "MEIKX": "MFS Massachusetts Investors Trust", "OAKBX": "Oakmark Equity & Income",
}

# Note: Yahoo Finance / yfinance does not carry Indian mutual fund NAVs
# (those live with AMFI, not Yahoo). This bucket uses India-listed ETFs
# instead, which trade on NSE with real tickers yfinance can fetch.
INDIA_ETFS_50 = {
    "NIFTYBEES.NS": "Nippon India ETF Nifty BeES", "JUNIORBEES.NS": "Nippon India ETF Junior BeES", "BANKBEES.NS": "Nippon India ETF Bank BeES", "GOLDBEES.NS": "Nippon India ETF Gold BeES",
    "ICICINIFTY.NS": "ICICI Prudential Nifty ETF", "HDFCNIFTY.NS": "HDFC Nifty 50 ETF", "SBINIFTY.NS": "SBI Nifty 50 ETF", "UTINIFTETF.NS": "UTI Nifty 50 ETF",
    "KOTAKNIFTY.NS": "Kotak Nifty ETF", "AXISNIFTY.NS": "Axis Nifty ETF", "ICICIB22.NS": "ICICI Prudential Bharat 22 ETF", "ITBEES.NS": "Nippon India ETF Nifty IT",
    "PSUBNKBEES.NS": "Nippon India ETF PSU Bank BeES", "MON100.NS": "Motilal Oswal Nasdaq 100 ETF", "MAFANG.NS": "Mirae Asset NYSE FANG+ ETF", "CPSE.NS": "CPSE ETF",
    "LIQUIDBEES.NS": "Nippon India ETF Liquid BeES", "SILVERBEES.NS": "Nippon India ETF Silver BeES", "HNGSNGBEES.NS": "Nippon India ETF Hang Seng BeES", "INFRABEES.NS": "Nippon India ETF Infra BeES",
    "PVTBANKIETF.NS": "ICICI Prudential Private Banks ETF", "MIDCAPETF.NS": "Motilal Oswal Midcap 150 ETF", "NV20.NS": "Nippon India ETF Nifty 50 Value 20", "LOWVOLIETF.NS": "ICICI Prudential Nifty Low Vol 30 ETF",
    "ALPHA.NS": "Nippon India ETF Nifty Alpha 50", "QUAL30IETF.NS": "ICICI Prudential Nifty200 Quality 30 ETF", "MOM30IETF.NS": "ICICI Prudential Nifty200 Momentum 30 ETF", "HDFCSML250.NS": "HDFC Nifty Smallcap 250 ETF",
    "MOM100.NS": "Motilal Oswal Nifty Midcap 100 ETF", "NEXT50IETF.NS": "ICICI Prudential Nifty Next 50 ETF", "SETFNIF50.NS": "SBI ETF Nifty 50", "SETFNIFBK.NS": "SBI ETF Nifty Bank",
    "HDFCPVTBAN.NS": "HDFC Nifty Private Bank ETF", "GOLDSHARE.NS": "UTI Gold ETF", "AXISGOLD.NS": "Axis Gold ETF", "SBIETFQLTY.NS": "SBI Nifty Quality ETF",
    "ABSLNN50ET.NS": "Aditya Birla SL Nifty 50 ETF", "EQUAL50.NS": "Nippon India Nifty 50 Equal Weight ETF", "MOSMALL250.NS": "Motilal Oswal Nifty Smallcap 250 ETF", "MOHEALTH.NS": "Motilal Oswal Healthcare ETF",
    "MOMOMENTUM.NS": "Motilal Oswal Nifty200 Momentum 30 ETF", "LICNETFSEN.NS": "LIC MF Sensex ETF", "HDFCSENSEX.NS": "HDFC Sensex ETF", "SBISENSEX.NS": "SBI ETF Sensex",
    "UTISENSETF.NS": "UTI Sensex ETF", "TATAGOLD.NS": "Tata Gold ETF", "QGOLDHALF.NS": "Quantum Gold Fund", "AUTOBEES.NS": "Nippon India ETF Nifty Auto",
    "CONSUMBEES.NS": "Nippon India ETF Nifty India Consumption", "PHARMABEES.NS": "Nippon India ETF Nifty Pharma", "METALIETF.NS": "ICICI Prudential Nifty Metal ETF", "FMCGIETF.NS": "ICICI Prudential Nifty FMCG ETF",
}

ALL_UNIVERSES = {
    "Top 100 US Stocks": US_TOP100,
    "Top 100 India (NSE) Stocks": INDIA_TOP100,
    "Top 50 US Mutual Funds": US_FUNDS_50,
    "Top 50 India ETFs (fund substitute)": INDIA_ETFS_50,
}
NAME_LOOKUP = {}
for _u in ALL_UNIVERSES.values():
    NAME_LOOKUP.update(_u)

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Configuration")
st.sidebar.caption(
    "\u26A0\uFE0F Indian mutual funds (AMFI NAVs) are not available on Yahoo Finance / yfinance. "
    "The 'India ETFs' bucket substitutes NSE-listed ETFs, which do have real tickers and price history."
)

bucket_choices = st.sidebar.multiselect(
    "Select one or more universes to pull tickers from",
    list(ALL_UNIVERSES.keys()),
    default=["Top 100 US Stocks"]
)

candidate_map = {}
for b in bucket_choices:
    candidate_map.update(ALL_UNIVERSES[b])

if candidate_map:
    options = [f"{tk} \u2014 {name}" for tk, name in candidate_map.items()]
    selected_labels = st.sidebar.multiselect(
        f"Select assets ({len(options)} available across chosen universes)",
        options, default=options
    )
    tickers = [lbl.split(" \u2014 ")[0] for lbl in selected_labels]
else:
    tickers = []

use_custom = st.sidebar.checkbox("Add custom tickers manually")
if use_custom:
    custom_input = st.sidebar.text_area(
        "Extra tickers, comma-separated (.NS suffix for NSE stocks, e.g. INFY.NS)", ""
    )
    extra = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers + extra))

n_selected = len(tickers)
if n_selected > MAX_ASSETS:
    st.sidebar.error(
        f"You selected {n_selected} assets, but only {MAX_ASSETS} can be optimized per run. "
        f"The first {MAX_ASSETS} (in selection order) will be used \u2014 narrow your selection for full control."
    )
    tickers = tickers[:MAX_ASSETS]

st.sidebar.caption(f"**{len(tickers)} / {MAX_ASSETS} assets selected for this run**")

years_back = st.sidebar.slider("Years of history", 1, 10, 3)
n_portfolios = st.sidebar.number_input("Monte Carlo simulations", 1000, 200000, 50000, step=1000)
risk_free_rate = st.sidebar.number_input("Risk-free rate (annual, decimal)", 0.0, 0.20, 0.065, step=0.005)
allow_short = st.sidebar.checkbox("Allow short selling (weights can be negative)", value=False)
max_weight = st.sidebar.slider("Max weight per asset (concentration cap)", 0.05, 1.0, 1.0, step=0.05)

run_btn = st.sidebar.button("Run Optimization", type="primary")

# ---------------- Data fetching (robust to Yahoo rate limits) ----------------
CHUNK_SIZE = 15
MAX_RETRIES = 3
BASE_BACKOFF = 2.0

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
    for attempt in range(MAX_RETRIES):
        try:
            data = yf.download(chunk, start=start, end=end, auto_adjust=True,
                                progress=False, group_by="ticker", threads=True)
            if data is not None and not data.empty:
                return data
        except Exception:
            pass
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
    stats_df = pd.DataFrame({
        "Name": [NAME_LOOKUP.get(a, a) for a in assets],
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
    fig_corr.update_layout(height=min(22 * n + 100, 900), template="plotly_white")
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
        f"Universe: {n} assets. Mixed currencies (USD/INR) are not FX-adjusted; returns/volatility are computed "
        "independently per asset's native price series."
    )
else:
    st.info("Choose one or more universes and assets in the sidebar (max 100 per run), then click **Run Optimization**.")
    st.markdown(
        "**Available universes:**\n"
        "- **Top 100 US Stocks** by market cap (NVDA, AAPL, MSFT, GOOGL...)\n"
        "- **Top 100 India (NSE) Stocks** by market cap (RELIANCE.NS, HDFCBANK.NS, TCS.NS...)\n"
        "- **Top 50 US Mutual Funds** by AUM (VTSAX, VFIAX, FXAIX...)\n"
        "- **Top 50 India ETFs** (substituting Indian mutual funds, which have no Yahoo Finance data \u2014 "
        "NIFTYBEES.NS, GOLDBEES.NS, BANKBEES.NS...)\n\n"
        "You can mix and match across universes, or add custom tickers, up to a **100-asset cap** per optimization run."
    )
