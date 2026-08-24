"""
Modern Portfolio Theory (MPT) Optimizer
-----------------------------------------
Proves: Linear Algebra (covariance/matrix ops), Monte Carlo simulation,
and constrained optimization (SciPy SLSQP) applied to the Markowitz
Efficient Frontier, plus out-of-sample backtesting against a benchmark.

Universe: Top 100 US stocks, Top 100 India (NSE) stocks, Top 50 US mutual
funds, and Top 50 India-listed ETFs.

Ticker rename notes (Yahoo Finance only serves data under CURRENT symbols):
- Marsh McLennan: MMC -> MRSH, effective Jan 14, 2026
- United Spirits: MCDOWELL-N.NS -> UNITDSPR.NS
- Tata Motors (passenger vehicle business): TATAMOTORS.NS -> TMPV.NS
- Zomato: ZOMATO.NS -> ETERNAL.NS
- LTIMindtree: LTIM.NS -> LTM.NS, effective Feb 11, 2026

Fund universe note: SPAXX and FZCXX (Fidelity money market funds) were
removed. Money market funds have a fixed $1.00 NAV by SEC design (Rule
2a-7), meaning zero day-to-day price variance -- this breaks mean-variance
optimization, which requires meaningful volatility to trade off against
return, and produces a singular (non-invertible) covariance matrix.
Replaced with PRGFX and PRWCX (T. Rowe Price funds with real price movement).

Search UX:
    All 300 tickers across every universe are combined into ONE searchable
    list with Streamlit's built-in fuzzy-search-as-you-type.

A hard cap of 100 assets per optimization run is enforced.

Data policy \u2014 full history, always growing, never shrinking:
    Fetches each ticker's FULL available price history (period="max"),
    cached 24h, automatically extending as new years pass.

Custom analysis window (years + as-of date):
    Pick both a number of years AND an "as of" end date for point-in-time
    historical analysis, slicing the cached full history without discarding it.
    The "years" input allows half-year steps (e.g. 2.5) -- this is converted
    to a day-count offset (years * 365.25) rather than passed to
    pd.DateOffset(years=...), since pandas/dateutil raise a ValueError on
    non-integer year offsets ("Non-integer years and months are ambiguous").
    (Note: no error is raised if the actual usable window is shorter than
    requested due to a short-history ticker -- the history-length table
    in the Data Synopsis surfaces this information instead, without an
    intrusive error banner.)

Shortest-history diagnostics (full history-length table):
    Because all tickers must be aligned to a common date range, the analysis
    window is capped by whichever SELECTED ticker has the SHORTEST available
    history. The app shows a full table of EVERY selected ticker's available
    history length, sorted shortest-first.

Backtesting:
    Splits the analysis window into an in-sample training period and an
    out-of-sample test period, freezing weights trained only on the past
    and comparing against an equal-weight benchmark. Test-period % returns
    for every strategy are shown in a clearly labeled table.

Plain-language chart summaries:
    After the efficient frontier chart and (if enabled) the backtest chart,
    the app auto-generates a short, jargon-free written takeaway.

Run locally:
    pip install streamlit yfinance numpy pandas scipy plotly matplotlib
    streamlit run mpt_optimizer.py

Deploy free & public:
    1. Push this file + requirements.txt to a public GitHub repo.
    2. Go to https://share.streamlit.io -> "New app" -> point to the repo.
    3. Streamlit Community Cloud builds and hosts it at a public URL for free.
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
st.caption("Markowitz Efficient Frontier \u00b7 Monte Carlo Simulation \u00b7 SLSQP Optimization \u00b7 Out-of-Sample Backtesting")

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
    "MRSH": "Marsh McLennan", "PYPL": "PayPal Holdings", "ADP": "Automatic Data Processing", "CB": "Chubb",
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
    "TMPV.NS": "Tata Motors Passenger Vehicles", "JSWSTEEL.NS": "JSW Steel", "TATASTEEL.NS": "Tata Steel", "GRASIM.NS": "Grasim Industries",
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
    "MAZDOCK.NS": "Mazagon Dock Shipbuilders", "IRFC.NS": "Indian Railway Finance Corp", "ETERNAL.NS": "Eternal (Zomato)",
    "PAYTM.NS": "One 97 Communications", "NYKAA.NS": "FSN E-Commerce (Nykaa)", "DMART.NS": "Avenue Supermarts",
    "TRENT.NS": "Trent Ltd", "PGHH.NS": "Procter & Gamble Hygiene", "COLPAL.NS": "Colgate-Palmolive India",
    "UNITDSPR.NS": "United Spirits", "VBL.NS": "Varun Beverages", "PAGEIND.NS": "Page Industries",
    "MOTHERSON.NS": "Samvardhana Motherson", "BOSCHLTD.NS": "Bosch Ltd", "TVSMOTOR.NS": "TVS Motor Co",
    "HEROMOTOCO.NS": "Hero MotoCorp", "BALKRISIND.NS": "Balkrishna Industries", "MRF.NS": "MRF Ltd",
    "LUPIN.NS": "Lupin Ltd", "AUROPHARMA.NS": "Aurobindo Pharma", "TORNTPHARM.NS": "Torrent Pharmaceuticals",
    "ALKEM.NS": "Alkem Laboratories", "BIOCON.NS": "Biocon Ltd", "MPHASIS.NS": "Mphasis Ltd",
    "LTM.NS": "LTM Limited (formerly LTIMindtree)", "PERSISTENT.NS": "Persistent Systems", "COFORGE.NS": "Coforge Ltd",
    "OBEROIRLTY.NS": "Oberoi Realty", "DLF.NS": "DLF Ltd", "GODREJPROP.NS": "Godrej Properties",
    "INDIGO.NS": "InterGlobe Aviation", "JUBLFOOD.NS": "Jubilant FoodWorks", "POLYCAB.NS": "Polycab India",
}

US_FUNDS_50 = {
    "VTSAX": "Vanguard Total Stock Market Index", "VFIAX": "Vanguard 500 Index", "FXAIX": "Fidelity 500 Index", "VTIAX": "Vanguard Total International Stock Index",
    "VBTLX": "Vanguard Total Bond Market Index", "PRGFX": "T. Rowe Price Growth Stock", "VIGAX": "Vanguard Growth Index", "FCNTX": "Fidelity Contrafund",
    "PRWCX": "T. Rowe Price Capital Appreciation", "VINIX": "Vanguard Institutional Index", "VTWAX": "Vanguard Total World Stock Index", "VWENX": "Vanguard Wellington",
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
    "CONSUMBEES.NS": "Nippon India ETF Nifty India Consumption", "PHARMABEES.NS": "Nippon India ETF Nifty Pharma",
}

_universe_sizes = {"US_TOP100": (US_TOP100, 100), "INDIA_TOP100": (INDIA_TOP100, 100),
                    "US_FUNDS_50": (US_FUNDS_50, 50), "INDIA_ETFS_50": (INDIA_ETFS_50, 50)}
for _name, (_d, _expected) in _universe_sizes.items():
    if len(_d) != _expected:
        st.error(f"Configuration error: {_name} has {len(_d)} entries, expected {_expected}. "
                 "Please report this \u2014 the app cannot run correctly until this list is fixed.")
        st.stop()

ALL_UNIVERSES = {
    "Top 100 US Stocks": US_TOP100,
    "Top 100 India (NSE) Stocks": INDIA_TOP100,
    "Top 50 US Mutual Funds": US_FUNDS_50,
    "Top 50 India ETFs": INDIA_ETFS_50,
}
NAME_LOOKUP = {}
TICKER_UNIVERSE_LABEL = {}
for _uname, _u in ALL_UNIVERSES.items():
    NAME_LOOKUP.update(_u)
    for _tk in _u:
        TICKER_UNIVERSE_LABEL[_tk] = _uname

SEARCH_DIRECTORY = {
    f"{tk} \u2014 {name} ({TICKER_UNIVERSE_LABEL[tk]})": tk
    for tk, name in sorted(NAME_LOOKUP.items(), key=lambda kv: kv[1])
}

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Configuration")

st.sidebar.subheader("\U0001F50D Search & select assets")
st.sidebar.caption(
    "Start typing a **company name** (e.g. \"reliance\", \"apple\") or a **ticker** "
    "(e.g. \"AAPL\", \"TCS.NS\") \u2014 matching results filter live as you type."
)
search_selected_labels = st.sidebar.multiselect(
    "Search assets by name or ticker",
    options=list(SEARCH_DIRECTORY.keys()),
    default=[],
    placeholder="Type to search, e.g. 'reliance', 'AAPL', 'nifty bank'..."
)
searched_tickers = [SEARCH_DIRECTORY[lbl] for lbl in search_selected_labels]

st.sidebar.markdown("**\u2014 or \u2014**")

bucket_choices = st.sidebar.multiselect(
    "Pull an entire universe (adds all its tickers)",
    list(ALL_UNIVERSES.keys()),
    default=[]
)
bucket_tickers = []
for b in bucket_choices:
    bucket_tickers.extend(ALL_UNIVERSES[b].keys())

use_custom = st.sidebar.checkbox("Add a custom ticker not in the list above")
custom_tickers = []
if use_custom:
    custom_input = st.sidebar.text_input(
        "Custom ticker (.NS suffix for NSE stocks, e.g. INFY.NS)", ""
    )
    if custom_input.strip():
        custom_tickers = [custom_input.strip().upper()]

tickers = list(dict.fromkeys(searched_tickers + bucket_tickers + custom_tickers))

n_selected = len(tickers)
if n_selected > MAX_ASSETS:
    st.sidebar.error(
        f"You selected {n_selected} assets, but only {MAX_ASSETS} can be optimized per run. "
        f"The first {MAX_ASSETS} (in selection order) will be used \u2014 narrow your selection for full control."
    )
    tickers = tickers[:MAX_ASSETS]

st.sidebar.caption(f"**{len(tickers)} / {MAX_ASSETS} assets selected for this run**")
if tickers:
    with st.sidebar.expander("View selected tickers"):
        for tk in tickers:
            st.write(f"{tk} \u2014 {NAME_LOOKUP.get(tk, 'Custom ticker')}")

st.sidebar.subheader("Historical window")
st.sidebar.caption(
    "The app always fetches each ticker's **full available history** and caches it for 24h. "
    "\u26A0\uFE0F The usable analysis window is capped by whichever selected ticker has the "
    "SHORTEST history. A full table of every ticker's history length is shown after you run."
)

exclude_short_history = st.sidebar.checkbox(
    "Auto-exclude tickers with unusually short history (avoid shrinking the whole window)",
    value=False
)
if exclude_short_history:
    min_history_years = st.sidebar.number_input(
        "Minimum years of history required to keep a ticker in this run",
        min_value=0.1, max_value=20.0, value=2.0, step=0.5
    )
else:
    min_history_years = 0.0

use_custom_asof = st.sidebar.checkbox(
    "Use a custom 'as of' end date (point-in-time analysis)", value=False
)
if use_custom_asof:
    asof_date = st.sidebar.date_input(
        "As-of date (analysis uses only data up to and including this date)",
        value=pd.Timestamp.today().date(),
        max_value=pd.Timestamp.today().date()
    )
else:
    asof_date = None

analysis_years = st.sidebar.number_input(
    "Years of history to use, counting back from the as-of date (0 = use everything available)",
    min_value=0.0, max_value=50.0, value=0.0, step=0.5
)

n_portfolios = st.sidebar.number_input("Monte Carlo simulations", 1000, 200000, 50000, step=1000)
risk_free_rate = st.sidebar.number_input("Risk-free rate (annual, decimal)", 0.0, 0.20, 0.065, step=0.005)
allow_short = st.sidebar.checkbox("Allow short selling (weights can be negative)", value=False)
max_weight = st.sidebar.slider("Max weight per asset (concentration cap)", 0.05, 1.0, 1.0, step=0.05)

st.sidebar.subheader("\U0001F9EA Backtesting (out-of-sample)")
st.sidebar.caption(
    "Splits your analysis window into a training period and a held-out test period, to check "
    "whether the optimization is genuinely predictive or just fit to past data."
)
run_backtest = st.sidebar.checkbox("Run out-of-sample backtest", value=False)
if run_backtest:
    test_fraction = st.sidebar.slider(
        "Test period size (% of analysis window held out at the end)",
        min_value=10, max_value=50, value=25, step=5
    ) / 100.0
else:
    test_fraction = 0.25

run_btn = st.sidebar.button("Run Optimization", type="primary")

# ---------------- Data fetching (full history, robust to Yahoo rate limits) ----------------
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

def _download_chunk_max(chunk):
    for attempt in range(MAX_RETRIES):
        try:
            data = yf.download(chunk, period="max", auto_adjust=True,
                                progress=False, group_by="ticker", threads=True)
            if data is not None and not data.empty:
                return data
        except Exception:
            pass
        time.sleep(BASE_BACKOFF * (2 ** attempt))
    return None

def _download_single_max(tk):
    for attempt in range(MAX_RETRIES):
        try:
            hist = yf.Ticker(tk).history(period="max", auto_adjust=True)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                s = hist["Close"].copy()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                return s
        except Exception:
            pass
        time.sleep(BASE_BACKOFF * (2 ** attempt))
    return None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_full_history(tickers):
    series_map = {}
    remaining = list(tickers)

    for i in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[i:i + CHUNK_SIZE]
        data = _download_chunk_max(chunk)
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
        s = _download_single_max(tk)
        if s is not None and not s.dropna().empty:
            series_map[tk] = s
        time.sleep(0.5)

    if not series_map:
        return pd.DataFrame()

    prices = pd.DataFrame(series_map)
    prices = prices.dropna(axis=1, how="all")
    prices = prices.sort_index().ffill()
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

def compute_drawdown(cum_returns):
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def compute_sortino(daily_returns, rf_daily):
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
    ann_excess = excess.mean() * 252
    return ann_excess / downside_std if downside_std and downside_std > 0 else np.nan

def explain_frontier_plain_language(mean_rets, cov, w_max_sharpe, ret_ms, vol_ms, sharpe_ms,
                                      w_min_vol, ret_mv, vol_mv, assets, name_lookup, n, rf):
    single_vols = np.sqrt(np.diag(cov))
    avg_single_vol = single_vols.mean()
    risk_reduction = (avg_single_vol - vol_mv) / avg_single_vol if avg_single_vol > 0 else 0

    top3_ms_idx = np.argsort(w_max_sharpe)[::-1][:3]
    top3_names = [name_lookup.get(assets[i], assets[i]) for i in top3_ms_idx if w_max_sharpe[i] > 0.01]

    return_gain = ret_ms - ret_mv
    vol_gain = vol_ms - vol_mv

    lines = []
    lines.append(
        f"**In plain terms:** out of the {n} investments you picked, spreading your money across all of them "
        f"instead of holding just one cuts your risk by roughly **{risk_reduction:.0%}** on average \u2014 "
        "this is the core benefit of diversification: it smooths out the bumps without necessarily costing you return."
    )
    if top3_names:
        lines.append(
            f"The single **best risk-adjusted mix** (gold star) leans most heavily on "
            f"**{', '.join(top3_names)}**. It's expected to grow about **{ret_ms:.1%} per year**, "
            f"with typical up-and-down swings of about **\u00b1{vol_ms:.1%}** \u2014 think of that as the "
            "size of a normal good or bad year, not a hard ceiling or floor."
        )
    lines.append(
        f"The **safest mix** (blue diamond) trades some of that return away \u2014 about "
        f"**{return_gain:.1%} per year less** \u2014 in exchange for noticeably calmer swings "
        f"(**{abs(vol_gain):.1%}** less bounce). Neither mix is \"right\"; it depends on whether "
        "you'd rather chase higher growth or sleep easier during downturns."
    )
    lines.append(
        f"The red curve is the boundary of what's mathematically achievable with this exact set of assets \u2014 "
        "every dot below or to the right of it represents a worse deal. Nothing on this chart guarantees future "
        "performance; it's built entirely from how these assets moved in the past."
    )
    return "\n\n".join(lines)

def explain_backtest_plain_language(bt_results, best_strategy, beat_benchmark, test_days):
    ms_row = bt_results.iloc[0]
    eq_row = bt_results.iloc[2]
    diff = ms_row["Test-Period Return (%)"] - eq_row["Test-Period Return (%)"]

    lines = []
    if beat_benchmark:
        lines.append(
            f"**In plain terms:** when the \"optimal\" mix was actually put to the test on "
            f"**{test_days} days of unseen data**, it earned about **{diff:+.1f} percentage points more** than "
            "simply splitting your money equally across every asset. That's a genuinely encouraging sign \u2014 "
            "it suggests the optimization found a real, lasting pattern in how these assets behave."
        )
    else:
        lines.append(
            f"**In plain terms:** when the \"optimal\" mix was actually put to the test on "
            f"**{test_days} days of unseen data**, it earned about **{abs(diff):.1f} percentage points LESS** than "
            "simply splitting your money equally across every asset. This is a common and important finding: "
            "a strategy that looks best on historical data doesn't always keep winning on data it never saw."
        )
    lines.append(
        f"**{best_strategy}** ended up being the best performer in this specific test window. Running the "
        "backtest with a different test period size or as-of date can change which strategy comes out ahead."
    )
    return "\n\n".join(lines)

# ---------------- Main pipeline ----------------
if run_btn:
    if len(tickers) < 2:
        st.error("Select at least 2 tickers using the search box or a universe.")
        st.stop()

    with st.spinner(f"Downloading FULL available price history for {len(tickers)} tickers (chunked, with retries)..."):
        full_prices = fetch_full_history(tuple(tickers))

    dropped = set(tickers) - set(full_prices.columns if not full_prices.empty else [])
    if dropped:
        st.warning(
            f"Could not fetch data for: {', '.join(sorted(dropped))} \u2014 excluded from optimization. "
            "Yahoo Finance sometimes rate-limits large batch requests (HTTP 429), or the ticker may have "
            "changed/delisted. Click **Run Optimization** again in 30\u201360 seconds, or reduce the selection."
        )

    if full_prices.empty or full_prices.shape[1] < 2:
        st.error(
            "Could not fetch enough price data. Wait about a minute and click **Run Optimization** again, "
            "try a smaller ticker selection, or reduce Monte Carlo simulations to retry faster."
        )
        st.stop()

    per_ticker_start = {tk: full_prices[tk].dropna().index.min() for tk in full_prices.columns}
    per_ticker_end = {tk: full_prices[tk].dropna().index.max() for tk in full_prices.columns}
    per_ticker_years = {tk: (per_ticker_end[tk] - start).days / 365.25 for tk, start in per_ticker_start.items()}

    if exclude_short_history and min_history_years > 0:
        short_tickers = [tk for tk, yrs in per_ticker_years.items() if yrs < min_history_years]
        if short_tickers:
            st.warning(
                f"Excluded {len(short_tickers)} ticker(s) with less than {min_history_years} years of history: "
                f"{', '.join(f'{tk} ({NAME_LOOKUP.get(tk, tk)}, {per_ticker_years[tk]:.1f}y)' for tk in short_tickers)}"
            )
            full_prices = full_prices.drop(columns=short_tickers)
            per_ticker_start = {tk: v for tk, v in per_ticker_start.items() if tk not in short_tickers}
            per_ticker_end = {tk: v for tk, v in per_ticker_end.items() if tk not in short_tickers}
            per_ticker_years = {tk: v for tk, v in per_ticker_years.items() if tk not in short_tickers}

    if full_prices.empty or full_prices.shape[1] < 2:
        st.error("Fewer than 2 tickers remain after excluding short-history assets. Adjust your filters.")
        st.stop()

    bottleneck_ticker = max(per_ticker_start, key=per_ticker_start.get)
    bottleneck_start = per_ticker_start[bottleneck_ticker]
    bottleneck_years = per_ticker_years[bottleneck_ticker]

    earliest = full_prices.index.min().date()
    latest = full_prices.index.max().date()
    total_years = (full_prices.index.max() - full_prices.index.min()).days / 365.25

    effective_end = pd.Timestamp(asof_date) if asof_date is not None else full_prices.index.max()
    effective_end = min(effective_end, full_prices.index.max())

    windowed = full_prices[full_prices.index <= effective_end]

    if windowed.empty:
        st.error(f"No data available on or before {effective_end.date()}. Earliest data starts {earliest}.")
        st.stop()

    if analysis_years and analysis_years > 0:
        # Use a day-count Timedelta rather than pd.DateOffset(years=...): DateOffset
        # requires an INTEGER year count internally (dateutil raises ValueError on
        # fractional years like 2.5), but this slider allows half-year steps.
        cutoff = effective_end - pd.Timedelta(days=int(round(analysis_years * 365.25)))
        prices = windowed[windowed.index >= cutoff].dropna(how="any")
        window_desc = f"{analysis_years} year(s) ending {effective_end.date()} (requested)"
    else:
        prices = windowed.dropna(how="any")
        window_desc = f"everything available up to {effective_end.date()}"

    actual_window_years = (prices.index.max() - prices.index.min()).days / 365.25 if len(prices) > 1 else 0

    history_table = pd.DataFrame({
        "Ticker": list(per_ticker_years.keys()),
        "Name": [NAME_LOOKUP.get(tk, tk) for tk in per_ticker_years.keys()],
        "History Start": [per_ticker_start[tk].date() for tk in per_ticker_years.keys()],
        "Years of History": [per_ticker_years[tk] for tk in per_ticker_years.keys()],
    }).sort_values("Years of History").reset_index(drop=True)

    requested_years_for_flag = analysis_years if (analysis_years and analysis_years > 0) else total_years
    history_table["Limits Requested Window?"] = history_table["Years of History"] < (requested_years_for_flag * 0.95)
    short_but_usable = history_table[
        (history_table["Years of History"] < requested_years_for_flag) &
        (history_table["Years of History"] >= min(1.0, requested_years_for_flag * 0.1))
    ]

    min_days_needed = 60 if run_backtest else 30
    if prices.shape[0] < min_days_needed:
        st.error(
            f"Only {prices.shape[0]} trading days available in this window \u2014 too few for reliable statistics"
            f"{' with backtesting enabled' if run_backtest else ''}. Pick a longer window, fewer years back, "
            "or remove short-history tickers from your selection."
        )
        st.stop()

    assets = list(prices.columns)
    n = len(assets)

    log_returns = np.log(prices / prices.shift(1)).dropna()
    mean_rets = (log_returns.mean() * 252).values
    cov = (log_returns.cov() * 252).values
    corr_full = log_returns.corr()
    corr = corr_full.values

    st.subheader("\U0001F4CB Data Synopsis")
    n_by_universe = {}
    for uname, umap in ALL_UNIVERSES.items():
        n_by_universe[uname] = sum(1 for a in assets if a in umap)

    best_ret_idx = int(np.argmax(mean_rets))
    worst_ret_idx = int(np.argmin(mean_rets))
    best_vol_idx = int(np.argmin(np.sqrt(np.diag(cov))))
    worst_vol_idx = int(np.argmax(np.sqrt(np.diag(cov))))
    corr_no_diag = corr_full.where(~np.eye(len(corr_full), dtype=bool))
    max_pair = corr_no_diag.stack().idxmax()
    min_pair = corr_no_diag.stack().idxmin()

    syn_col1, syn_col2 = st.columns(2)
    with syn_col1:
        st.markdown(f"""
- **Assets used:** {n} (requested {len(tickers)}, {len(dropped)} unavailable)
- **Universe mix:** {", ".join(f"{v} {k}" for k, v in n_by_universe.items() if v > 0)}
- **Shortest-history ticker (sets the window cap):** {NAME_LOOKUP.get(bottleneck_ticker, bottleneck_ticker)} ({bottleneck_ticker}) \u2014 {bottleneck_years:.1f} years
- **Full cached history (all assets combined):** {earliest} \u2192 {latest} (~{total_years:.1f} years)
- **Analysis window setting:** {window_desc}
- **Actual data range used:** {prices.index.min().date()} \u2192 {prices.index.max().date()} ({len(prices)} trading days, ~{actual_window_years:.1f} years)
        """)
    with syn_col2:
        st.markdown(f"""
- **Highest annualized return:** {NAME_LOOKUP.get(assets[best_ret_idx], assets[best_ret_idx])} ({mean_rets[best_ret_idx]:.1%})
- **Lowest annualized return:** {NAME_LOOKUP.get(assets[worst_ret_idx], assets[worst_ret_idx])} ({mean_rets[worst_ret_idx]:.1%})
- **Most volatile:** {NAME_LOOKUP.get(assets[worst_vol_idx], assets[worst_vol_idx])} ({np.sqrt(np.diag(cov))[worst_vol_idx]:.1%} ann. vol)
- **Most stable:** {NAME_LOOKUP.get(assets[best_vol_idx], assets[best_vol_idx])} ({np.sqrt(np.diag(cov))[best_vol_idx]:.1%} ann. vol)
- **Most correlated pair:** {NAME_LOOKUP.get(max_pair[0], max_pair[0])} & {NAME_LOOKUP.get(max_pair[1], max_pair[1])} ({corr_no_diag.loc[max_pair]:.2f})
- **Most diversifying pair:** {NAME_LOOKUP.get(min_pair[0], min_pair[0])} & {NAME_LOOKUP.get(min_pair[1], min_pair[1])} ({corr_no_diag.loc[min_pair]:.2f})
        """)

    if not short_but_usable.empty:
        st.markdown("**\u23F1\uFE0F Assets with shorter history than requested (still usable, but limiting the window):**")
        st.dataframe(
            short_but_usable[["Ticker", "Name", "History Start", "Years of History"]]
                .style.format({"Years of History": "{:.1f} yrs"}),
            hide_index=True
        )
        st.caption(
            "These tickers have less history than your requested window, so the FINAL analysis window is capped "
            "by the shortest one among them."
        )

    with st.expander("\U0001F4CA Full history-length table for ALL selected tickers"):
        st.dataframe(
            history_table.style.format({"Years of History": "{:.1f} yrs"}),
            hide_index=True
        )
        st.caption("Sorted shortest history first. 'Limits Requested Window?' flags tickers shorter than your request.")

    st.caption("The full history stays cached for 24h regardless of which window is analyzed here.")

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
        z=corr, x=assets, y=assets, colorscale="RdBu", zmid=0, colorbar=dict(title="Corr")
    ))
    fig_corr.update_layout(height=min(22 * n + 100, 900), template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Full annualized covariance is used internally; this heatmap shows pairwise correlation for readability.")

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
        marker=dict(size=3, color=sharpe_mc, colorscale="Viridis", colorbar=dict(title="Sharpe"), showscale=True),
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
        xaxis_title="Annualized Volatility (Risk)", yaxis_title="Annualized Expected Return",
        template="plotly_white", height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Gold star = tangency portfolio (max Sharpe). Blue diamond = global minimum-variance portfolio. "
        "Red curve = efficient frontier via SLSQP. "
        f"Universe: {n} assets, analysis window: {window_desc}. Mixed currencies (USD/INR) are not FX-adjusted."
    )

    st.markdown("#### \U0001F4A1 What This Chart Means")
    st.markdown(explain_frontier_plain_language(
        mean_rets, cov, w_max_sharpe, ret_ms, vol_ms, sharpe_ms,
        w_min_vol, ret_mv, vol_mv, assets, NAME_LOOKUP, n, risk_free_rate
    ))

    # ---------------- 5. Out-of-sample backtest ----------------
    if run_backtest:
        st.subheader("5. \U0001F9EA Out-of-Sample Backtest")

        split_idx = int(len(prices) * (1 - test_fraction))
        train_prices = prices.iloc[:split_idx]
        test_prices = prices.iloc[split_idx:]

        if len(train_prices) < 30 or len(test_prices) < 10:
            st.warning(
                "Not enough data to split into a meaningful train/test period at this window size. "
                "Try a longer analysis window or a smaller test period percentage."
            )
        else:
            train_log_rets = np.log(train_prices / train_prices.shift(1)).dropna()
            train_mean = (train_log_rets.mean() * 252).values
            train_cov = (train_log_rets.cov() * 252).values

            with st.spinner("Optimizing on training period and simulating out-of-sample performance..."):
                w_train_sharpe = optimize_max_sharpe(train_mean, train_cov, risk_free_rate, bounds)
                w_train_minvol = optimize_min_vol(train_mean, train_cov, bounds).x
                w_equal = np.repeat(1 / n, n)

                test_log_rets = np.log(test_prices / test_prices.shift(1)).dropna()

                def portfolio_cum_return(weights, daily_rets):
                    port_daily = daily_rets @ weights
                    cum = (1 + port_daily).cumprod()
                    return port_daily, cum

                pd_ms, cum_ms = portfolio_cum_return(w_train_sharpe, test_log_rets)
                pd_mv, cum_mv = portfolio_cum_return(w_train_minvol, test_log_rets)
                pd_eq, cum_eq = portfolio_cum_return(w_equal, test_log_rets)

                rf_daily = risk_free_rate / 252

                bt_results = pd.DataFrame({
                    "Strategy": ["Max Sharpe (trained)", "Min Volatility (trained)", "Equal-Weight Benchmark"],
                    "Test-Period Return (%)": [
                        (cum_ms.iloc[-1] - 1) * 100, (cum_mv.iloc[-1] - 1) * 100, (cum_eq.iloc[-1] - 1) * 100
                    ],
                    "Test-Period Ann. Vol (%)": [
                        pd_ms.std() * np.sqrt(252) * 100, pd_mv.std() * np.sqrt(252) * 100, pd_eq.std() * np.sqrt(252) * 100
                    ],
                    "Max Drawdown (%)": [
                        compute_drawdown(cum_ms) * 100, compute_drawdown(cum_mv) * 100, compute_drawdown(cum_eq) * 100
                    ],
                    "Sortino Ratio": [
                        compute_sortino(pd_ms, rf_daily), compute_sortino(pd_mv, rf_daily), compute_sortino(pd_eq, rf_daily)
                    ],
                })

            st.markdown(
                f"**Training period:** {train_prices.index.min().date()} \u2192 {train_prices.index.max().date()} "
                f"({len(train_prices)} days) \u2014 weights optimized here, then FROZEN.\n\n"
                f"**Test period (out-of-sample):** {test_prices.index.min().date()} \u2192 {test_prices.index.max().date()} "
                f"({len(test_prices)} days) \u2014 frozen weights applied here, unseen during optimization."
            )

            st.markdown("**\U0001F4B0 Return earned by each strategy during the test period:**")
            styled_bt = bt_results.style.format({
                "Test-Period Return (%)": "{:+.2f}%", "Test-Period Ann. Vol (%)": "{:.2f}%",
                "Max Drawdown (%)": "{:.2f}%", "Sortino Ratio": "{:.3f}"
            })
            try:
                styled_bt = styled_bt.background_gradient(subset=["Test-Period Return (%)"], cmap="RdYlGn")
            except (ImportError, ModuleNotFoundError):
                pass
            st.dataframe(styled_bt, hide_index=True)

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=cum_ms.index, y=(cum_ms - 1) * 100, name="Max Sharpe (trained)",
                                          line=dict(color="gold", width=2.5)))
            fig_bt.add_trace(go.Scatter(x=cum_mv.index, y=(cum_mv - 1) * 100, name="Min Volatility (trained)",
                                          line=dict(color="blue", width=2.5)))
            fig_bt.add_trace(go.Scatter(x=cum_eq.index, y=(cum_eq - 1) * 100, name="Equal-Weight Benchmark",
                                          line=dict(color="gray", width=2, dash="dash")))
            fig_bt.update_layout(
                title="Out-of-Sample Cumulative Return (%)",
                xaxis_title="Date", yaxis_title="Cumulative Return (%)",
                template="plotly_white", height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            best_strategy = bt_results.loc[bt_results["Test-Period Return (%)"].idxmax(), "Strategy"]
            beat_benchmark = bt_results.iloc[0]["Test-Period Return (%)"] > bt_results.iloc[2]["Test-Period Return (%)"]
            st.caption(
                f"**{best_strategy}** had the highest out-of-sample return in this test period. "
                f"The Max Sharpe portfolio {'beat' if beat_benchmark else 'did NOT beat'} the equal-weight "
                "benchmark here."
            )

            st.markdown("#### \U0001F4A1 What This Chart Means")
            st.markdown(explain_backtest_plain_language(bt_results, best_strategy, beat_benchmark, len(test_prices)))
else:
    st.info("Search for assets by name/ticker or pull a whole universe in the sidebar (max 100 per run), then click **Run Optimization**.")
    st.markdown(
        "**\U0001F50D How search works:** type into the sidebar search box \u2014 it matches company names and "
        "ticker symbols across all 300 available assets.\n\n"
        "\U0001F4C5 **Full-history data policy:** every ticker's entire available price history is fetched and "
        "cached for 24 hours.\n\n"
        "\u26A0\uFE0F **Note:** your analysis window is capped by whichever selected ticker has the "
        "shortest available history. A full history-length table for every selected ticker is shown after "
        "you run.\n\n"
        "\U0001F553 **Custom analysis window:** run the optimization as if today were an earlier date.\n\n"
        "\U0001F9EA **Backtesting:** see how the optimized weights would have actually performed on unseen "
        "data, with the exact % return earned by each strategy shown in a table.\n\n"
        "\U0001F4A1 **Plain-language takeaways:** after each chart, a jargon-free summary of what it means."
    )
