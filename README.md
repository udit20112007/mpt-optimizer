# 📈 Modern Portfolio Theory (MPT) Optimizer

An interactive web app that applies **Harry Markowitz's Modern Portfolio Theory** to real market data, combining linear algebra, Monte Carlo simulation, and constrained numerical optimization to map and solve the efficient frontier for any set of assets.

**[Live demo instructions below](#deployment-streamlit-community-cloud-free)** — deploy in under 5 minutes.

## What This Project Demonstrates

This project was built to showcase three distinct technical skills in a single applied pipeline:

- **Linear algebra**: constructing an annualized covariance matrix from historical log returns and using matrix/quadratic-form multiplication (`w^T Σ w`) to compute portfolio-level variance for thousands of weight combinations at once.
- **Stochastic simulation**: a vectorized Monte Carlo engine that generates 50,000 random, valid (non-negative, sum-to-one) portfolio weight vectors using a Dirichlet distribution, then evaluates the return/risk/Sharpe ratio for all of them in a single NumPy operation.
- **Constrained optimization**: `scipy.optimize.minimize` with the **SLSQP** (Sequential Least Squares Programming) method, used to solve three distinct nonlinear optimization problems — maximizing the Sharpe ratio, minimizing volatility, and tracing the exact efficient frontier curve by minimizing volatility at 60 different target return levels.

## How It Works

### 1. Data ingestion
The app pulls historical daily close prices for any list of tickers (5 recommended) via the `yfinance` API, over a user-selected lookback window (1–10 years).

### 2. Statistics layer
Daily log returns are computed as `ln(P_t / P_{t-1})`, then annualized:
- **Annualized return** = mean(daily log return) × 252
- **Annualized covariance matrix** = cov(daily log returns) × 252

Log returns are used instead of simple returns because they are time-additive and better satisfy the normality assumptions behind mean-variance optimization.

### 3. Monte Carlo simulation (50,000 portfolios)
Random weight vectors are drawn from a Dirichlet distribution (guaranteeing weights are non-negative and sum to 1 for a long-only portfolio, or optionally allow short-selling via a toggle). Portfolio return, volatility, and Sharpe ratio are computed for all 50,000 combinations simultaneously using `np.einsum` for the batched quadratic form, mapping the feasible risk/return cloud.

### 4. SLSQP optimization
Three separate constrained optimization problems are solved with `scipy.optimize.minimize(method="SLSQP")`:

| Optimization | Objective | Constraints |
|---|---|---|
| Max Sharpe Ratio | Minimize `-(R_p - R_f) / σ_p` | weights sum to 1, 0 ≤ w ≤ 1 |
| Min Volatility | Minimize `σ_p = sqrt(w^T Σ w)` | weights sum to 1, 0 ≤ w ≤ 1 |
| Efficient Frontier | Minimize `σ_p` for each of 60 target returns | weights sum to 1, R_p = target, 0 ≤ w ≤ 1 |

The Max Sharpe portfolio is the tangency point where a line from the risk-free rate is tangent to the frontier — the theoretically optimal risk-adjusted portfolio. The Min Volatility portfolio is the leftmost point of the frontier, representing the lowest-risk diversified combination available.

### 5. Visualization
An interactive Plotly chart renders:
- The Monte Carlo cloud of 50,000 simulated portfolios, colored by Sharpe ratio
- The red efficient frontier curve traced by the SLSQP optimizer
- A gold star marking the Max Sharpe portfolio
- A blue diamond marking the Min Volatility portfolio

## Features

- Configurable ticker list (works with any number of assets, 5 recommended for the classic MPT setup)
- Adjustable lookback period, risk-free rate, and Monte Carlo sample size (up to 200,000)
- Optional short-selling mode (allows negative weights)
- Cached data fetching (1-hour TTL) to avoid redundant API calls
- Fully interactive chart (hover, zoom, pan) rather than a static image

## Tech Stack

- **Python** — core language
- **NumPy / Pandas** — linear algebra, returns, and covariance computation
- **SciPy** (`optimize.minimize`, SLSQP) — constrained nonlinear optimization
- **yfinance** — live historical price data
- **Plotly** — interactive charting
- **Streamlit** — web app framework and UI

## Running Locally

```bash
pip install -r requirements.txt
streamlit run mpt_optimizer.py
```

## Deployment (Streamlit Community Cloud, free)

1. Fork or clone this repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository and branch, and set `mpt_optimizer.py` as the main file.
4. Click **Deploy**. Streamlit builds the app and hosts it at a public URL (e.g. `https://<app-name>.streamlit.app`) for free.

## Possible Extensions

- Replace naive historical mean returns with **CAPM-implied** or **shrinkage-adjusted** expected returns (e.g. Ledoit-Wolf covariance shrinkage) to reduce estimation error.
- Add a **backtesting module** to compare the optimized portfolio's realized performance against a benchmark (e.g. SPY) out-of-sample.
- Add **sector/asset-class constraints** (e.g. max 40% in any single sector) to the SLSQP constraint set.
- Support **rolling-window** re-optimization to visualize how the efficient frontier and optimal weights drift over time.

## License

MIT
