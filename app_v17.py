import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Equity Research Platform V17",
    layout="wide"
)

st.markdown("""
<style>
    .stMetric { background: #0f1117; border: 1px solid #1e2130; border-radius: 8px; padding: 12px; }
    .risk-badge { display:inline-block; padding:4px 10px; border-radius:20px; font-size:13px; font-weight:600; }
    .risk-low { background:#0d3b1e; color:#2ecc71; }
    .risk-med { background:#3b2a0d; color:#f39c12; }
    .risk-high { background:#3b0d0d; color:#e74c3c; }
    .canslim-bar { height:8px; border-radius:4px; background:#1e2130; margin-top:4px; }
    .canslim-fill { height:8px; border-radius:4px; background:linear-gradient(90deg,#1a9e5c,#27ae60); }
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Equity Research Platform V17")
st.caption("Score-driven probabilities · CANSLIM scoring · Smart research notes · Risk management layer · Just type RELIANCE, TCS — no .NS needed")

# =========================================================
# DATA FETCH
# =========================================================

def normalize_ticker(ticker):
    """
    Let users type RELIANCE, TCS, INFY etc.
    Auto-appends .NS for NSE if no exchange suffix present.
    Handles: RELIANCE -> RELIANCE.NS
             RELIANCE.NS -> RELIANCE.NS (unchanged)
             RELIANCE.BO -> RELIANCE.BO (unchanged)
    """
    ticker = ticker.strip().upper()
    if "." not in ticker:
        ticker = ticker + ".NS"
    return ticker


@st.cache_data(ttl=3600)
def fetch_data(ticker, period="5y"):
    ticker = normalize_ticker(ticker)
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.strip().title() for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Close"}, inplace=True)
    needed = [c for c in ["Close", "Open", "High", "Low", "Volume"] if c in df.columns]
    if not needed or "Close" not in needed:
        return pd.DataFrame()
    df = df[needed].copy()
    df.dropna(subset=["Close"], inplace=True)
    return df


# =========================================================
# CANSLIM SCORING ENGINE
# =========================================================

def compute_canslim_score(df):
    close = df["Close"]
    scores = {}

    if len(close) >= 63:
        c_return = (close.iloc[-1] / close.iloc[-63] - 1) * 100
        scores["C_CurrentMomentum"] = min(max(c_return / 20 * 20, 0), 20)
    else:
        scores["C_CurrentMomentum"] = 0

    sma200 = close.rolling(200).mean()
    if len(sma200.dropna()) > 0:
        gap_pct = (close.iloc[-1] / sma200.iloc[-1] - 1) * 100
        scores["A_AnnualTrend"] = min(max(gap_pct / 10 * 15, 0), 15)
    else:
        scores["A_AnnualTrend"] = 0

    if len(close) >= 252:
        high_52w = close.rolling(252).max().iloc[-1]
        proximity = (close.iloc[-1] / high_52w) * 100
        scores["N_NewHighs"] = 15 if proximity >= 95 else (8 if proximity >= 85 else 0)
    else:
        scores["N_NewHighs"] = 0

    if "Volume" in df.columns and len(df) >= 50:
        vol = df["Volume"]
        avg_vol_20 = vol.rolling(20).mean().iloc[-1]
        avg_vol_50 = vol.rolling(50).mean().iloc[-1]
        scores["S_SupplyDemand"] = 15 if avg_vol_20 > avg_vol_50 * 1.1 else (8 if avg_vol_20 > avg_vol_50 else 3)
    else:
        scores["S_SupplyDemand"] = 7

    sma50 = close.rolling(50).mean()
    if len(sma50.dropna()) > 0 and len(sma200.dropna()) > 0:
        scores["L_Leader"] = 15 if sma50.iloc[-1] > sma200.iloc[-1] else 0
    else:
        scores["L_Leader"] = 0

    if len(sma200.dropna()) >= 20:
        sma200_slope = (sma200.iloc[-1] / sma200.iloc[-20] - 1) * 100
        above_200 = close.iloc[-1] > sma200.iloc[-1]
        scores["I_Institutional"] = 10 if (above_200 and sma200_slope > 0) else (5 if above_200 else 0)
    else:
        scores["I_Institutional"] = 0

    scores["M_MarketDirection"] = 10
    return scores, sum(scores.values())


# =========================================================
# PROBABILITY ENGINE
# =========================================================

def compute_probability_scenarios(score, canslim_total, rsi, momentum, relative_strength):
    rsi_signal = max(0, min(100, (rsi - 30) / 40 * 100)) if rsi else 50
    mom_signal = 70 if momentum > 5 else (50 if momentum > 0 else 30)
    rs_signal = 70 if relative_strength > 3 else (50 if relative_strength > 0 else 30)
    composite = (score * 0.30 + min(canslim_total, 100) * 0.30 +
                 rsi_signal * 0.15 + mom_signal * 0.15 + rs_signal * 0.10)
    bullish = round(max(10, min(75, composite * 0.70)), 1)
    bearish = round(max(5, min(60, (100 - composite) * 0.55)), 1)
    sideways = round(100 - bullish - bearish, 1)
    if sideways < 5:
        sideways = 5
        bearish = round(100 - bullish - sideways, 1)
    return {"Bullish Continuation": bullish, "Sideways Consolidation": sideways, "Bearish Breakdown": bearish}


# =========================================================
# RISK MANAGEMENT ENGINE
# =========================================================

def compute_risk_metrics(df, capital=100000, risk_per_trade_pct=1.5):
    close = df["Close"]
    if "High" in df.columns and "Low" in df.columns:
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - close.shift()).abs(),
            (df["Low"] - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
    else:
        atr = close.pct_change().std() * close.iloc[-1] * np.sqrt(14)

    current_price = float(close.iloc[-1])
    stop_loss = current_price - (2 * atr)
    stop_loss_pct = ((current_price - stop_loss) / current_price) * 100
    risk_amount = capital * (risk_per_trade_pct / 100)
    risk_per_share = current_price - stop_loss
    shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
    position_value = shares * current_price
    vol_annual = close.pct_change().std() * np.sqrt(252) * 100
    rolling_max = close.cummax()
    drawdown = (close / rolling_max - 1) * 100

    return {
        "ATR": round(float(atr), 2),
        "CurrentPrice": round(current_price, 2),
        "StopLoss": round(float(stop_loss), 2),
        "StopLossPct": round(float(stop_loss_pct), 2),
        "Target1R": round(float(current_price + 2 * atr), 2),
        "Target2R": round(float(current_price + 4 * atr), 2),
        "Target3R": round(float(current_price + 6 * atr), 2),
        "RiskAmount": round(risk_amount, 0),
        "Shares": shares,
        "PositionValue": round(position_value, 0),
        "PositionPct": round((position_value / capital) * 100, 1),
        "VolatilityAnnual": round(vol_annual, 1),
        "RiskTier": "Low" if vol_annual < 20 else ("Medium" if vol_annual < 35 else "High"),
        "MaxDrawdown": round(float(drawdown.min()), 2),
        "CurrentDrawdown": round(float(drawdown.iloc[-1]), 2)
    }


# =========================================================
# CORE METRICS ENGINE
# =========================================================

def compute_metrics(df, benchmark_df):
    close = df["Close"].copy()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    delta = close.diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + rs))
    momentum = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100
    bench_return = ((benchmark_df["Close"].iloc[-1] / benchmark_df["Close"].iloc[-63]) - 1) * 100
    stock_return = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100
    relative_strength = stock_return - bench_return
    volatility = close.pct_change().std() * np.sqrt(252) * 100

    score = 0
    if float(close.iloc[-1]) > float(sma200.iloc[-1]): score += 30
    if float(sma50.iloc[-1]) > float(sma200.iloc[-1]): score += 25
    if float(rsi.iloc[-1]) > 55: score += 20
    if momentum > 0: score += 15
    if relative_strength > 0: score += 10

    rec = "🟢 Strong Buy" if score >= 80 else ("🟢 Buy" if score >= 60 else ("🟠 Watch" if score >= 40 else "🔴 Avoid"))
    structure = (
        "Bullish Structure" if float(close.iloc[-1]) > float(sma200.iloc[-1]) and float(sma50.iloc[-1]) > float(sma200.iloc[-1])
        else "Early Accumulation" if float(close.iloc[-1]) > float(sma200.iloc[-1])
        else "Bearish Structure"
    )

    return {
        "Price": round(float(close.iloc[-1]), 2),
        "RSI": round(float(rsi.iloc[-1]), 2),
        "Momentum": round(float(momentum), 2),
        "RelativeStrength": round(float(relative_strength), 2),
        "Volatility": round(float(volatility), 2),
        "SMA50": round(float(sma50.iloc[-1]), 2),
        "SMA200": round(float(sma200.iloc[-1]), 2),
        "Recommendation": rec,
        "Structure": structure,
        "Score": int(score)
    }


# =========================================================
# RULE-BASED RESEARCH NOTE (zero cost, no API)
# =========================================================

def generate_research_note(ticker, metrics, canslim_scores, canslim_total, risk_metrics, scenarios):
    score = metrics["Score"]
    rsi = metrics["RSI"]
    momentum = metrics["Momentum"]
    rs = metrics["RelativeStrength"]
    bull_prob = scenarios["Bullish Continuation"]
    bear_prob = scenarios["Bearish Breakdown"]

    # 1. Investment Thesis
    if score >= 80:
        thesis = (f"{ticker} presents a high-conviction setup scoring {score}/100. "
                  f"The {metrics['Structure'].lower()} is confirmed across trend, momentum, and relative strength. "
                  f"Risk/reward favours bulls at {bull_prob:.0f}% bullish probability.")
    elif score >= 60:
        thesis = (f"{ticker} shows a constructive setup scoring {score}/100 with some mixed signals. "
                  f"The {metrics['Structure'].lower()} is intact but selective entry with strict risk management is warranted. "
                  f"Bullish probability stands at {bull_prob:.0f}%.")
    elif score >= 40:
        thesis = (f"{ticker} is in a neutral zone at {score}/100 — a decision point. "
                  f"Neither a clear buy nor a clear avoid. Wait for a cleaner signal before committing capital.")
    else:
        thesis = (f"{ticker} is in a weak technical setup scoring {score}/100. "
                  f"The {metrics['Structure'].lower()} suggests selling pressure dominates with {bear_prob:.0f}% bearish probability. "
                  f"Capital is better deployed elsewhere for now.")

    # 2. Technical Setup
    sma_line = (
        "Price is above both SMA50 and SMA200 — classic golden cross, a primary institutional accumulation signal."
        if metrics["Price"] > metrics["SMA50"] > metrics["SMA200"]
        else "Price is above SMA200 but SMA50 is still below it — early recovery, not yet fully confirmed."
        if metrics["Price"] > metrics["SMA200"]
        else "Price is below SMA200 — primary trend is bearish, caution warranted."
    )
    rsi_line = (
        f"RSI at {rsi:.0f} is overbought — momentum is strong but a pullback is possible before the next leg."
        if rsi > 70 else
        f"RSI at {rsi:.0f} is in healthy bullish territory, supporting continuation."
        if rsi > 55 else
        f"RSI at {rsi:.0f} is neutral — momentum has not yet recovered sufficiently."
        if rsi > 40 else
        f"RSI at {rsi:.0f} is oversold — potential bounce candidate but confirm before entering."
    )
    mom_line = (
        f"3-month momentum is a strong +{momentum:.1f}% — sustained buying interest."
        if momentum > 5 else
        f"3-month momentum is mildly positive at {momentum:.1f}% — trending right but lacking conviction."
        if momentum > 0 else
        f"3-month momentum is negative at {momentum:.1f}% — near-term selling pressure persists."
    )
    rs_line = (
        f"Outperforming Nifty by {rs:.1f}% over 3 months — hallmark of a market leader."
        if rs > 5 else
        f"Mild Nifty outperformance of {rs:.1f}% — holding its own but not yet leading."
        if rs > 0 else
        f"Underperforming Nifty by {abs(rs):.1f}% — relative weakness is a concern."
    )

    # 3. CANSLIM Assessment
    strengths, weaknesses = [], []
    if canslim_scores.get("C_CurrentMomentum", 0) >= 10: strengths.append("strong current quarterly momentum (C)")
    else: weaknesses.append("weak current quarterly momentum (C)")
    if canslim_scores.get("N_NewHighs", 0) >= 15: strengths.append("near 52-week highs — price discovery (N)")
    elif canslim_scores.get("N_NewHighs", 0) == 0: weaknesses.append("far from 52-week highs (N)")
    if canslim_scores.get("S_SupplyDemand", 0) >= 15: strengths.append("rising volume confirming price strength (S)")
    else: weaknesses.append("volume not yet confirming price moves (S)")
    if canslim_scores.get("L_Leader", 0) >= 15: strengths.append("golden cross structure — market leader (L)")
    else: weaknesses.append("lagging moving average structure (L)")
    if canslim_scores.get("I_Institutional", 0) >= 10: strengths.append("rising 200DMA — institutional support (I)")

    canslim_text = (
        f"CANSLIM score: {int(canslim_total)}/100. "
        + (f"Strengths: {', '.join(strengths)}. " if strengths else "No strong CANSLIM signals present. ")
        + (f"Gaps: {', '.join(weaknesses)}." if weaknesses else "No major CANSLIM weaknesses identified.")
    )

    # 4. Risk Assessment
    vol_line = (
        f"Annual volatility of {risk_metrics['VolatilityAnnual']:.0f}% — high risk tier, position sizing must be conservative."
        if risk_metrics["RiskTier"] == "High" else
        f"Annual volatility of {risk_metrics['VolatilityAnnual']:.0f}% — moderate, standard position sizing applies."
        if risk_metrics["RiskTier"] == "Medium" else
        f"Annual volatility of {risk_metrics['VolatilityAnnual']:.0f}% — low, suitable for larger allocation."
    )
    dd_line = (
        f"Historical max drawdown of {risk_metrics['MaxDrawdown']:.0f}% — deep correction risk, stop discipline is non-negotiable."
        if risk_metrics["MaxDrawdown"] < -40 else
        f"Historical max drawdown of {risk_metrics['MaxDrawdown']:.0f}% — within acceptable range for swing/positional trades."
    )
    sizing_line = (
        f"ATR stop at ₹{risk_metrics['StopLoss']} ({risk_metrics['StopLossPct']:.1f}% risk) with 2R target at ₹{risk_metrics['Target2R']} "
        f"gives a 1:2 risk/reward. Suggested position: {risk_metrics['Shares']} shares "
        f"(₹{risk_metrics['PositionValue']:,.0f}, {risk_metrics['PositionPct']:.1f}% of capital)."
    )

    # 5. Conviction
    if score >= 80 and canslim_total >= 60:
        conviction, reason = "HIGH", "Both technical score and CANSLIM confirm a strong multi-timeframe setup."
    elif score >= 60 and canslim_total >= 40:
        conviction, reason = "MEDIUM", "Setup is constructive but mixed CANSLIM signals warrant a smaller initial position."
    else:
        conviction, reason = "LOW", "Too many conditions are unfavourable to justify full-size commitment right now."

    return f"""
**1. INVESTMENT THESIS**
{thesis}

**2. TECHNICAL SETUP**
{sma_line} {rsi_line} {mom_line} {rs_line}

**3. CANSLIM ASSESSMENT**
{canslim_text}

**4. RISK ASSESSMENT**
{vol_line} {dd_line} {sizing_line}

**5. CONVICTION LEVEL: {conviction}**
{reason}
"""


# =========================================================
# BENCHMARK — robust fetch with fallbacks
# =========================================================

@st.cache_data(ttl=3600)
def fetch_benchmark():
    """
    Try multiple ticker variants — yfinance is inconsistent
    with index tickers on Streamlit Cloud.
    """
    for ticker in ["^NSEI", "^NSEI.NS", "NIFTYBEES.NS"]:
        try:
            df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.strip().title() for c in df.columns]
            if "Close" not in df.columns and "Adj Close" in df.columns:
                df.rename(columns={"Adj Close": "Close"}, inplace=True)
            if "Close" in df.columns and len(df) > 100:
                return df[["Close"]].dropna()
        except Exception:
            continue
    return pd.DataFrame()

benchmark_df = fetch_benchmark()

if benchmark_df.empty or "Close" not in benchmark_df.columns:
    st.error(
        "❌ Could not fetch benchmark data. This is a temporary Yahoo Finance issue — "
        "please wait 1 minute and reload the page."
    )
    st.stop()

# =========================================================
# SCREENER
# =========================================================

st.header("📊 Screener")

ticker_input = st.text_input(
    "Enter Tickers (comma-separated)",
    "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK"
)

if st.button("Run Screener"):
    tickers = [x.strip() for x in ticker_input.split(",")]
    screener_rows = []
    progress = st.progress(0)

    for i, ticker in enumerate(tickers):
        try:
            df = fetch_data(ticker)
            if df.empty or "Close" not in df.columns or len(df) < 250:
                continue
            metrics = compute_metrics(df, benchmark_df)
            canslim_scores, canslim_total = compute_canslim_score(df)
            risk = compute_risk_metrics(df)
            screener_rows.append({
                "Ticker": ticker,
                "Score": metrics["Score"],
                "CANSLIM": int(canslim_total),
                "Recommendation": metrics["Recommendation"],
                "Structure": metrics["Structure"],
                "RSI": metrics["RSI"],
                "Momentum%": metrics["Momentum"],
                "RelStrength%": metrics["RelativeStrength"],
                "Volatility%": metrics["Volatility"],
                "RiskTier": risk["RiskTier"],
                "StopLoss%": risk["StopLossPct"],
                "MaxDD%": risk["MaxDrawdown"]
            })
        except Exception:
            pass
        progress.progress((i + 1) / len(tickers))

    if screener_rows:
        st.dataframe(
            pd.DataFrame(screener_rows).sort_values(by="Score", ascending=False),
            use_container_width=True
        )
    else:
        st.error("No valid stock data fetched.")

# =========================================================
# PORTFOLIO BACKTEST
# =========================================================

st.header("📈 Portfolio Backtest")

portfolio_input = st.text_input(
    "Portfolio Tickers",
    "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK"
)

col_years, col_capital = st.columns(2)
with col_years:
    years = st.slider("Backtest Years", 1, 10, 5)
with col_capital:
    capital = st.number_input("Portfolio Capital (₹)", min_value=10000, max_value=10000000,
                               value=100000, step=10000, format="%d")

risk_per_trade = st.slider("Risk Per Trade (% of capital)", 0.5, 5.0, 1.5, 0.25)

if st.button("Run Portfolio Backtest"):
    tickers = [x.strip() for x in portfolio_input.split(",")]
    benchmark_returns = benchmark_df["Close"].pct_change().fillna(0)
    portfolio_returns, selected_tickers, portfolio_risk_data = [], [], []

    for ticker in tickers:
        try:
            df = fetch_data(ticker, period=f"{years}y")
            if df.empty or "Close" not in df.columns or len(df) < 250:
                continue
            selected_tickers.append(ticker)
            portfolio_returns.append(df["Close"].pct_change().fillna(0))
            risk = compute_risk_metrics(df, capital=capital / len(tickers), risk_per_trade_pct=risk_per_trade)
            portfolio_risk_data.append({"Ticker": ticker, **risk})
        except Exception:
            pass

    if portfolio_returns:
        aligned = pd.concat(portfolio_returns, axis=1)
        aligned.columns = selected_tickers
        strategy_returns = aligned.mean(axis=1)
        strategy_curve = (1 + strategy_returns).cumprod()
        benchmark_curve = (1 + benchmark_returns.reindex(strategy_curve.index).fillna(0)).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strategy_curve.index, y=strategy_curve, name="Strategy",
                                  line=dict(color="#2ecc71", width=2)))
        fig.add_trace(go.Scatter(x=benchmark_curve.index, y=benchmark_curve, name="Nifty 50",
                                  line=dict(color="#3498db", width=2, dash="dash")))
        fig.update_layout(template="plotly_dark", title="Portfolio vs Benchmark (Equal Weight)",
                          yaxis_title="Cumulative Return",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        strategy_return = (strategy_curve.iloc[-1] - 1) * 100
        bench_return = (benchmark_curve.iloc[-1] - 1) * 100
        cagr = (strategy_curve.iloc[-1] ** (1 / years) - 1) * 100
        rf_daily = 0.065 / 252
        excess = strategy_returns - rf_daily
        sharpe = (excess.mean() / strategy_returns.std()) * np.sqrt(252)
        downside = strategy_returns[strategy_returns < 0]
        sortino = (excess.mean() / downside.std()) * np.sqrt(252) if downside.std() != 0 else 0
        max_drawdown = ((strategy_curve / strategy_curve.cummax() - 1).min()) * 100
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Strategy Return", f"{strategy_return:.1f}%", f"{strategy_return - bench_return:+.1f}% vs Nifty")
        c2.metric("CAGR", f"{cagr:.1f}%")
        c3.metric("Sharpe Ratio", f"{sharpe:.2f}", help="Adjusted for 6.5% Indian risk-free rate")
        c4.metric("Sortino Ratio", f"{sortino:.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Max Drawdown", f"{max_drawdown:.1f}%")
        c6.metric("Calmar Ratio", f"{calmar:.2f}")
        c7.metric("Benchmark Return", f"{bench_return:.1f}%")
        c8.metric("Alpha", f"{strategy_return - bench_return:.1f}%")

        st.subheader("⚠️ Per-Stock Risk Profile")
        if portfolio_risk_data:
            rdf = pd.DataFrame(portfolio_risk_data)[
                ["Ticker", "RiskTier", "StopLoss", "StopLossPct",
                 "Target2R", "VolatilityAnnual", "MaxDrawdown", "Shares", "PositionValue"]
            ]
            rdf.columns = ["Ticker", "Risk Tier", "Stop ₹", "Stop %", "2R Target ₹",
                           "Vol %", "Max DD %", "Shares", "Position ₹"]
            st.dataframe(rdf, use_container_width=True)

        st.subheader("🧠 Portfolio Interpretation")
        if strategy_return > bench_return:
            st.success(f"✅ Strategy outperformed Nifty 50 by {strategy_return - bench_return:.1f}% over {years} years.")
        else:
            st.warning(f"⚠️ Strategy underperformed Nifty 50 by {bench_return - strategy_return:.1f}% over {years} years.")
        if sharpe > 1.5:
            st.info(f"📊 Sharpe of {sharpe:.2f} — strong risk-adjusted returns for Indian market conditions.")
        elif sharpe > 1:
            st.info(f"📊 Sharpe of {sharpe:.2f} — acceptable. Tighten stock selection to improve further.")
        else:
            st.warning(f"📊 Sharpe of {sharpe:.2f} — below acceptable. Review entry criteria and diversification.")
    else:
        st.error("No valid portfolio data fetched.")

# =========================================================
# SINGLE STOCK ANALYSIS
# =========================================================

st.header("🔎 Single Stock Deep Dive")

col_t, col_c = st.columns([2, 1])
with col_t:
    single_ticker = st.text_input("Ticker", "RELIANCE")
with col_c:
    analysis_capital = st.number_input("Capital for Sizing (₹)", min_value=10000,
                                        max_value=10000000, value=500000, step=10000, format="%d")

analysis_risk_pct = st.slider("Risk Per Trade %", 0.5, 5.0, 1.5, 0.25, key="single_risk")

if st.button("Analyze Stock", type="primary"):

    df = fetch_data(single_ticker)

    if df.empty or "Close" not in df.columns or len(df) < 250:
        st.error("Not enough data (minimum 250 trading days required).")
    else:
        with st.spinner("Computing metrics..."):
            metrics = compute_metrics(df, benchmark_df)
            canslim_scores, canslim_total = compute_canslim_score(df)
            risk_metrics = compute_risk_metrics(df, capital=analysis_capital, risk_per_trade_pct=analysis_risk_pct)
            scenarios = compute_probability_scenarios(
                metrics["Score"], canslim_total,
                metrics["RSI"], metrics["Momentum"], metrics["RelativeStrength"]
            )

        st.subheader(
            f"Recommendation: {metrics['Recommendation']}  |  Score: {metrics['Score']}/100  |  CANSLIM: {int(canslim_total)}/100"
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Price", f"₹{metrics['Price']}")
        c2.metric("RSI (14)", metrics["RSI"])
        c3.metric("Momentum 3M", f"{metrics['Momentum']:+.1f}%")
        c4.metric("Rel. Strength", f"{metrics['RelativeStrength']:+.1f}%")
        c5.metric("Annual Vol", f"{metrics['Volatility']:.1f}%")

        # Price Chart
        st.subheader("📉 Price Chart")
        close = df["Close"]
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=close, name="Price", line=dict(color="#ffffff", width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA 50", line=dict(color="#f39c12", width=1.2, dash="dash")))
        fig.add_trace(go.Scatter(x=df.index, y=sma200, name="SMA 200", line=dict(color="#e74c3c", width=1.2, dash="dot")))
        fig.add_hline(y=risk_metrics["StopLoss"], line_color="#e74c3c", line_dash="dash", line_width=1,
                      annotation_text=f"Stop ₹{risk_metrics['StopLoss']}", annotation_position="bottom right")
        fig.add_hline(y=risk_metrics["Target2R"], line_color="#2ecc71", line_dash="dash", line_width=1,
                      annotation_text=f"2R Target ₹{risk_metrics['Target2R']}", annotation_position="top right")
        fig.update_layout(template="plotly_dark", height=420, margin=dict(l=0, r=0, t=30, b=0),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # CANSLIM + Risk side by side
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 📐 CANSLIM Breakdown")
            for key, (label, max_score) in {
                "C_CurrentMomentum": ("C – Current Momentum", 20),
                "A_AnnualTrend": ("A – Annual Trend", 15),
                "N_NewHighs": ("N – New High Proximity", 15),
                "S_SupplyDemand": ("S – Supply/Demand (Vol)", 15),
                "L_Leader": ("L – Leader vs Laggard", 15),
                "I_Institutional": ("I – Institutional Support", 10),
                "M_MarketDirection": ("M – Market Direction", 10),
            }.items():
                val = canslim_scores.get(key, 0)
                pct = int((val / max_score) * 100)
                st.markdown(f"**{label}** — {val:.0f}/{max_score}")
                st.markdown(
                    f'<div class="canslim-bar"><div class="canslim-fill" style="width:{pct}%"></div></div>',
                    unsafe_allow_html=True
                )
            st.markdown(f"**Total: {int(canslim_total)}/100**")

        with col_right:
            st.markdown("### ⚠️ Risk Management")
            risk_color = {"Low": "risk-low", "Medium": "risk-med", "High": "risk-high"}[risk_metrics["RiskTier"]]
            st.markdown(f'Risk Tier: <span class="risk-badge {risk_color}">{risk_metrics["RiskTier"]}</span>',
                        unsafe_allow_html=True)
            st.markdown("**Entry / Exit Levels**")
            st.dataframe(pd.DataFrame({
                "Level": ["Current Price", "Stop Loss (2× ATR)", "1R Target", "2R Target", "3R Target"],
                "Price (₹)": [
                    f"₹{risk_metrics['CurrentPrice']}",
                    f"₹{risk_metrics['StopLoss']} ({risk_metrics['StopLossPct']:.1f}% risk)",
                    f"₹{risk_metrics['Target1R']}",
                    f"₹{risk_metrics['Target2R']}",
                    f"₹{risk_metrics['Target3R']}"
                ]
            }), use_container_width=True, hide_index=True)
            st.markdown("**Position Sizing**")
            st.markdown(f"""
- Capital: ₹{analysis_capital:,.0f}
- Risk per trade: {analysis_risk_pct}% = ₹{risk_metrics['RiskAmount']:,.0f}
- ATR (14-day): ₹{risk_metrics['ATR']}
- Suggested shares: **{risk_metrics['Shares']}**
- Position value: **₹{risk_metrics['PositionValue']:,.0f}** ({risk_metrics['PositionPct']:.1f}% of capital)
- Max Drawdown (historical): {risk_metrics['MaxDrawdown']:.1f}%
- Current Drawdown: {risk_metrics['CurrentDrawdown']:.1f}%
""")

        # Probability Scenarios
        st.markdown("### 🎯 Probability Scenarios")
        st.caption("Derived from Score, CANSLIM, RSI, Momentum, and Relative Strength — not hardcoded")

        bull_prob = scenarios["Bullish Continuation"]
        side_prob = scenarios["Sideways Consolidation"]
        bear_prob = scenarios["Bearish Breakdown"]

        s1, s2, s3 = st.columns(3)
        s1.metric("🟢 Bullish Continuation", f"{bull_prob:.0f}%")
        s2.metric("🟡 Sideways Consolidation", f"{side_prob:.0f}%")
        s3.metric("🔴 Bearish Breakdown", f"{bear_prob:.0f}%")

        fig_prob = go.Figure(go.Bar(
            x=[bull_prob, side_prob, bear_prob], y=["Bullish", "Sideways", "Bearish"],
            orientation='h', marker_color=["#2ecc71", "#f39c12", "#e74c3c"],
            text=[f"{v:.0f}%" for v in [bull_prob, side_prob, bear_prob]], textposition="inside"
        ))
        fig_prob.update_layout(template="plotly_dark", height=160,
                               margin=dict(l=0, r=0, t=10, b=10),
                               showlegend=False, xaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_prob, use_container_width=True)

        # Bullish / Risk Factors
        col_bull, col_risk = st.columns(2)
        with col_bull:
            st.markdown("### ✅ Bullish Factors")
            bullish = []
            if metrics["Price"] > metrics["SMA200"]:
                bullish.append("Price above 200DMA — long-term institutional support intact")
            if metrics["RelativeStrength"] > 0:
                bullish.append(f"Outperforming Nifty by {metrics['RelativeStrength']:.1f}% (3M)")
            if metrics["RSI"] > 55:
                bullish.append(f"RSI at {metrics['RSI']:.0f} — healthy momentum territory")
            if canslim_scores.get("N_NewHighs", 0) >= 15:
                bullish.append("Near 52-week highs — price discovery phase")
            if canslim_scores.get("S_SupplyDemand", 0) >= 15:
                bullish.append("Volume trend confirming price — institutional accumulation signal")
            if not bullish:
                bullish.append("No major bullish technical factors currently visible")
            for item in bullish:
                st.write("•", item)

        with col_risk:
            st.markdown("### ⚠️ Risk Factors")
            risks = []
            if metrics["SMA50"] < metrics["SMA200"]:
                risks.append("Death cross — SMA50 below SMA200, medium-term trend is bearish")
            if metrics["Momentum"] < 0:
                risks.append(f"Negative 3-month momentum ({metrics['Momentum']:.1f}%) — selling pressure")
            if risk_metrics["RiskTier"] == "High":
                risks.append(f"High volatility ({risk_metrics['VolatilityAnnual']:.0f}% annualised) — wide stops required")
            if risk_metrics["MaxDrawdown"] < -40:
                risks.append(f"Historical max drawdown of {risk_metrics['MaxDrawdown']:.0f}% — deep correction risk")
            if metrics["RSI"] > 75:
                risks.append(f"RSI at {metrics['RSI']:.0f} — overbought territory, correction risk")
            if not risks:
                risks.append("No major technical weakness currently visible")
            for item in risks:
                st.write("•", item)

        # What To Watch
        st.markdown("### 👀 What To Watch Next")
        watch_items = []
        if metrics["Momentum"] < 0:
            watch_items.append("Watch for momentum to turn positive — key early signal")
        if metrics["SMA50"] < metrics["SMA200"]:
            watch_items.append("Watch for golden cross: SMA50 crossing above SMA200")
        if metrics["RSI"] > 70:
            watch_items.append(f"RSI at {metrics['RSI']:.0f} — watch for cooling off before adding size")
        if risk_metrics["CurrentDrawdown"] < -15:
            watch_items.append(f"Currently {risk_metrics['CurrentDrawdown']:.0f}% from highs — watch for base formation")
        if canslim_scores.get("S_SupplyDemand", 0) < 8:
            watch_items.append("Volume trend is weak — watch for volume expansion on rallies")
        if not watch_items:
            watch_items.append("Trend structure healthy — monitor for continuation signals")
        for item in watch_items:
            st.write("•", item)

        # Research Note
        st.markdown("### 🧠 Research Note")
        st.caption("Smart rule-based analysis — structured like an institutional research note · No API cost")

        note = generate_research_note(
            single_ticker, metrics, canslim_scores,
            canslim_total, risk_metrics, scenarios
        )
        st.markdown(note)
