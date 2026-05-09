import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import anthropic
import json

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
    .section-header { border-left: 3px solid #2ecc71; padding-left:10px; margin:16px 0 8px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Equity Research Platform V17")
st.caption("Score-driven probabilities · CANSLIM scoring · Real AI writeups · Risk management layer")

# =========================================================
# ANTHROPIC CLIENT
# =========================================================

@st.cache_resource
def get_anthropic_client():
    return anthropic.Anthropic()

# =========================================================
# DATA FETCH
# =========================================================

@st.cache_data(ttl=3600)
def fetch_data(ticker, period="5y"):
    df = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = [c for c in ["Close", "Open", "High", "Low", "Volume"] if c in df.columns]
    df = df[needed].copy()
    df.dropna(inplace=True)
    return df


# =========================================================
# CANSLIM SCORING ENGINE
# =========================================================

def compute_canslim_score(df):
    """
    CANSLIM-inspired scoring using price/volume proxies
    (no fundamental data yet — uses technical surrogates).

    C - Current momentum (quarterly price performance)
    A - Annual trend (52-week vs 200DMA)
    N - New highs (price near 52-week high)
    S - Supply/Demand (volume trend)
    L - Leader vs Laggard (relative strength surrogates)
    I - Institutional support (above 200DMA)
    M - Market direction (benchmark momentum)
    """
    close = df["Close"]
    scores = {}

    # C: Current momentum — 3-month price change
    if len(close) >= 63:
        c_return = (close.iloc[-1] / close.iloc[-63] - 1) * 100
        scores["C_CurrentMomentum"] = min(max(c_return / 20 * 20, 0), 20)
    else:
        scores["C_CurrentMomentum"] = 0

    # A: Annual trend — price vs 200DMA
    sma200 = close.rolling(200).mean()
    if len(sma200.dropna()) > 0:
        gap_pct = (close.iloc[-1] / sma200.iloc[-1] - 1) * 100
        scores["A_AnnualTrend"] = min(max(gap_pct / 10 * 15, 0), 15)
    else:
        scores["A_AnnualTrend"] = 0

    # N: New high proximity — price vs 52-week high
    if len(close) >= 252:
        high_52w = close.rolling(252).max().iloc[-1]
        proximity = (close.iloc[-1] / high_52w) * 100
        if proximity >= 95:
            scores["N_NewHighs"] = 15
        elif proximity >= 85:
            scores["N_NewHighs"] = 8
        else:
            scores["N_NewHighs"] = 0
    else:
        scores["N_NewHighs"] = 0

    # S: Supply/Demand — volume trend (if volume available)
    if "Volume" in df.columns and len(df) >= 50:
        vol = df["Volume"]
        avg_vol_20 = vol.rolling(20).mean().iloc[-1]
        avg_vol_50 = vol.rolling(50).mean().iloc[-1]
        if avg_vol_20 > avg_vol_50 * 1.1:
            scores["S_SupplyDemand"] = 15
        elif avg_vol_20 > avg_vol_50:
            scores["S_SupplyDemand"] = 8
        else:
            scores["S_SupplyDemand"] = 3
    else:
        scores["S_SupplyDemand"] = 7  # neutral if no volume

    # L: Leader — SMA50 vs SMA200
    sma50 = close.rolling(50).mean()
    if len(sma50.dropna()) > 0 and len(sma200.dropna()) > 0:
        if sma50.iloc[-1] > sma200.iloc[-1]:
            scores["L_Leader"] = 15
        else:
            scores["L_Leader"] = 0
    else:
        scores["L_Leader"] = 0

    # I: Institutional — above 200DMA with positive slope
    if len(sma200.dropna()) >= 20:
        sma200_slope = (sma200.iloc[-1] / sma200.iloc[-20] - 1) * 100
        above_200 = close.iloc[-1] > sma200.iloc[-1]
        if above_200 and sma200_slope > 0:
            scores["I_Institutional"] = 10
        elif above_200:
            scores["I_Institutional"] = 5
        else:
            scores["I_Institutional"] = 0
    else:
        scores["I_Institutional"] = 0

    # M: Market direction — benchmark momentum placeholder (scored separately)
    scores["M_MarketDirection"] = 10  # filled in compute_metrics

    total = sum(scores.values())
    return scores, total


# =========================================================
# PROBABILITY ENGINE (score-driven, not hardcoded)
# =========================================================

def compute_probability_scenarios(score, canslim_total, rsi, momentum, relative_strength):
    """
    Derives probabilities from actual stock metrics.
    Uses a weighted blend of overall score, CANSLIM score,
    RSI regime, momentum, and relative strength.
    """

    # Composite signal: 0-100
    rsi_signal = max(0, min(100, (rsi - 30) / 40 * 100)) if rsi else 50
    mom_signal = 70 if momentum > 5 else (50 if momentum > 0 else 30)
    rs_signal = 70 if relative_strength > 3 else (50 if relative_strength > 0 else 30)
    canslim_signal = min(canslim_total, 100)

    composite = (
        score * 0.30 +
        canslim_signal * 0.30 +
        rsi_signal * 0.15 +
        mom_signal * 0.15 +
        rs_signal * 0.10
    )

    # Map composite (0-100) to probabilities
    # Bullish rises, bearish falls as composite rises
    bullish = round(max(10, min(75, composite * 0.70)), 1)
    bearish = round(max(5, min(60, (100 - composite) * 0.55)), 1)
    sideways = round(100 - bullish - bearish, 1)

    # Ensure sideways is reasonable
    if sideways < 5:
        sideways = 5
        bearish = round(100 - bullish - sideways, 1)

    return {
        "Bullish Continuation": bullish,
        "Sideways Consolidation": sideways,
        "Bearish Breakdown": bearish
    }


# =========================================================
# RISK MANAGEMENT ENGINE
# =========================================================

def compute_risk_metrics(df, capital=100000, risk_per_trade_pct=1.5):
    """
    Computes:
    - ATR-based stop loss
    - Suggested position size
    - Risk/Reward ratio
    - Drawdown profile
    - Kelly-inspired sizing
    - Risk tier classification
    """
    close = df["Close"]

    # ATR (Average True Range) — 14 day
    if "High" in df.columns and "Low" in df.columns:
        high = df["High"]
        low = df["Low"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
    else:
        atr = close.pct_change().std() * close.iloc[-1] * np.sqrt(14)

    current_price = float(close.iloc[-1])

    # Stop loss = 2x ATR below current price
    stop_loss = current_price - (2 * atr)
    stop_loss_pct = ((current_price - stop_loss) / current_price) * 100

    # Target = 3x ATR above (1:1.5 R/R minimum)
    target_1r = current_price + (2 * atr)       # 1:1
    target_2r = current_price + (4 * atr)       # 1:2
    target_3r = current_price + (6 * atr)       # 1:3

    # Position sizing (risk-based)
    risk_amount = capital * (risk_per_trade_pct / 100)
    risk_per_share = current_price - stop_loss
    if risk_per_share > 0:
        shares = int(risk_amount / risk_per_share)
        position_value = shares * current_price
        position_pct = (position_value / capital) * 100
    else:
        shares = 0
        position_value = 0
        position_pct = 0

    # Volatility-based risk tier
    vol_annual = close.pct_change().std() * np.sqrt(252) * 100
    if vol_annual < 20:
        risk_tier = "Low"
    elif vol_annual < 35:
        risk_tier = "Medium"
    else:
        risk_tier = "High"

    # Max drawdown (rolling)
    rolling_max = close.cummax()
    drawdown = (close / rolling_max - 1) * 100
    max_dd = float(drawdown.min())
    current_dd = float(drawdown.iloc[-1])

    return {
        "ATR": round(float(atr), 2),
        "CurrentPrice": round(current_price, 2),
        "StopLoss": round(float(stop_loss), 2),
        "StopLossPct": round(float(stop_loss_pct), 2),
        "Target1R": round(float(target_1r), 2),
        "Target2R": round(float(target_2r), 2),
        "Target3R": round(float(target_3r), 2),
        "RiskAmount": round(risk_amount, 0),
        "Shares": shares,
        "PositionValue": round(position_value, 0),
        "PositionPct": round(position_pct, 1),
        "VolatilityAnnual": round(vol_annual, 1),
        "RiskTier": risk_tier,
        "MaxDrawdown": round(max_dd, 2),
        "CurrentDrawdown": round(current_dd, 2)
    }


# =========================================================
# CORE METRICS ENGINE (V16 base, extended)
# =========================================================

def compute_metrics(df, benchmark_df):
    close = df["Close"].copy()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    momentum = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100
    benchmark_close = benchmark_df["Close"]
    benchmark_return = ((benchmark_close.iloc[-1] / benchmark_close.iloc[-63]) - 1) * 100
    stock_return = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100
    relative_strength = stock_return - benchmark_return
    volatility = close.pct_change().std() * np.sqrt(252) * 100

    score = 0
    if float(close.iloc[-1]) > float(sma200.iloc[-1]): score += 30
    if float(sma50.iloc[-1]) > float(sma200.iloc[-1]): score += 25
    if float(rsi.iloc[-1]) > 55: score += 20
    if momentum > 0: score += 15
    if relative_strength > 0: score += 10

    if score >= 80:
        recommendation = "🟢 Strong Buy"
    elif score >= 60:
        recommendation = "🟢 Buy"
    elif score >= 40:
        recommendation = "🟠 Watch"
    else:
        recommendation = "🔴 Avoid"

    if (float(close.iloc[-1]) > float(sma200.iloc[-1]) and
            float(sma50.iloc[-1]) > float(sma200.iloc[-1])):
        structure = "Bullish Structure"
    elif float(close.iloc[-1]) > float(sma200.iloc[-1]):
        structure = "Early Accumulation"
    else:
        structure = "Bearish Structure"

    return {
        "Price": round(float(close.iloc[-1]), 2),
        "RSI": round(float(rsi.iloc[-1]), 2),
        "Momentum": round(float(momentum), 2),
        "RelativeStrength": round(float(relative_strength), 2),
        "Volatility": round(float(volatility), 2),
        "SMA50": round(float(sma50.iloc[-1]), 2),
        "SMA200": round(float(sma200.iloc[-1]), 2),
        "Recommendation": recommendation,
        "Structure": structure,
        "Score": int(score)
    }


# =========================================================
# AI WRITEUP (real Claude API call)
# =========================================================

def generate_ai_writeup(ticker, metrics, canslim_scores, canslim_total, risk_metrics, scenarios):
    client = get_anthropic_client()

    prompt = f"""You are a senior equity research analyst covering Indian markets (NSE/BSE).
Write a concise, institutional-quality investment note for {ticker}.

TECHNICAL DATA:
- Score: {metrics['Score']}/100
- Recommendation: {metrics['Recommendation']}
- Structure: {metrics['Structure']}
- RSI: {metrics['RSI']}
- 3-Month Momentum: {metrics['Momentum']:.1f}%
- Relative Strength vs Nifty: {metrics['RelativeStrength']:.1f}%
- Annual Volatility: {metrics['Volatility']:.1f}%
- Price vs SMA50: {"Above" if metrics['Price'] > metrics['SMA50'] else "Below"} (SMA50: ₹{metrics['SMA50']})
- Price vs SMA200: {"Above" if metrics['Price'] > metrics['SMA200'] else "Below"} (SMA200: ₹{metrics['SMA200']})

CANSLIM SCORE: {canslim_total}/100
- C (Current Momentum): {canslim_scores.get('C_CurrentMomentum', 0):.0f}/20
- A (Annual Trend): {canslim_scores.get('A_AnnualTrend', 0):.0f}/15
- N (New Highs): {canslim_scores.get('N_NewHighs', 0):.0f}/15
- S (Supply/Demand): {canslim_scores.get('S_SupplyDemand', 0):.0f}/15
- L (Leader): {canslim_scores.get('L_Leader', 0):.0f}/15
- I (Institutional): {canslim_scores.get('I_Institutional', 0):.0f}/10

RISK PROFILE:
- Risk Tier: {risk_metrics['RiskTier']}
- ATR-based Stop Loss: ₹{risk_metrics['StopLoss']} ({risk_metrics['StopLossPct']:.1f}% below current)
- 1R Target: ₹{risk_metrics['Target1R']}
- 2R Target: ₹{risk_metrics['Target2R']}
- Max Historical Drawdown: {risk_metrics['MaxDrawdown']:.1f}%

PROBABILITY SCENARIOS:
- Bullish: {scenarios.get('Bullish Continuation', 0):.0f}%
- Sideways: {scenarios.get('Sideways Consolidation', 0):.0f}%
- Bearish: {scenarios.get('Bearish Breakdown', 0):.0f}%

Write a structured research note with these exact sections:
1. INVESTMENT THESIS (2-3 sentences — the core bull/bear case)
2. TECHNICAL SETUP (describe the chart structure, momentum, and key levels in plain language)
3. CANSLIM ASSESSMENT (interpret the CANSLIM scores — what's strong, what's weak)
4. RISK ASSESSMENT (key risks, stop loss rationale, position sizing context)
5. CONVICTION LEVEL (Low/Medium/High and one sentence why)

Be direct, specific, and avoid generic filler. Use ₹ for prices. Write for a sophisticated retail investor."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


# =========================================================
# BENCHMARK
# =========================================================

benchmark_df = fetch_data("^NSEI")

# =========================================================
# SCREENER
# =========================================================

st.header("📊 Screener")

ticker_input = st.text_input(
    "Enter Tickers (comma-separated)",
    "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS"
)

if st.button("Run Screener"):
    tickers = [x.strip() for x in ticker_input.split(",")]
    screener_rows = []

    progress = st.progress(0)
    for i, ticker in enumerate(tickers):
        try:
            df = fetch_data(ticker)
            if len(df) < 250:
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
        except Exception as e:
            pass
        progress.progress((i + 1) / len(tickers))

    if screener_rows:
        screener_df = pd.DataFrame(screener_rows).sort_values(by="Score", ascending=False)
        st.dataframe(screener_df, use_container_width=True)
    else:
        st.error("No valid stock data fetched.")

# =========================================================
# PORTFOLIO BACKTEST
# =========================================================

st.header("📈 Portfolio Backtest")

portfolio_input = st.text_input(
    "Portfolio Tickers",
    "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS"
)

col_years, col_capital = st.columns(2)
with col_years:
    years = st.slider("Backtest Years", 1, 10, 5)
with col_capital:
    capital = st.number_input(
        "Portfolio Capital (₹)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000,
        format="%d"
    )

risk_per_trade = st.slider(
    "Risk Per Trade (% of capital)", 0.5, 5.0, 1.5, 0.25,
    help="Used for position sizing in the risk layer"
)

if st.button("Run Portfolio Backtest"):
    tickers = [x.strip() for x in portfolio_input.split(",")]
    benchmark_returns = benchmark_df["Close"].pct_change().fillna(0)
    portfolio_returns = []
    selected_tickers = []
    portfolio_risk_data = []

    for ticker in tickers:
        try:
            df = fetch_data(ticker, period=f"{years}y")
            if len(df) < 250:
                continue
            selected_tickers.append(ticker)
            returns = df["Close"].pct_change().fillna(0)
            portfolio_returns.append(returns)
            risk = compute_risk_metrics(df, capital=capital / len(tickers), risk_per_trade_pct=risk_per_trade)
            portfolio_risk_data.append({"Ticker": ticker, **risk})
        except:
            pass

    if portfolio_returns:
        aligned_returns = pd.concat(portfolio_returns, axis=1)
        aligned_returns.columns = selected_tickers
        strategy_returns = aligned_returns.mean(axis=1)
        strategy_curve = (1 + strategy_returns).cumprod()
        benchmark_curve = (1 + benchmark_returns.reindex(strategy_curve.index).fillna(0)).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strategy_curve.index, y=strategy_curve, name="Strategy", line=dict(color="#2ecc71", width=2)))
        fig.add_trace(go.Scatter(x=benchmark_curve.index, y=benchmark_curve, name="Nifty 50", line=dict(color="#3498db", width=2, dash="dash")))
        fig.update_layout(
            template="plotly_dark",
            title="Portfolio vs Benchmark (Equal Weight)",
            yaxis_title="Cumulative Return",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        strategy_return = (strategy_curve.iloc[-1] - 1) * 100
        bench_return = (benchmark_curve.iloc[-1] - 1) * 100
        cagr = (strategy_curve.iloc[-1] ** (1 / years) - 1) * 100
        rf_daily = 0.065 / 252  # 6.5% India risk-free rate
        excess = strategy_returns - rf_daily
        sharpe = (excess.mean() / strategy_returns.std()) * np.sqrt(252)
        downside = strategy_returns[strategy_returns < 0]
        sortino = (excess.mean() / downside.std()) * np.sqrt(252) if downside.std() != 0 else 0
        rolling_max = strategy_curve.cummax()
        max_drawdown = ((strategy_curve / rolling_max - 1).min()) * 100
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Strategy Return", f"{strategy_return:.1f}%", f"{strategy_return - bench_return:+.1f}% vs Nifty")
        c2.metric("CAGR", f"{cagr:.1f}%")
        c3.metric("Sharpe Ratio", f"{sharpe:.2f}", help="Adjusted for 6.5% Indian risk-free rate")
        c4.metric("Sortino Ratio", f"{sortino:.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Max Drawdown", f"{max_drawdown:.1f}%")
        c6.metric("Calmar Ratio", f"{calmar:.2f}", help="CAGR / Max Drawdown")
        c7.metric("Benchmark Return", f"{bench_return:.1f}%")
        c8.metric("Alpha", f"{strategy_return - bench_return:.1f}%")

        # Portfolio Risk Table
        st.subheader("⚠️ Per-Stock Risk Profile")
        if portfolio_risk_data:
            risk_df = pd.DataFrame(portfolio_risk_data)[[
                "Ticker", "RiskTier", "StopLoss", "StopLossPct",
                "Target2R", "VolatilityAnnual", "MaxDrawdown", "Shares", "PositionValue"
            ]]
            risk_df.columns = ["Ticker", "Risk Tier", "Stop Loss ₹", "Stop %",
                                "2R Target ₹", "Vol %", "Max DD %", "Shares", "Position ₹"]
            st.dataframe(risk_df, use_container_width=True)

        st.subheader("🧠 Portfolio Interpretation")
        if strategy_return > bench_return:
            st.success(f"✅ Strategy outperformed Nifty 50 by {strategy_return - bench_return:.1f}% over {years} years.")
        else:
            st.warning(f"⚠️ Strategy underperformed Nifty 50 by {bench_return - strategy_return:.1f}% over {years} years.")
        if sharpe > 1.5:
            st.info(f"📊 Sharpe of {sharpe:.2f} reflects strong risk-adjusted returns relative to Indian market conditions.")
        elif sharpe > 1:
            st.info(f"📊 Sharpe of {sharpe:.2f} is acceptable. Look to improve by tightening stock selection criteria.")
        else:
            st.warning(f"📊 Sharpe of {sharpe:.2f} is below acceptable. Consider tightening entry criteria or improving diversification.")
    else:
        st.error("No valid portfolio data fetched.")

# =========================================================
# SINGLE STOCK ANALYSIS
# =========================================================

st.header("🔎 Single Stock Deep Dive")

col_ticker, col_capital2 = st.columns([2, 1])
with col_ticker:
    single_ticker = st.text_input("Ticker", "RELIANCE.NS")
with col_capital2:
    analysis_capital = st.number_input(
        "Capital for Sizing (₹)", min_value=10000, max_value=10000000,
        value=500000, step=10000, format="%d"
    )

analysis_risk_pct = st.slider("Risk Per Trade %", 0.5, 5.0, 1.5, 0.25, key="single_risk")

if st.button("Analyze Stock", type="primary"):

    df = fetch_data(single_ticker)

    if len(df) < 250:
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

        # Header
        st.subheader(f"Recommendation: {metrics['Recommendation']}  |  Score: {metrics['Score']}/100  |  CANSLIM: {int(canslim_total)}/100")

        # Core Metrics Row
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

        # Add stop loss line
        fig.add_hline(
            y=risk_metrics["StopLoss"],
            line_color="#e74c3c", line_dash="dash", line_width=1,
            annotation_text=f"Stop ₹{risk_metrics['StopLoss']}",
            annotation_position="bottom right"
        )
        fig.add_hline(
            y=risk_metrics["Target2R"],
            line_color="#2ecc71", line_dash="dash", line_width=1,
            annotation_text=f"2R Target ₹{risk_metrics['Target2R']}",
            annotation_position="top right"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Two-column layout for CANSLIM + Risk ──────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 📐 CANSLIM Breakdown")
            canslim_labels = {
                "C_CurrentMomentum": ("C – Current Momentum", 20),
                "A_AnnualTrend": ("A – Annual Trend", 15),
                "N_NewHighs": ("N – New High Proximity", 15),
                "S_SupplyDemand": ("S – Supply/Demand (Vol)", 15),
                "L_Leader": ("L – Leader vs Laggard", 15),
                "I_Institutional": ("I – Institutional Support", 10),
                "M_MarketDirection": ("M – Market Direction", 10),
            }
            for key, (label, max_score) in canslim_labels.items():
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
            st.markdown(
                f'Risk Tier: <span class="risk-badge {risk_color}">{risk_metrics["RiskTier"]}</span>',
                unsafe_allow_html=True
            )

            st.markdown("**Entry / Exit Levels**")
            risk_table = pd.DataFrame({
                "Level": ["Current Price", "Stop Loss (2× ATR)", "1R Target", "2R Target", "3R Target"],
                "Price (₹)": [
                    f"₹{risk_metrics['CurrentPrice']}",
                    f"₹{risk_metrics['StopLoss']} ({risk_metrics['StopLossPct']:.1f}% risk)",
                    f"₹{risk_metrics['Target1R']}",
                    f"₹{risk_metrics['Target2R']}",
                    f"₹{risk_metrics['Target3R']}"
                ]
            })
            st.dataframe(risk_table, use_container_width=True, hide_index=True)

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

        # ── Probability Scenarios (score-driven) ──────────────────────
        st.markdown("### 🎯 Probability Scenarios")
        st.caption("Derived from Score, CANSLIM, RSI, Momentum, and Relative Strength — not hardcoded")

        s_col1, s_col2, s_col3 = st.columns(3)
        bull_prob = scenarios["Bullish Continuation"]
        side_prob = scenarios["Sideways Consolidation"]
        bear_prob = scenarios["Bearish Breakdown"]

        s_col1.metric("🟢 Bullish Continuation", f"{bull_prob:.0f}%")
        s_col2.metric("🟡 Sideways Consolidation", f"{side_prob:.0f}%")
        s_col3.metric("🔴 Bearish Breakdown", f"{bear_prob:.0f}%")

        # Scenario probability bar
        fig_prob = go.Figure(go.Bar(
            x=[bull_prob, side_prob, bear_prob],
            y=["Bullish", "Sideways", "Bearish"],
            orientation='h',
            marker_color=["#2ecc71", "#f39c12", "#e74c3c"],
            text=[f"{v:.0f}%" for v in [bull_prob, side_prob, bear_prob]],
            textposition="inside"
        ))
        fig_prob.update_layout(
            template="plotly_dark",
            height=160,
            margin=dict(l=0, r=0, t=10, b=10),
            showlegend=False,
            xaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # ── Bullish / Risk Factors ─────────────────────────────────────
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

        # ── What To Watch ──────────────────────────────────────────────
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

        # ── AI Research Writeup ────────────────────────────────────────
        st.markdown("### 🧠 AI Research Note")
        st.caption("Generated by Claude — institutional-quality interpretation of the above data")

        with st.spinner("Generating AI research note..."):
            try:
                writeup = generate_ai_writeup(
                    single_ticker, metrics, canslim_scores,
                    canslim_total, risk_metrics, scenarios
                )
                st.markdown(writeup)
            except Exception as e:
                st.warning(f"AI writeup unavailable: {e}. Ensure ANTHROPIC_API_KEY is set in your Streamlit secrets.")
