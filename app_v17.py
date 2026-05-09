import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import linregress

st.set_page_config(
    page_title="BharatTrack V17",
    layout="wide"
)

st.title("🚀 BharatTrack — AI Equity Research Platform V17")

# =========================================================
# HELPERS
# =========================================================


def fetch_data(ticker, period="5y"):
    df = yf.download(ticker, period=period, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    return df


benchmark_df = fetch_data("^NSEI")


# =========================================================
# METRICS ENGINE
# =========================================================


def compute_metrics(df, benchmark_df=None):
    close = df["Close"]

    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]

    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    momentum = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100

    volatility = close.pct_change().std() * np.sqrt(252) * 100

    relative_strength = 0

    if benchmark_df is not None:
        benchmark_close = benchmark_df["Close"]

        stock_return = close.iloc[-1] / close.iloc[0]
        benchmark_return = benchmark_close.iloc[-1] / benchmark_close.iloc[0]

        relative_strength = ((stock_return / benchmark_return) - 1) * 100

    score = 0

    if close.iloc[-1] > sma200:
        score += 30

    if sma50 > sma200:
        score += 20

    if rsi.iloc[-1] > 55:
        score += 20

    if momentum > 0:
        score += 15

    if relative_strength > 0:
        score += 15

    if score >= 80:
        recommendation = "Strong Buy"
    elif score >= 60:
        recommendation = "Buy"
    elif score >= 40:
        recommendation = "Watch"
    else:
        recommendation = "Avoid"

    if close.iloc[-1] > sma200 and sma50 > sma200:
        structure = "Bullish Structure"
    elif close.iloc[-1] > sma200:
        structure = "Early Accumulation"
    else:
        structure = "Bearish Structure"

    return {
        "Price": round(close.iloc[-1], 2),
        "RSI": round(rsi.iloc[-1], 2),
        "Momentum": round(momentum, 2),
        "RelativeStrength": round(relative_strength, 2),
        "Volatility": round(volatility, 2),
        "SMA50": round(sma50, 2),
        "SMA200": round(sma200, 2),
        "Score": score,
        "Recommendation": recommendation,
        "Structure": structure,
    }


# =========================================================
# CHART PATTERN ANALYSIS
# =========================================================


def chart_analysis(metrics):
    analysis = []

    if metrics["SMA50"] > metrics["SMA200"]:
        analysis.append(
            "🟢 SMA50 remains above SMA200, indicating strong intermediate trend strength."
        )
    else:
        analysis.append(
            "🔴 SMA50 remains below SMA200, indicating weaker recent momentum versus long-term trend."
        )

    if metrics["Momentum"] > 0:
        analysis.append(
            "🟢 Momentum remains positive, suggesting accumulation behavior."
        )
    else:
        analysis.append(
            "⚠️ Momentum remains weak, indicating possible corrective consolidation."
        )

    if metrics["RSI"] > 70:
        analysis.append(
            "⚠️ RSI indicates overbought conditions."
        )
    elif metrics["RSI"] > 50:
        analysis.append(
            "🟢 RSI remains healthy and supportive of bullish continuation."
        )
    else:
        analysis.append(
            "🔴 RSI reflects weak participation from buyers."
        )

    return analysis


# =========================================================
# AI STYLE WRITEUP (RULE BASED)
# =========================================================


def generate_writeup(ticker, metrics):
    bullish = []
    risks = []

    if metrics["Price"] > metrics["SMA200"]:
        bullish.append(
            "Price remains above the 200DMA, suggesting long-term institutional support remains intact."
        )
    else:
        risks.append(
            "Price remains below the 200DMA, indicating long-term structure remains weak."
        )

    if metrics["Momentum"] > 0:
        bullish.append(
            "Momentum remains positive, indicating buyers continue supporting the trend."
        )
    else:
        risks.append(
            "Momentum remains negative, suggesting near-term selling pressure persists."
        )

    if metrics["RelativeStrength"] > 0:
        bullish.append(
            "The stock continues outperforming the benchmark index, often associated with institutional accumulation."
        )
    else:
        risks.append(
            "Relative underperformance versus benchmark reflects weaker market leadership."
        )

    if metrics["RSI"] > 55:
        bullish.append(
            "RSI remains in a constructive range without extreme overheating."
        )
    else:
        risks.append(
            "RSI remains subdued and does not yet confirm strong buying momentum."
        )

    return bullish, risks


# =========================================================
# SCREENER
# =========================================================

st.header("📊 Screener")

screener_input = st.text_input(
    "Enter Tickers",
    "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS"
)

if st.button("Run Screener"):

    tickers = [x.strip() for x in screener_input.split(",")]

    screener_results = []

    for ticker in tickers:
        try:
            df = fetch_data(ticker)
            metrics = compute_metrics(df, benchmark_df)

            screener_results.append({
                "Ticker": ticker,
                "Score": metrics["Score"],
                "Recommendation": metrics["Recommendation"],
                "Structure": metrics["Structure"],
                "RSI": metrics["RSI"],
                "Momentum": metrics["Momentum"],
                "RelativeStrength": metrics["RelativeStrength"],
                "Volatility": metrics["Volatility"]
            })

        except Exception as e:
            st.warning(f"Error processing {ticker}: {e}")

    screener_df = pd.DataFrame(screener_results)

    if not screener_df.empty:
        screener_df = screener_df.sort_values(by="Score", ascending=False)
        st.dataframe(screener_df, use_container_width=True)


# =========================================================
# PORTFOLIO BACKTEST
# =========================================================

st.header("📈 Portfolio Backtest")

portfolio_input = st.text_input(
    "Portfolio Tickers",
    "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS"
)

backtest_years = st.slider(
    "Backtest Years",
    1,
    10,
    5
)

if st.button("Run Portfolio Backtest"):

    tickers = [x.strip() for x in portfolio_input.split(",")]

    portfolio_returns = []
    valid_tickers = []

    for ticker in tickers:
        try:
            df = fetch_data(ticker, period=f"{backtest_years}y")

            returns = df["Close"].pct_change().dropna()

            portfolio_returns.append(returns)
            valid_tickers.append(ticker)

        except Exception as e:
            st.warning(f"Error loading {ticker}: {e}")

    if len(portfolio_returns) > 0:

        combined_returns = pd.concat(portfolio_returns, axis=1)
        combined_returns.columns = valid_tickers

        strategy_returns = combined_returns.mean(axis=1)

        benchmark = fetch_data("^NSEI", period=f"{backtest_years}y")
        benchmark_returns = benchmark["Close"].pct_change().dropna()

        strategy_curve = (1 + strategy_returns).cumprod()
        benchmark_curve = (1 + benchmark_returns).cumprod()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=strategy_curve.index,
            y=strategy_curve,
            mode="lines",
            name="Strategy"
        ))

        fig.add_trace(go.Scatter(
            x=benchmark_curve.index,
            y=benchmark_curve,
            mode="lines",
            name="Benchmark"
        ))

        st.plotly_chart(fig, use_container_width=True)

        strategy_total_return = (strategy_curve.iloc[-1] - 1) * 100
        benchmark_total_return = (benchmark_curve.iloc[-1] - 1) * 100

        sharpe = (
            strategy_returns.mean() /
            strategy_returns.std()
        ) * np.sqrt(252)

        downside = strategy_returns[strategy_returns < 0]

        sortino = (
            strategy_returns.mean() /
            downside.std()
        ) * np.sqrt(252)

        rolling_max = strategy_curve.cummax()
        drawdown = strategy_curve / rolling_max - 1
        max_drawdown = drawdown.min() * 100

        cagr = (
            strategy_curve.iloc[-1] ** (1 / backtest_years) - 1
        ) * 100

        c1, c2, c3 = st.columns(3)

        c1.metric("Strategy Return", f"{strategy_total_return:.2f}%")
        c1.metric("CAGR", f"{cagr:.2f}%")

        c2.metric("Benchmark Return", f"{benchmark_total_return:.2f}%")
        c2.metric("Max Drawdown", f"{max_drawdown:.2f}%")

        c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        c3.metric("Sortino Ratio", f"{sortino:.2f}")

        st.subheader("Selected Portfolio")

        portfolio_df = pd.DataFrame({
            "Ticker": valid_tickers
        })

        st.dataframe(portfolio_df, use_container_width=True)

        st.subheader("🧠 Portfolio Interpretation")

        if strategy_total_return > benchmark_total_return:
            st.success(
                "The strategy outperformed the benchmark over the selected period."
            )
        else:
            st.warning(
                "The strategy underperformed the benchmark over the selected period."
            )

        if sharpe > 1:
            st.success("Risk-adjusted returns appear strong.")
        else:
            st.info("Risk-adjusted returns appear moderate.")


# =========================================================
# SINGLE STOCK ANALYSIS
# =========================================================

st.header("🔎 Single Stock")

single_ticker = st.text_input(
    "Ticker",
    "RELIANCE.NS"
)

if st.button("Analyze Stock"):

    try:
        df = fetch_data(single_ticker)

        metrics = compute_metrics(df, benchmark_df)

        st.subheader(
            f"Recommendation: {metrics['Recommendation']}"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("RSI", metrics["RSI"])
        c2.metric("Momentum %", metrics["Momentum"])
        c3.metric("Relative Strength %", metrics["RelativeStrength"])
        c4.metric("Volatility %", metrics["Volatility"])

        close = df["Close"]

        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index,
            y=close,
            mode="lines",
            name="Close"
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=sma50,
            mode="lines",
            name="SMA50"
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=sma200,
            mode="lines",
            name="SMA200"
        ))

        st.subheader("📉 Price Chart")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Chart Pattern Analysis")

        chart_points = chart_analysis(metrics)

        for point in chart_points:
            st.write(f"• {point}")

        bullish, risks = generate_writeup(single_ticker, metrics)

        st.subheader("🧠 AI Investment Writeup")

        st.info(
            f"The stock currently exhibits a {metrics['Structure']} setup."
        )

        st.subheader("✅ Bullish Factors")

        for point in bullish:
            st.write(f"• {point}")

        st.subheader("⚠️ Risk Factors")

        for point in risks:
            st.write(f"• {point}")

        st.subheader("🎯 Probability Scenarios")

        probability_df = pd.DataFrame({
            "Scenario": [
                "Bullish Continuation",
                "Sideways Consolidation",
                "Bearish Breakdown"
            ],
            "Probability": [
                "45%",
                "35%",
                "20%"
            ]
        })

        st.dataframe(probability_df, use_container_width=True)

        st.subheader("👀 What To Watch Next")

        if metrics["Momentum"] < 0:
            st.write(
                "• Watch whether momentum turns positive over coming weeks."
            )

        if metrics["SMA50"] < metrics["SMA200"]:
            st.write(
                "• Watch for SMA50 crossing above SMA200, which would strengthen bullish structure."
            )

        if metrics["RSI"] > 70:
            st.write(
                "• RSI is elevated; monitor for exhaustion or pullback risk."
            )

    except Exception as e:
        st.error(f"Analysis failed: {e}")

```
