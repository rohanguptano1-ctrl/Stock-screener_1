import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import time

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BharatTrack V20",
    layout="wide",
    page_icon="🚀"
)

st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: #1a1d2e;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #a0aabf !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] { font-size: 13px !important; font-weight: 600 !important; }
    .risk-badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:700; }
    .risk-low  { background:#0d3b1e; color:#2ecc71; }
    .risk-med  { background:#3b2a0d; color:#f39c12; }
    .risk-high { background:#3b0d0d; color:#e74c3c; }
    .canslim-bar  { height:8px; border-radius:4px; background:#2e3250; margin-top:4px; margin-bottom:8px; }
    .canslim-fill { height:8px; border-radius:4px; background:linear-gradient(90deg,#1a9e5c,#27ae60); }
    .pass-badge { display:inline-block; padding:2px 8px; border-radius:4px;
                  background:#0d3b1e; color:#2ecc71; font-size:12px; font-weight:700; }
    .fail-badge { display:inline-block; padding:2px 8px; border-radius:4px;
                  background:#3b0d0d; color:#e74c3c; font-size:12px; font-weight:700; }
    .regime-bull { background:#0d3b1e; color:#2ecc71; padding:6px 14px;
                   border-radius:8px; font-weight:700; display:inline-block; }
    .regime-bear { background:#3b0d0d; color:#e74c3c; padding:6px 14px;
                   border-radius:8px; font-weight:700; display:inline-block; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 BharatTrack V20")
st.caption("CANSLIM · Minervini Trend Template · Momentum Factor · IBD RS Rank · Basket Screener · Market Regime Filter")

# =========================================================
# INDEX BASKETS
# =========================================================

BASKETS = {
    "Nifty 50": [
        "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
        "BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BEL","BPCL",
        "BHARTIARTL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB",
        "DRREDDY","EICHERMOT","GRASIM","HCLTECH","HDFCBANK",
        "HDFCLIFE","HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK",
        "INDUSINDBK","INFY","ITC","JIOFIN","JSWSTEEL",
        "KOTAKBANK","LT","M&M","MARUTI","NESTLEIND",
        "NTPC","ONGC","POWERGRID","RELIANCE","SBILIFE",
        "SHRIRAMFIN","SBIN","SUNPHARMA","TATACONSUM","TATAMOTORS",
        "TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO"
    ],
    "Bank Nifty": [
        "AXISBANK","BANDHANBNK","FEDERALBNK","HDFCBANK","ICICIBANK",
        "IDFCFIRSTB","INDUSINDBK","KOTAKBANK","PNB","SBIN",
        "AUBANK","BANKBARODA"
    ],
    "Nifty IT": [
        "INFY","TCS","HCLTECH","WIPRO","TECHM",
        "LTIM","MPHASIS","COFORGE","PERSISTENT","OFSS"
    ],
    "Nifty Pharma": [
        "SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","AUROPHARMA",
        "LUPIN","TORNTPHARM","ALKEM","IPCA","BIOCON"
    ],
    "Nifty Midcap (Sample)": [
        "ABCAPITAL","ABFRL","APLAPOLLO","ASTRAL","BALKRISIND",
        "CANFINHOME","CHAMBLFERT","CROMPTON","DEEPAKNTR","DIXON",
        "GLENMARK","GODREJPROP","GRANULES","GSPL","HFCL",
        "IDFC","INDIAMART","JINDALSTEL","JUBLFOOD","KANSAINER",
        "MAXHEALTH","MGL","NATIONALUM","NAVINFLUOR","NLCINDIA",
        "OBEROIRLTY","PAGEIND","PIIND","POLYCAB","RAIN"
    ],
    "Custom": []
}


# =========================================================
# SECTOR MAP
# =========================================================

SECTOR_MAP = {
    # Banking
    "HDFCBANK":"Banking","ICICIBANK":"Banking","KOTAKBANK":"Banking",
    "SBIN":"Banking","AXISBANK":"Banking","INDUSINDBK":"Banking",
    "BANDHANBNK":"Banking","FEDERALBNK":"Banking","PNB":"Banking",
    "AUBANK":"Banking","BANKBARODA":"Banking","IDFCFIRSTB":"Banking",
    # NBFC / Insurance / Finance
    "BAJFINANCE":"NBFC","BAJAJFINSV":"NBFC","SHRIRAMFIN":"NBFC",
    "JIOFIN":"NBFC","SBILIFE":"Insurance","HDFCLIFE":"Insurance",
    # IT
    "TCS":"IT","INFY":"IT","HCLTECH":"IT","WIPRO":"IT","TECHM":"IT",
    "LTIM":"IT","MPHASIS":"IT","COFORGE":"IT","PERSISTENT":"IT","OFSS":"IT",
    # Oil & Gas / Energy
    "RELIANCE":"Oil & Gas","ONGC":"Oil & Gas","BPCL":"Oil & Gas",
    "NTPC":"Power","POWERGRID":"Power","COALINDIA":"Mining","NLCINDIA":"Power",
    # Metals
    "TATASTEEL":"Metals","JSWSTEEL":"Metals","HINDALCO":"Metals","NATIONALUM":"Metals",
    # Auto
    "MARUTI":"Auto","TATAMOTORS":"Auto","M&M":"Auto",
    "BAJAJ-AUTO":"Auto","HEROMOTOCO":"Auto","EICHERMOT":"Auto",
    # Pharma / Healthcare
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma",
    "DIVISLAB":"Pharma","APOLLOHOSP":"Healthcare",
    # FMCG
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG",
    "BRITANNIA":"FMCG","TATACONSUM":"FMCG",
    # Cement / Infra
    "ULTRACEMCO":"Cement","GRASIM":"Cement","LT":"Infrastructure",
    # Consumer / Retail
    "TITAN":"Consumer","TRENT":"Retail","ASIANPAINT":"Consumer",
    # Telecom
    "BHARTIARTL":"Telecom",
    # Defence
    "BEL":"Defence","MAZDOCK":"Defence","HAL":"Defence",
    # Ports / Conglomerate
    "ADANIPORTS":"Ports","ADANIENT":"Conglomerate",
    # Consumer / Jewellery
    "KALYANKJIL":"Consumer","TRENT":"Retail","DMART":"Retail",
    "ASIANPAINT":"Consumer","TITAN":"Consumer","VBL":"FMCG",
    # Auto ancillary / Tyres
    "APOLLOTYRE":"Auto","BALKRISIND":"Auto","MRF":"Auto",
    # Real Estate
    "ANANTRAJ":"Real Estate","GODREJPROP":"Real Estate",
    "OBEROIRLTY":"Real Estate","PRESTIGE":"Real Estate",
    # Capital Goods / Engineering
    "TECHNOE":"Capital Goods","KRNHEAT":"Capital Goods",
    "POLYCAB":"Capital Goods","ABB":"Capital Goods",
    # Financial Services
    "CRISIL":"Financial Services","MUTHOOTFIN":"Financial Services",
    "CHOLAFIN":"Financial Services","CANFINHOME":"Financial Services",
    # Healthcare IT / Services
    "INDEGENE":"Healthcare IT","SAGILITY":"Healthcare IT",
    "MAXHEALTH":"Healthcare","FORTIS":"Healthcare",
    # Telecom / Infra
    "RAILTEL":"Infrastructure","HFCL":"Telecom",
    "INDIAMART":"IT","DIXON":"Electronics",
    # Chemicals / Materials
    "STYRENIX":"Chemicals","DEEPAKNTR":"Chemicals",
    "NAVINFLUOR":"Chemicals","PIIND":"Chemicals",
    # Small / Other
    "TAKE":"IT","NLCINDIA":"Power","HAL":"Defence",
    "MAZDOCK":"Defence","BEL":"Defence",
}

def get_sector(ticker):
    t = ticker.upper().replace(".NS","").replace(".BO","")
    return SECTOR_MAP.get(t, "Other")

# =========================================================
# TICKER UTILS
# =========================================================

def normalize_ticker(ticker):
    ticker = ticker.strip().upper()
    if "." not in ticker:
        ticker = ticker + ".NS"
    return ticker

# Manual cache — only stores SUCCESSFUL fetches, never empty results
# This prevents Streamlit's cache from locking in failed yfinance responses
_fetch_cache = {}

def fetch_data(ticker, period="5y"):
    ticker = normalize_ticker(ticker)
    cache_key = f"{ticker}_{period}"

    # Return cached result only if it was a successful fetch
    if cache_key in _fetch_cache:
        return _fetch_cache[cache_key]

    for attempt in range(3):  # retry up to 3 times with backoff
        try:
            if attempt > 0:
                time.sleep(3 * attempt)  # 3s then 6s between retries
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.strip().title() for c in df.columns]
            if "Close" not in df.columns and "Adj Close" in df.columns:
                df.rename(columns={"Adj Close": "Close"}, inplace=True)
            needed = [c for c in ["Close", "Open", "High", "Low", "Volume"] if c in df.columns]
            if not needed or "Close" not in needed:
                continue
            df = df[needed].copy()
            df.dropna(subset=["Close"], inplace=True)
            if len(df) > 0:
                _fetch_cache[cache_key] = df  # only cache successes
                return df
        except Exception:
            continue
    return pd.DataFrame()  # never cached — always retried next time

# =========================================================
# BENCHMARK + MARKET REGIME
# =========================================================

_benchmark_cache = {}

def fetch_benchmark():
    if "benchmark" in _benchmark_cache:
        return _benchmark_cache["benchmark"]
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
                result = df[["Close"]].dropna()
                _benchmark_cache["benchmark"] = result
                return result
        except Exception:
            continue
    return pd.DataFrame()

def get_market_regime(benchmark_df):
    """
    Returns 'Bull' if Nifty is above its 200DMA, 'Bear' otherwise.
    In bear regime, all buy signals are downgraded by one tier.
    """
    if benchmark_df.empty or len(benchmark_df) < 200:
        return "Bull"
    close = benchmark_df["Close"]
    sma200 = close.rolling(200).mean().iloc[-1]
    return "Bull" if float(close.iloc[-1]) > float(sma200) else "Bear"

benchmark_df = fetch_benchmark()

if benchmark_df.empty or "Close" not in benchmark_df.columns:
    st.error("❌ Could not fetch benchmark data. This is a temporary Yahoo Finance issue — please reload.")
    st.stop()

market_regime = get_market_regime(benchmark_df)

# =========================================================
# CORE METRICS
# =========================================================

def compute_metrics(df, benchmark_df):
    close = df["Close"].copy()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    delta = close.diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + rs))
    momentum_3m = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100
    bench_return_3m = ((benchmark_df["Close"].iloc[-1] / benchmark_df["Close"].iloc[-63]) - 1) * 100
    relative_strength = momentum_3m - bench_return_3m
    volatility = close.pct_change().std() * np.sqrt(252) * 100

    score = 0
    if float(close.iloc[-1]) > float(sma200.iloc[-1]): score += 30
    if float(sma50.iloc[-1]) > float(sma200.iloc[-1]): score += 25
    if float(rsi.iloc[-1]) > 55: score += 20
    if momentum_3m > 0: score += 15
    if relative_strength > 0: score += 10

    structure = (
        "Bullish Structure" if float(close.iloc[-1]) > float(sma200.iloc[-1]) and float(sma50.iloc[-1]) > float(sma200.iloc[-1])
        else "Early Accumulation" if float(close.iloc[-1]) > float(sma200.iloc[-1])
        else "Bearish Structure"
    )

    return {
        "Price": round(float(close.iloc[-1]), 2),
        "RSI": round(float(rsi.iloc[-1]), 2),
        "Momentum3M": round(float(momentum_3m), 2),
        "RelativeStrength": round(float(relative_strength), 2),
        "Volatility": round(float(volatility), 2),
        "SMA50": round(float(sma50.iloc[-1]), 2),
        "SMA200": round(float(sma200.iloc[-1]), 2),
        "Structure": structure,
        "Score": int(score)
    }

# =========================================================
# CANSLIM SCORE
# =========================================================

def compute_canslim_score(df):
    close = df["Close"]
    scores = {}

    scores["C_CurrentMomentum"] = min(max(
        (close.iloc[-1] / close.iloc[-63] - 1) * 100 / 20 * 20, 0), 20) if len(close) >= 63 else 0

    sma200 = close.rolling(200).mean()
    scores["A_AnnualTrend"] = min(max(
        (close.iloc[-1] / sma200.iloc[-1] - 1) * 100 / 10 * 15, 0), 15) if len(sma200.dropna()) > 0 else 0

    if len(close) >= 252:
        proximity = (close.iloc[-1] / close.rolling(252).max().iloc[-1]) * 100
        scores["N_NewHighs"] = 15 if proximity >= 95 else (8 if proximity >= 85 else 0)
    else:
        scores["N_NewHighs"] = 0

    if "Volume" in df.columns and len(df) >= 50:
        v20 = df["Volume"].rolling(20).mean().iloc[-1]
        v50 = df["Volume"].rolling(50).mean().iloc[-1]
        scores["S_SupplyDemand"] = 15 if v20 > v50 * 1.1 else (8 if v20 > v50 else 3)
    else:
        scores["S_SupplyDemand"] = 7

    sma50 = close.rolling(50).mean()
    scores["L_Leader"] = 15 if (len(sma50.dropna()) > 0 and len(sma200.dropna()) > 0
                                 and sma50.iloc[-1] > sma200.iloc[-1]) else 0

    if len(sma200.dropna()) >= 20:
        slope = (sma200.iloc[-1] / sma200.iloc[-20] - 1) * 100
        above = close.iloc[-1] > sma200.iloc[-1]
        scores["I_Institutional"] = 10 if (above and slope > 0) else (5 if above else 0)
    else:
        scores["I_Institutional"] = 0

    scores["M_MarketDirection"] = 10
    return scores, sum(scores.values())

# =========================================================
# MINERVINI TREND TEMPLATE
# =========================================================

def compute_minervini_score(df):
    """
    Mark Minervini's 8 Trend Template conditions.
    All 8 must pass for a Stage 2 uptrend confirmation.
    Score = conditions passed (0-8), shown as pct (0-100).
    """
    close = df["Close"]
    conditions = {}

    if len(close) < 252:
        return {}, 0, 0

    sma50  = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()

    price  = float(close.iloc[-1])
    s50    = float(sma50.iloc[-1])
    s150   = float(sma150.iloc[-1])
    s200   = float(sma200.iloc[-1])

    high_52w = float(close.rolling(252).max().iloc[-1])
    low_52w  = float(close.rolling(252).min().iloc[-1])

    # 200DMA slope (trending up for at least 1 month = 20 trading days)
    sma200_1m_ago = float(sma200.iloc[-20]) if len(sma200.dropna()) >= 20 else s200

    conditions["1. Price > 200 DMA"]           = price > s200
    conditions["2. 200 DMA trending up (1M)"]  = s200 > sma200_1m_ago
    conditions["3. 150 DMA > 200 DMA"]         = s150 > s200
    conditions["4. Price > 150 DMA"]           = price > s150
    conditions["5. Price > 50 DMA"]            = price > s50
    conditions["6. 50 DMA > 150 & 200 DMA"]   = (s50 > s150) and (s50 > s200)
    conditions["7. Price ≥ 30% above 52W low"] = price >= low_52w * 1.30
    conditions["8. Price within 25% of 52W high"] = price >= high_52w * 0.75

    passed = sum(conditions.values())
    score_pct = round(passed / 8 * 100)
    return conditions, passed, score_pct

# =========================================================
# MOMENTUM FACTOR (Extended)
# =========================================================

def compute_momentum_score(df):
    """
    Academic momentum: 12-1 month (skip last month to avoid reversal).
    Also computes raw 3M, 6M, 12M for display.
    Score 0-100 based on composite momentum rank signal.
    """
    close = df["Close"]
    result = {}

    result["Momentum3M"]  = round((close.iloc[-1] / close.iloc[-63]  - 1) * 100, 2) if len(close) >= 63  else None
    result["Momentum6M"]  = round((close.iloc[-1] / close.iloc[-126] - 1) * 100, 2) if len(close) >= 126 else None
    result["Momentum12M"] = round((close.iloc[-1] / close.iloc[-252] - 1) * 100, 2) if len(close) >= 252 else None

    # 12-1 momentum (skip last 21 days — standard academic approach)
    if len(close) >= 273:
        mom_12_1 = (close.iloc[-21] / close.iloc[-252] - 1) * 100
        result["Momentum12_1"] = round(mom_12_1, 2)
    else:
        result["Momentum12_1"] = result["Momentum12M"]

    # Trend consistency: % of last 20 weeks that closed up
    if len(close) >= 100:
        weekly = close.resample("W").last()
        recent = weekly.iloc[-20:]
        up_weeks = (recent.diff() > 0).sum()
        result["TrendConsistency"] = round(up_weeks / len(recent) * 100)
    else:
        result["TrendConsistency"] = 50

    # Composite momentum score (0-100)
    scores = []
    if result["Momentum3M"]  is not None: scores.append(min(max(result["Momentum3M"]  / 30 * 40, 0), 40))
    if result["Momentum6M"]  is not None: scores.append(min(max(result["Momentum6M"]  / 40 * 35, 0), 35))
    if result["Momentum12M"] is not None: scores.append(min(max(result["Momentum12M"] / 60 * 25, 0), 25))
    result["MomentumScore"] = round(sum(scores)) if scores else 0

    return result

# =========================================================
# IBD-STYLE RS RANK (within screened universe)
# =========================================================

def compute_rs_ranks(universe_returns_12m):
    """
    Ranks all stocks in the universe by 12M return.
    Returns a dict: ticker -> RS Rank (1-99, higher is better).
    Mimics IBD RS Rating.
    """
    if not universe_returns_12m:
        return {}
    series = pd.Series(universe_returns_12m)
    ranks = series.rank(pct=True) * 99
    return ranks.round().astype(int).to_dict()

# =========================================================
# MASTER COMBINED SCORE
# =========================================================

def compute_master_score(base_score, canslim_total, minervini_pct,
                          momentum_score, rs_rank, market_regime):
    """
    Blended score across all frameworks.
    Market regime filter: Bear market downgrades all scores by 20%.
    """
    raw = (
        base_score        * 0.20 +   # Technical base (SMA, RSI)
        canslim_total     * 0.25 +   # CANSLIM
        minervini_pct     * 0.25 +   # Minervini conditions
        momentum_score    * 0.20 +   # Momentum factor
        rs_rank           * 0.10     # RS Rank within universe
    )
    # Market regime penalty
    if market_regime == "Bear":
        raw = raw * 0.80

    raw = round(min(raw, 100))

    if raw >= 80:   rec = "🟢 Strong Buy"
    elif raw >= 65: rec = "🟢 Buy"
    elif raw >= 45: rec = "🟠 Watch"
    else:           rec = "🔴 Avoid"

    return raw, rec

# =========================================================
# RISK MANAGEMENT
# =========================================================

def compute_risk_metrics(df, capital=100000, risk_per_trade_pct=1.5):
    close = df["Close"]
    if "High" in df.columns and "Low" in df.columns:
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - close.shift()).abs(),
            (df["Low"]  - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
    else:
        atr = close.pct_change().std() * close.iloc[-1] * np.sqrt(14)

    price = float(close.iloc[-1])
    stop  = price - 2 * float(atr)
    stop_pct = (price - stop) / price * 100
    risk_amt = capital * risk_per_trade_pct / 100
    shares = int(risk_amt / (price - stop)) if (price - stop) > 0 else 0
    pos_val = shares * price
    vol = close.pct_change().std() * np.sqrt(252) * 100
    dd  = (close / close.cummax() - 1) * 100

    return {
        "ATR": round(float(atr), 2),
        "CurrentPrice": round(price, 2),
        "StopLoss": round(stop, 2),
        "StopLossPct": round(stop_pct, 2),
        "Target1R": round(price + 2 * float(atr), 2),
        "Target2R": round(price + 4 * float(atr), 2),
        "Target3R": round(price + 6 * float(atr), 2),
        "RiskAmount": round(risk_amt, 0),
        "Shares": shares,
        "PositionValue": round(pos_val, 0),
        "PositionPct": round(pos_val / capital * 100, 1),
        "VolatilityAnnual": round(float(vol), 1),
        "RiskTier": "Low" if vol < 20 else ("Medium" if vol < 35 else "High"),
        "MaxDrawdown": round(float(dd.min()), 2),
        "CurrentDrawdown": round(float(dd.iloc[-1]), 2)
    }

# =========================================================
# PROBABILITY ENGINE
# =========================================================

def compute_probability_scenarios(master_score, rsi, momentum, relative_strength):
    rsi_sig = max(0, min(100, (rsi - 30) / 40 * 100)) if rsi else 50
    mom_sig = 70 if momentum > 5 else (50 if momentum > 0 else 30)
    rs_sig  = 70 if relative_strength > 3 else (50 if relative_strength > 0 else 30)
    composite = master_score * 0.50 + rsi_sig * 0.20 + mom_sig * 0.20 + rs_sig * 0.10
    bullish  = round(max(10, min(75, composite * 0.70)), 1)
    bearish  = round(max(5,  min(60, (100 - composite) * 0.55)), 1)
    sideways = round(100 - bullish - bearish, 1)
    if sideways < 5:
        sideways = 5
        bearish = round(100 - bullish - sideways, 1)
    return {"Bullish Continuation": bullish, "Sideways Consolidation": sideways, "Bearish Breakdown": bearish}

# =========================================================
# RESEARCH NOTE (rule-based)
# =========================================================

def generate_research_note(ticker, metrics, canslim_scores, canslim_total,
                            minervini_passed, momentum_data, master_score,
                            master_rec, risk_metrics, scenarios, market_regime):
    bull_prob = scenarios["Bullish Continuation"]
    bear_prob = scenarios["Bearish Breakdown"]

    # Thesis
    if master_score >= 80:
        thesis = (f"{ticker} presents a high-conviction multi-framework setup. "
                  f"Master Score {master_score}/100 with {minervini_passed}/8 Minervini conditions passed. "
                  f"Bullish probability: {bull_prob:.0f}%.")
    elif master_score >= 65:
        thesis = (f"{ticker} shows a constructive setup (Master Score {master_score}/100) "
                  f"with {minervini_passed}/8 Minervini conditions confirmed. "
                  f"Selective entry with strict stops warranted.")
    elif master_score >= 45:
        thesis = (f"{ticker} is in a neutral zone (Master Score {master_score}/100). "
                  f"Only {minervini_passed}/8 Minervini conditions pass. Wait for more confluence.")
    else:
        thesis = (f"{ticker} is in a weak setup (Master Score {master_score}/100). "
                  f"Only {minervini_passed}/8 Minervini conditions pass and bear probability is {bear_prob:.0f}%. "
                  f"Capital better deployed elsewhere.")

    if market_regime == "Bear":
        thesis += " ⚠️ Nifty is below its 200DMA — market regime is bearish, all signals downgraded."

    # Technical
    sma_line = (
        "Golden cross intact — price above both SMA50 and SMA200."
        if metrics["Price"] > metrics["SMA50"] > metrics["SMA200"]
        else "Early recovery — price above SMA200 but SMA50 still lagging."
        if metrics["Price"] > metrics["SMA200"]
        else "Bearish structure — price below SMA200."
    )
    mom3 = momentum_data.get("Momentum3M")
    mom6 = momentum_data.get("Momentum6M")
    mom12 = momentum_data.get("Momentum12M")
    consist = momentum_data.get("TrendConsistency", 50)
    mom_line = (f"Momentum profile: 3M {mom3:+.1f}% / "
                f"6M {f'{mom6:+.1f}%' if mom6 is not None else 'N/A'} / "
                f"12M {f'{mom12:+.1f}%' if mom12 is not None else 'N/A'}. "
                f"Trend consistency: {consist}% of recent weeks closed up.")

    # CANSLIM
    strengths, gaps = [], []
    if canslim_scores.get("C_CurrentMomentum", 0) >= 10: strengths.append("strong current momentum (C)")
    else: gaps.append("weak current momentum (C)")
    if canslim_scores.get("N_NewHighs", 0) >= 15: strengths.append("near 52W highs (N)")
    elif canslim_scores.get("N_NewHighs", 0) == 0: gaps.append("far from 52W highs (N)")
    if canslim_scores.get("S_SupplyDemand", 0) >= 15: strengths.append("volume confirming price (S)")
    else: gaps.append("volume not confirming (S)")
    if canslim_scores.get("L_Leader", 0) >= 15: strengths.append("golden cross structure (L)")
    else: gaps.append("lagging MA structure (L)")
    canslim_text = (f"CANSLIM {int(canslim_total)}/100. "
                    + (f"Strengths: {', '.join(strengths)}. " if strengths else "No strong CANSLIM signals. ")
                    + (f"Gaps: {', '.join(gaps)}." if gaps else ""))

    # Risk
    sizing = (f"ATR stop at ₹{risk_metrics['StopLoss']} ({risk_metrics['StopLossPct']:.1f}% risk). "
              f"2R target ₹{risk_metrics['Target2R']}. "
              f"Suggested: {risk_metrics['Shares']} shares "
              f"(₹{risk_metrics['PositionValue']:,.0f}, {risk_metrics['PositionPct']:.1f}% of capital).")

    # Conviction
    if master_score >= 80 and minervini_passed >= 7:
        conviction, reason = "HIGH", "Master Score and Minervini both confirm a clean Stage 2 setup."
    elif master_score >= 65 and minervini_passed >= 5:
        conviction, reason = "MEDIUM", "Good setup but not all frameworks fully aligned — start with half position."
    else:
        conviction, reason = "LOW", "Too many framework conditions unmet to justify full commitment."

    if market_regime == "Bear":
        conviction = "LOW" if conviction == "MEDIUM" else conviction
        reason += " Bear market regime active — size down."

    return f"""
**1. INVESTMENT THESIS**
{thesis}

**2. TECHNICAL SETUP**
{sma_line} RSI at {metrics['RSI']:.0f}. {mom_line} Relative strength vs Nifty: {metrics['RelativeStrength']:+.1f}% (3M).

**3. CANSLIM ASSESSMENT**
{canslim_text}

**4. RISK ASSESSMENT**
{risk_metrics['RiskTier']} volatility ({risk_metrics['VolatilityAnnual']:.0f}% annualised). Historical max drawdown: {risk_metrics['MaxDrawdown']:.0f}%. {sizing}

**5. CONVICTION LEVEL: {conviction}**
{reason}
"""


# =========================================================
# PORTFOLIO FUNCTIONS
# =========================================================

def parse_portfolio_input(text):
    """
    Parse lines of: TICKER, SHARES, BUY_PRICE
    Returns list of dicts.
    """
    holdings = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            holdings.append({
                "ticker": parts[0].upper(),
                "shares": float(parts[1]),
                "buy_price": float(parts[2])
            })
        except ValueError:
            continue
    return holdings


def compute_action_signal(master_score, metrics, risk_metrics,
                           pnl_pct, market_regime):
    """
    Returns (signal, color, reason) based on framework scores + P&L context.
    """
    price   = metrics["Price"]
    sma50   = metrics["SMA50"]
    sma200  = metrics["SMA200"]
    rsi     = metrics["RSI"]
    current_dd = risk_metrics["CurrentDrawdown"]
    stop    = risk_metrics["StopLoss"]

    # EXIT conditions
    if price < stop:
        return "🔴 EXIT", "#e74c3c", "Price below ATR stop loss — exit to protect capital"
    if master_score < 30:
        return "🔴 EXIT", "#e74c3c", f"Master Score {master_score}/100 — framework thesis broken"
    if price < sma200 and master_score < 40 and pnl_pct < -15:
        return "🔴 EXIT", "#e74c3c", "Below 200DMA, weak score, and meaningful loss — cut position"

    # TRIM conditions
    if pnl_pct > 50 and master_score < 50:
        return "🟠 TRIM", "#f39c12", f"Up {pnl_pct:.0f}% but momentum fading (Score {master_score}) — book partial profits"
    if rsi > 78:
        return "🟠 TRIM", "#f39c12", f"RSI {rsi:.0f} — extremely overbought, reduce exposure"
    if master_score < 45 and pnl_pct > 20:
        return "🟠 TRIM", "#f39c12", "Score weakening while in profit — trim to reduce risk"
    if current_dd < -20 and master_score < 50:
        return "🟠 TRIM", "#f39c12", f"Down {current_dd:.0f}% from highs with weak score — reduce position"

    # ADD conditions
    if (master_score >= 65 and price > sma50 > sma200
            and rsi < 72 and metrics["Momentum3M"] > 0
            and market_regime == "Bull"):
        return "🟢 ADD", "#2ecc71", f"Strong setup (Score {master_score}) in bull market — consider adding"
    if (master_score >= 70 and price > sma200 and rsi < 68):
        return "🟢 ADD", "#2ecc71", f"High conviction score ({master_score}) with room to run"

    # HOLD
    if master_score >= 45:
        return "🔵 HOLD", "#3498db", f"Score {master_score}/100 — thesis intact, stay the course"

    return "🟠 WATCH", "#f39c12", f"Score {master_score}/100 — monitor closely, conditions mixed"

# =========================================================
# MARKET REGIME BANNER
# =========================================================

regime_color = "regime-bull" if market_regime == "Bull" else "regime-bear"
regime_label = "🟢 Bull Market — Nifty above 200DMA" if market_regime == "Bull" else "🔴 Bear Market — Nifty below 200DMA · All signals downgraded"
st.markdown(f'<span class="{regime_color}">{regime_label}</span>', unsafe_allow_html=True)
st.markdown("")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 Basket Screener", "📈 Portfolio Backtest", "🔎 Single Stock Deep Dive", "💼 Portfolio Intelligence"])

# =========================================================
# TAB 1 — BASKET SCREENER
# =========================================================

with tab1:
    st.header("📊 Basket Screener")

    col_basket, col_framework = st.columns(2)

    with col_basket:
        basket_choice = st.selectbox("Select Index Basket", list(BASKETS.keys()))

    with col_framework:
        framework_choice = st.selectbox(
            "Scoring Framework",
            ["Master Score (All Frameworks)", "CANSLIM Only", "Minervini Only", "Momentum Only"]
        )

    if basket_choice == "Custom":
        custom_input = st.text_input("Enter Custom Tickers (comma-separated)", "COALINDIA,BEL,TRENT,MAZDOCK,NLCINDIA")
        tickers_to_screen = [t.strip() for t in custom_input.split(",") if t.strip()]
    else:
        tickers_to_screen = BASKETS[basket_choice]

    st.caption(f"{len(tickers_to_screen)} stocks in {basket_choice} · Framework: {framework_choice}")

    min_master = st.slider("Minimum Master Score to show", 0, 100, 0, 5)

    if st.button("🚀 Run Screener", type="primary", key="run_screener"):
        screener_rows = []
        skipped = []
        returns_12m = {}

        progress_bar = st.progress(0)
        status = st.empty()

        # Pass 1: fetch all data and compute 12M returns for RS Rank
        all_data = {}
        for i, ticker in enumerate(tickers_to_screen):
            status.text(f"Fetching {ticker}... ({i+1}/{len(tickers_to_screen)})")
            if i > 0:
                time.sleep(0.2)
            df = fetch_data(ticker)
            if df.empty or "Close" not in df.columns or len(df) < 252:
                skipped.append(f"{ticker} — insufficient data")
                progress_bar.progress((i + 1) / len(tickers_to_screen))
                continue
            all_data[ticker] = df
            if len(df["Close"]) >= 252:
                returns_12m[ticker] = float(df["Close"].iloc[-1] / df["Close"].iloc[-252] - 1) * 100
            progress_bar.progress((i + 1) / len(tickers_to_screen))

        # Compute RS Ranks across universe
        rs_ranks = compute_rs_ranks(returns_12m)

        # Pass 2: full scoring
        for ticker, df in all_data.items():
            try:
                metrics      = compute_metrics(df, benchmark_df)
                canslim_s, canslim_t = compute_canslim_score(df)
                _, min_passed, min_pct = compute_minervini_score(df)
                mom_data     = compute_momentum_score(df)
                rs_rank      = rs_ranks.get(ticker, 50)
                master, rec  = compute_master_score(
                    metrics["Score"], canslim_t, min_pct,
                    mom_data["MomentumScore"], rs_rank, market_regime
                )

                if master < min_master:
                    continue

                # Choose display score by framework
                if framework_choice == "CANSLIM Only":
                    display_score, display_rec = int(canslim_t), (
                        "🟢 Strong Buy" if canslim_t >= 80 else
                        "🟢 Buy" if canslim_t >= 60 else
                        "🟠 Watch" if canslim_t >= 40 else "🔴 Avoid"
                    )
                elif framework_choice == "Minervini Only":
                    display_score, display_rec = min_pct, (
                        "🟢 Strong Buy" if min_pct == 100 else
                        "🟢 Buy" if min_pct >= 75 else
                        "🟠 Watch" if min_pct >= 50 else "🔴 Avoid"
                    )
                elif framework_choice == "Momentum Only":
                    display_score, display_rec = mom_data["MomentumScore"], (
                        "🟢 Strong Buy" if mom_data["MomentumScore"] >= 75 else
                        "🟢 Buy" if mom_data["MomentumScore"] >= 55 else
                        "🟠 Watch" if mom_data["MomentumScore"] >= 35 else "🔴 Avoid"
                    )
                else:
                    display_score, display_rec = master, rec

                screener_rows.append({
                    "Ticker":        ticker,
                    "Master Score":  master,
                    "CANSLIM":       int(canslim_t),
                    "Minervini/8":   min_passed,
                    "Momentum Score":mom_data["MomentumScore"],
                    "RS Rank":       rs_rank,
                    "Rec":           display_rec,
                    "Structure":     metrics["Structure"],
                    "RSI":           metrics["RSI"],
                    "3M Mom%":       metrics["Momentum3M"],
                    "6M Mom%":       mom_data.get("Momentum6M"),
                    "12M Mom%":      mom_data.get("Momentum12M"),
                    "Trend Consist%":mom_data.get("TrendConsistency"),
                    "Rel Strength%": metrics["RelativeStrength"],
                    "Volatility%":   metrics["Volatility"],
                    "RiskTier":      compute_risk_metrics(df)["RiskTier"],
                })
            except Exception as e:
                skipped.append(f"{ticker} — error: {str(e)[:40]}")

        status.empty()
        progress_bar.empty()

        if screener_rows:
            sort_col = "Master Score" if "Master" in framework_choice else (
                "CANSLIM" if "CANSLIM" in framework_choice else (
                "Minervini/8" if "Minervini" in framework_choice else "Momentum Score"
            ))
            result_df = pd.DataFrame(screener_rows).sort_values(by=sort_col, ascending=False)
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(result_df)} of {len(tickers_to_screen)} stocks")
        else:
            st.warning("No stocks passed the minimum score filter.")

        if skipped:
            with st.expander(f"⚠️ {len(skipped)} ticker(s) skipped"):
                for s in skipped:
                    st.write("•", s)

# =========================================================
# TAB 2 — PORTFOLIO BACKTEST
# =========================================================

with tab2:
    st.header("📈 Portfolio Backtest")

    portfolio_input = st.text_input("Portfolio Tickers", "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK")
    col_y, col_c = st.columns(2)
    with col_y:
        years = st.slider("Backtest Years", 1, 10, 5)
    with col_c:
        capital = st.number_input("Portfolio Capital (₹)", 10000, 10000000, 100000, 10000, format="%d")
    risk_per_trade = st.slider("Risk Per Trade (% of capital)", 0.5, 5.0, 1.5, 0.25)

    if st.button("Run Portfolio Backtest", key="run_backtest"):
        tickers = [t.strip() for t in portfolio_input.split(",") if t.strip()]
        benchmark_returns = benchmark_df["Close"].pct_change().fillna(0)
        port_returns, sel_tickers, port_risk = [], [], []

        for ticker in tickers:
            df = fetch_data(ticker, period=f"{years}y")
            if df.empty or "Close" not in df.columns or len(df) < 250:
                continue
            sel_tickers.append(ticker)
            port_returns.append(df["Close"].pct_change().fillna(0))
            port_risk.append({"Ticker": ticker,
                               **compute_risk_metrics(df, capital/len(tickers), risk_per_trade)})

        if port_returns:
            aligned = pd.concat(port_returns, axis=1)
            aligned.columns = sel_tickers
            strat_ret = aligned.mean(axis=1)
            strat_curve = (1 + strat_ret).cumprod()
            bench_curve = (1 + benchmark_returns.reindex(strat_curve.index).fillna(0)).cumprod()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strat_curve.index, y=strat_curve, name="Strategy",
                                      line=dict(color="#2ecc71", width=2)))
            fig.add_trace(go.Scatter(x=bench_curve.index, y=bench_curve, name="Nifty 50",
                                      line=dict(color="#3498db", width=2, dash="dash")))
            fig.update_layout(template="plotly_dark", title="Portfolio vs Benchmark (Equal Weight)",
                              yaxis_title="Cumulative Return",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

            sr = (strat_curve.iloc[-1] - 1) * 100
            br = (bench_curve.iloc[-1] - 1) * 100
            cagr = (strat_curve.iloc[-1] ** (1/years) - 1) * 100
            rf = 0.065 / 252
            excess = strat_ret - rf
            sharpe  = (excess.mean() / strat_ret.std()) * np.sqrt(252)
            down    = strat_ret[strat_ret < 0]
            sortino = (excess.mean() / down.std()) * np.sqrt(252) if down.std() != 0 else 0
            max_dd  = ((strat_curve / strat_curve.cummax() - 1).min()) * 100
            calmar  = cagr / abs(max_dd) if max_dd != 0 else 0

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Strategy Return", f"{sr:.1f}%", f"{sr-br:+.1f}% vs Nifty")
            c2.metric("CAGR", f"{cagr:.1f}%")
            c3.metric("Sharpe Ratio", f"{sharpe:.2f}", help="6.5% risk-free rate")
            c4.metric("Sortino Ratio", f"{sortino:.2f}")

            c5,c6,c7,c8 = st.columns(4)
            c5.metric("Max Drawdown", f"{max_dd:.1f}%")
            c6.metric("Calmar Ratio", f"{calmar:.2f}")
            c7.metric("Benchmark Return", f"{br:.1f}%")
            c8.metric("Alpha", f"{sr-br:.1f}%")

            st.subheader("⚠️ Per-Stock Risk Profile")
            if port_risk:
                rdf = pd.DataFrame(port_risk)[
                    ["Ticker","RiskTier","StopLoss","StopLossPct","Target2R",
                     "VolatilityAnnual","MaxDrawdown","Shares","PositionValue"]
                ]
                rdf.columns = ["Ticker","Risk Tier","Stop ₹","Stop %",
                                "2R Target ₹","Vol %","Max DD %","Shares","Position ₹"]
                st.dataframe(rdf, use_container_width=True, hide_index=True)

            st.subheader("🧠 Interpretation")
            if sr > br:
                st.success(f"✅ Outperformed Nifty by {sr-br:.1f}% over {years} years.")
            else:
                st.warning(f"⚠️ Underperformed Nifty by {br-sr:.1f}% over {years} years.")
            if sharpe > 1.5:
                st.info(f"📊 Sharpe {sharpe:.2f} — strong risk-adjusted returns.")
            elif sharpe > 1:
                st.info(f"📊 Sharpe {sharpe:.2f} — acceptable. Tighten stock selection to improve.")
            else:
                st.warning(f"📊 Sharpe {sharpe:.2f} — below acceptable. Review entry criteria.")
        else:
            st.error("No valid portfolio data fetched.")

# =========================================================
# TAB 3 — SINGLE STOCK DEEP DIVE
# =========================================================

with tab3:
    st.header("🔎 Single Stock Deep Dive")

    col_t, col_c2 = st.columns([2, 1])
    with col_t:
        single_ticker = st.text_input("Ticker (e.g. RELIANCE, TCS, NLCINDIA)", "RELIANCE")
    with col_c2:
        analysis_capital = st.number_input("Capital for Sizing (₹)", 10000, 10000000,
                                            500000, 10000, format="%d")
    analysis_risk = st.slider("Risk Per Trade %", 0.5, 5.0, 1.5, 0.25, key="single_risk")

    if st.button("Analyze Stock", type="primary", key="analyze"):
        df = fetch_data(single_ticker)

        if df.empty or "Close" not in df.columns or len(df) < 252:
            st.error("Not enough data — need at least 252 trading days (1 year). Check the ticker name.")
        else:
            with st.spinner("Running all frameworks..."):
                metrics          = compute_metrics(df, benchmark_df)
                canslim_s, canslim_t = compute_canslim_score(df)
                min_conds, min_passed, min_pct = compute_minervini_score(df)
                mom_data         = compute_momentum_score(df)
                risk_metrics     = compute_risk_metrics(df, analysis_capital, analysis_risk)
                # RS rank vs Nifty 50 universe for context
                rs_rank_val      = 50  # placeholder (needs full universe)
                master, master_rec = compute_master_score(
                    metrics["Score"], canslim_t, min_pct,
                    mom_data["MomentumScore"], rs_rank_val, market_regime
                )
                scenarios = compute_probability_scenarios(
                    master, metrics["RSI"], metrics["Momentum3M"], metrics["RelativeStrength"]
                )

            # Header
            st.subheader(f"{master_rec}  |  Master Score: {master}/100  |  CANSLIM: {int(canslim_t)}/100  |  Minervini: {min_passed}/8")

            # Key metrics
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Price", f"₹{metrics['Price']}")
            c2.metric("RSI (14)", f"{metrics['RSI']:.1f}")
            c3.metric("3M Momentum", f"{metrics['Momentum3M']:+.1f}%")
            c4.metric("Rel. Strength", f"{metrics['RelativeStrength']:+.1f}%")
            c5.metric("Annual Vol", f"{metrics['Volatility']:.1f}%")

            c6,c7,c8 = st.columns(3)
            c6.metric("6M Momentum", f"{mom_data.get('Momentum6M', 0):+.1f}%" if mom_data.get('Momentum6M') else "N/A")
            c7.metric("12M Momentum", f"{mom_data.get('Momentum12M', 0):+.1f}%" if mom_data.get('Momentum12M') else "N/A")
            c8.metric("Trend Consistency", f"{mom_data.get('TrendConsistency', 0)}% up-weeks")

            # Price Chart
            st.subheader("📉 Price Chart")
            close  = df["Close"]
            sma50  = close.rolling(50).mean()
            sma150 = close.rolling(150).mean()
            sma200 = close.rolling(200).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=close,  name="Price",   line=dict(color="#ffffff", width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=sma50,  name="SMA 50",  line=dict(color="#f39c12", width=1.2, dash="dash")))
            fig.add_trace(go.Scatter(x=df.index, y=sma150, name="SMA 150", line=dict(color="#9b59b6", width=1.0, dash="dot")))
            fig.add_trace(go.Scatter(x=df.index, y=sma200, name="SMA 200", line=dict(color="#e74c3c", width=1.2, dash="dot")))

            # 52W high/low reference
            high_52w = close.rolling(252).max().iloc[-1]
            low_52w  = close.rolling(252).min().iloc[-1]
            fig.add_hline(y=float(high_52w), line_color="#2ecc71", line_dash="dot", line_width=1,
                          annotation_text=f"52W High ₹{high_52w:.0f}", annotation_position="top right")
            fig.add_hline(y=float(low_52w), line_color="#e74c3c", line_dash="dot", line_width=1,
                          annotation_text=f"52W Low ₹{low_52w:.0f}", annotation_position="bottom right")
            fig.add_hline(y=risk_metrics["StopLoss"], line_color="#e74c3c", line_dash="dash", line_width=1.5,
                          annotation_text=f"Stop ₹{risk_metrics['StopLoss']}", annotation_position="bottom left")
            fig.add_hline(y=risk_metrics["Target2R"], line_color="#2ecc71", line_dash="dash", line_width=1.5,
                          annotation_text=f"2R ₹{risk_metrics['Target2R']}", annotation_position="top left")

            fig.update_layout(template="plotly_dark", height=450,
                              margin=dict(l=0, r=0, t=30, b=0),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

            # Three-column: CANSLIM | Minervini | Risk
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("### 📐 CANSLIM")
                for key, (label, max_s) in {
                    "C_CurrentMomentum": ("C – Current Momentum", 20),
                    "A_AnnualTrend":     ("A – Annual Trend", 15),
                    "N_NewHighs":        ("N – New Highs", 15),
                    "S_SupplyDemand":    ("S – Supply/Demand", 15),
                    "L_Leader":          ("L – Leader", 15),
                    "I_Institutional":   ("I – Institutional", 10),
                    "M_MarketDirection": ("M – Market", 10),
                }.items():
                    val = canslim_s.get(key, 0)
                    pct = int(val / max_s * 100)
                    st.markdown(f"**{label}** — {val:.0f}/{max_s}")
                    st.markdown(f'<div class="canslim-bar"><div class="canslim-fill" style="width:{pct}%"></div></div>',
                                unsafe_allow_html=True)
                st.markdown(f"**Total: {int(canslim_t)}/100**")

            with col_b:
                st.markdown("### 📏 Minervini Template")
                for cond, passed in min_conds.items():
                    badge = '<span class="pass-badge">PASS</span>' if passed else '<span class="fail-badge">FAIL</span>'
                    st.markdown(f"{badge} {cond}", unsafe_allow_html=True)
                st.markdown(f"**{min_passed}/8 conditions passed ({min_pct}%)**")
                if min_passed == 8:
                    st.success("✅ Full Stage 2 Uptrend Confirmed")
                elif min_passed >= 6:
                    st.info(f"⚡ Strong setup — {8 - min_passed} condition(s) short of full confirmation")
                elif min_passed >= 4:
                    st.warning(f"⚠️ Partial setup — {8 - min_passed} conditions still unmet")
                else:
                    st.error(f"❌ Weak setup — only {min_passed}/8 conditions met")

            with col_c:
                st.markdown("### ⚠️ Risk Management")
                risk_color = {"Low":"risk-low","Medium":"risk-med","High":"risk-high"}[risk_metrics["RiskTier"]]
                st.markdown(f'Risk Tier: <span class="risk-badge {risk_color}">{risk_metrics["RiskTier"]}</span>',
                            unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({
                    "Level":    ["Current Price","Stop (2×ATR)","1R Target","2R Target","3R Target"],
                    "Price ₹":  [f"₹{risk_metrics['CurrentPrice']}",
                                  f"₹{risk_metrics['StopLoss']} ({risk_metrics['StopLossPct']:.1f}%)",
                                  f"₹{risk_metrics['Target1R']}",
                                  f"₹{risk_metrics['Target2R']}",
                                  f"₹{risk_metrics['Target3R']}"]
                }), use_container_width=True, hide_index=True)
                st.markdown(f"""
- Capital: ₹{analysis_capital:,.0f}
- Risk: {analysis_risk}% = ₹{risk_metrics['RiskAmount']:,.0f}
- ATR: ₹{risk_metrics['ATR']}
- Shares: **{risk_metrics['Shares']}**
- Position: **₹{risk_metrics['PositionValue']:,.0f}** ({risk_metrics['PositionPct']:.1f}%)
- Max DD: {risk_metrics['MaxDrawdown']:.1f}% · Current DD: {risk_metrics['CurrentDrawdown']:.1f}%
""")

            # Probability Scenarios
            st.markdown("### 🎯 Probability Scenarios")
            st.caption("Derived from Master Score, RSI, Momentum, and Relative Strength")
            bull_p = scenarios["Bullish Continuation"]
            side_p = scenarios["Sideways Consolidation"]
            bear_p = scenarios["Bearish Breakdown"]
            s1,s2,s3 = st.columns(3)
            s1.metric("🟢 Bullish", f"{bull_p:.0f}%")
            s2.metric("🟡 Sideways", f"{side_p:.0f}%")
            s3.metric("🔴 Bearish", f"{bear_p:.0f}%")
            fig_p = go.Figure(go.Bar(
                x=[bull_p, side_p, bear_p], y=["Bullish","Sideways","Bearish"],
                orientation="h", marker_color=["#2ecc71","#f39c12","#e74c3c"],
                text=[f"{v:.0f}%" for v in [bull_p, side_p, bear_p]], textposition="inside"
            ))
            fig_p.update_layout(template="plotly_dark", height=150,
                                 margin=dict(l=0,r=0,t=5,b=5),
                                 showlegend=False, xaxis=dict(range=[0,100]))
            st.plotly_chart(fig_p, use_container_width=True)

            # Bullish / Risk factors
            col_bull, col_risk = st.columns(2)
            with col_bull:
                st.markdown("### ✅ Bullish Factors")
                bl = []
                if metrics["Price"] > metrics["SMA200"]: bl.append("Price above 200DMA")
                if metrics["RelativeStrength"] > 0: bl.append(f"Outperforming Nifty by {metrics['RelativeStrength']:.1f}%")
                if metrics["RSI"] > 55: bl.append(f"RSI at {metrics['RSI']:.0f} — healthy")
                if min_passed >= 6: bl.append(f"Minervini {min_passed}/8 conditions — strong structure")
                if canslim_s.get("N_NewHighs", 0) >= 15: bl.append("Near 52-week highs")
                if canslim_s.get("S_SupplyDemand", 0) >= 15: bl.append("Volume confirming price strength")
                if mom_data.get("TrendConsistency", 0) >= 60: bl.append(f"High trend consistency ({mom_data['TrendConsistency']}% up-weeks)")
                if not bl: bl.append("No major bullish signals currently")
                for b in bl: st.write("•", b)

            with col_risk:
                st.markdown("### ⚠️ Risk Factors")
                rl = []
                if market_regime == "Bear": rl.append("Bear market regime — Nifty below 200DMA")
                if metrics["SMA50"] < metrics["SMA200"]: rl.append("Death cross — SMA50 below SMA200")
                if metrics["Momentum3M"] < 0: rl.append(f"Negative 3M momentum ({metrics['Momentum3M']:.1f}%)")
                if risk_metrics["RiskTier"] == "High": rl.append(f"High volatility ({risk_metrics['VolatilityAnnual']:.0f}%)")
                if risk_metrics["MaxDrawdown"] < -40: rl.append(f"Historical max drawdown {risk_metrics['MaxDrawdown']:.0f}%")
                if metrics["RSI"] > 75: rl.append(f"RSI {metrics['RSI']:.0f} — overbought")
                if min_passed < 5: rl.append(f"Only {min_passed}/8 Minervini conditions met")
                if not rl: rl.append("No major risk factors currently")
                for r in rl: st.write("•", r)

            # What to watch
            st.markdown("### 👀 What To Watch Next")
            wl = []
            if metrics["Momentum3M"] < 0: wl.append("Watch for 3M momentum to turn positive")
            if metrics["SMA50"] < metrics["SMA200"]: wl.append("Watch for golden cross: SMA50 crossing above SMA200")
            if metrics["RSI"] > 70: wl.append(f"RSI at {metrics['RSI']:.0f} — watch for pullback before adding")
            if risk_metrics["CurrentDrawdown"] < -15: wl.append(f"Currently {risk_metrics['CurrentDrawdown']:.0f}% from highs — watch for base")
            if min_passed < 8: wl.append(f"Minervini {min_passed}/8 — watch for remaining conditions to confirm")
            if canslim_s.get("S_SupplyDemand",0) < 8: wl.append("Volume weak — watch for volume expansion on rallies")
            if not wl: wl.append("All systems healthy — monitor for continuation")
            for w in wl: st.write("•", w)

            # Research Note
            st.markdown("### 🧠 Research Note")
            st.caption("Rule-based · Multi-framework synthesis · No API cost")
            note = generate_research_note(
                single_ticker, metrics, canslim_s, canslim_t,
                min_passed, mom_data, master, master_rec,
                risk_metrics, scenarios, market_regime
            )
            st.markdown(note)


# =========================================================
# NAME → NSE TICKER MAP (for Groww Excel parsing)
# =========================================================

NAME_TO_TICKER = {
    # Rohan's portfolio
    "ANANT RAJ LIMITED":              "ANANTRAJ",
    "APOLLO TYRES LTD":              "APOLLOTYRE",
    "AVENUE SUPERMARTS LIMITED":     "DMART",
    "CRISIL LTD":                    "CRISIL",
    "HINDUSTAN AERONAUTICS LTD":     "HAL",
    "ICICI BANK LTD.":               "ICICIBANK",
    "ICICI BANK LTD":                "ICICIBANK",
    "INDEGENE LIMITED":              "INDEGENE",
    "JIO FIN SERVICES LTD":          "JIOFIN",
    "JIO FINANCIAL SERVICES LTD":    "JIOFIN",
    "KALYAN JEWELLERS IND LTD":      "KALYANKJIL",
    "KRN HEAT EXCHANGE N REF L":     "KRNHEAT",
    "NTPC LTD":                      "NTPC",
    "NTPC LIMITED":                  "NTPC",
    "RAILTEL CORP OF IND LTD":       "RAILTEL",
    "RELIANCE INDUSTRIES LTD":       "RELIANCE",
    "RELIANCE INDUSTRIES LIMITED":   "RELIANCE",
    "SAGILITY LIMITED":              "SAGILITY",
    "STYRENIX PERFORMANCE LTD":      "STYRENIX",
    "TAKE SOLUTIONS LTD":            "TAKE",
    "TATA MOTORS PASS VEH LTD":      "TATAMOTORS",  # Groww shows PV demerger shares as this
    "TATA MOTORS LIMITED":           "TATAMOTORS",
    "TATA MOTORS LTD":               "TATAMOTORS",
    "TECHNO ELEC & ENG CO. LTD":     "TECHNOE",
    "TECHNO ELEC & ENG CO LTD":      "TECHNOE",
    "VARUN BEVERAGES LIMITED":       "VBL",
    # Nifty 50 / common stocks
    "TATA CONSULTANCY SERVICES LTD": "TCS",
    "INFOSYS LTD":                   "INFY",
    "HDFC BANK LTD":                 "HDFCBANK",
    "HINDUSTAN UNILEVER LTD":        "HINDUNILVR",
    "ITC LTD":                       "ITC",
    "AXIS BANK LTD":                 "AXISBANK",
    "KOTAK MAHINDRA BANK LTD":       "KOTAKBANK",
    "STATE BANK OF INDIA":           "SBIN",
    "BAJAJ FINANCE LTD":             "BAJFINANCE",
    "BHARTI AIRTEL LTD":             "BHARTIARTL",
    "WIPRO LTD":                     "WIPRO",
    "HCL TECHNOLOGIES LTD":          "HCLTECH",
    "MARUTI SUZUKI INDIA LTD":       "MARUTI",
    "SUN PHARMACEUTICAL INDS. LTD":  "SUNPHARMA",
    "TITAN COMPANY LTD":             "TITAN",
    "ULTRATECH CEMENT LTD":          "ULTRACEMCO",
    "LARSEN & TOUBRO LTD":           "LT",
    "POWER GRID CORP OF INDIA LTD":  "POWERGRID",
    "COAL INDIA LTD":                "COALINDIA",
    "BHARAT ELECTRONICS LTD":        "BEL",
    "TATA STEEL LTD":                "TATASTEEL",
    "HINDALCO INDUSTRIES LTD":       "HINDALCO",
    "JSW STEEL LTD":                 "JSWSTEEL",
    "ADANI PORTS AND SEZ LTD":       "ADANIPORTS",
    "ADANI ENTERPRISES LTD":         "ADANIENT",
    "MAHINDRA & MAHINDRA LTD":       "M&M",
    "BAJAJ AUTO LTD":                "BAJAJ-AUTO",
    "HERO MOTOCORP LTD":             "HEROMOTOCO",
    "EICHER MOTORS LTD":             "EICHERMOT",
    "DR. REDDY'S LABORATORIES LTD": "DRREDDY",
    "CIPLA LTD":                     "CIPLA",
    "DIVIS LABORATORIES LTD":        "DIVISLAB",
    "APOLLO HOSPITALS ENTERPRISE LTD":"APOLLOHOSP",
    "NESTLE INDIA LTD":              "NESTLEIND",
    "BRITANNIA INDUSTRIES LTD":      "BRITANNIA",
    "TATA CONSUMER PRODUCTS LTD":    "TATACONSUM",
    "GRASIM INDUSTRIES LTD":         "GRASIM",
    "BAJAJ FINSERV LTD":             "BAJAJFINSV",
    "SBI LIFE INSURANCE CO. LTD":    "SBILIFE",
    "HDFC LIFE INSURANCE CO. LTD":   "HDFCLIFE",
    "SHRIRAM FINANCE LTD":           "SHRIRAMFIN",
    "TECH MAHINDRA LTD":             "TECHM",
    "ASIAN PAINTS LTD":              "ASIANPAINT",
    "TRENT LIMITED":                 "TRENT",
    "BPCL LTD":                      "BPCL",
    "BHARAT PETROLEUM CORP LTD":     "BPCL",
    "OIL AND NATURAL GAS CORP LTD":  "ONGC",
    "INDUSIND BANK LTD":             "INDUSINDBK",
}

def parse_groww_excel(file) -> tuple:
    """
    Parse Groww Stocks Holdings Statement Excel.
    Actual structure (0-indexed):
      Row 0:  Name, Rohan Gupta
      Row 6:  Invested Value, <amount>
      Row 7:  Closing Value,  <amount>
      Row 8:  Unrealised P&L, <amount>
      Row 10: Header row (Stock Name, ISIN, Quantity, ...)
      Row 11+: Data rows
    Equity ISINs start with INE. Non-equity filtered out.
    """
    # Try sheet by name first, then by index — handles edge cases on cloud
    df = None
    read_error = None
    for sheet in ["Sheet1", 0]:
        try:
            df = pd.read_excel(file, sheet_name=sheet, header=None, engine="openpyxl")
            if df is not None and not df.empty:
                break
        except Exception as e:
            read_error = str(e)
            continue
    if df is None or df.empty:
        return [], {}, [f"Could not read Excel file: {read_error}"]

    # Scan for summary values by label (robust to row shifts)
    summary = {}
    for i in range(min(15, len(df))):
        label = str(df.iloc[i, 0]).strip().lower()
        val   = df.iloc[i, 1]
        if "invested" in label:
            try: summary["invested"] = float(val)
            except: pass
        elif "closing" in label:
            try: summary["value"] = float(val)
            except: pass
        elif "unrealised" in label or "p&l" in label:
            try: summary["pnl"] = float(val)
            except: pass

    # Find the header row by scanning for "Stock Name"
    header_row = None
    for i in range(len(df)):
        cell = str(df.iloc[i, 0]).strip().lower()
        if "stock name" in cell:
            header_row = i
            break

    if header_row is None:
        return [], summary, ["Could not find data header row in file"]

    # Data starts immediately after header row
    data_rows = df.iloc[header_row + 1:].copy()
    data_rows.columns = range(len(data_rows.columns))

    holdings, skipped = [], []

    for _, row in data_rows.iterrows():
        name      = str(row[0]).strip() if pd.notna(row[0]) else ""
        isin      = str(row[1]).strip() if pd.notna(row[1]) else ""
        qty       = row[2]
        avg_price = row[3]

        if not name or name.lower() == "nan" or not isin or isin.lower() == "nan":
            continue

        # Filter: only equities (ISIN starts with INE)
        if not isin.startswith("INE"):
            skipped.append(f"{name} — skipped (non-equity: {isin[:8]})")
            continue

        try:
            qty_f = float(qty)
            if qty_f <= 0:
                skipped.append(f"{name} — skipped (zero quantity)")
                continue
        except (ValueError, TypeError):
            continue

        try:
            avg_f = float(avg_price)
            if avg_f <= 0:
                skipped.append(f"{name} — skipped (zero buy price)")
                continue
        except (ValueError, TypeError):
            skipped.append(f"{name} — skipped (invalid buy price)")
            continue

        # Map name to NSE ticker
        name_upper = name.upper().strip()
        ticker = NAME_TO_TICKER.get(name_upper)
        if not ticker:
            for k, v in NAME_TO_TICKER.items():
                if k in name_upper or name_upper in k:
                    ticker = v
                    break
        if not ticker:
            ticker = name_upper.split()[0]
            skipped.append(f"{name} — ticker guessed as '{ticker}' (verify manually)")

        holdings.append({
            "ticker":    ticker,
            "name":      name,
            "shares":    qty_f,
            "buy_price": avg_f,
        })

    return holdings, summary, skipped


# =========================================================
# TAB 4 — PORTFOLIO INTELLIGENCE
# =========================================================

with tab4:
    st.header("💼 Portfolio Intelligence")

    # ── How to download instructions ──────────────────────
    with st.expander("📥 How to download your Holdings Statement from Groww", expanded=False):
        st.markdown("""
**On Groww App (Mobile):**
1. Tap the **Profile icon** at the bottom right
2. Tap **Reports**
3. Under **Holdings**, tap **Stock Holdings Statement**
4. Select **Excel** format
5. Tap **Download**

**On Groww Website (groww.in):**
1. Click your **Profile** icon (top right)
2. Click **Reports**
3. Under **Holdings**, click **Stock Holdings Statement**
4. Choose **Excel** format and click **Download**

👉 **Direct link:** [groww.in/reports](https://groww.in/reports)

The downloaded file will be named like:
`Stocks_Holdings_Statement_XXXXXXXXXX_DD-MM-YYYY.xlsx`

Upload that file below — no manual entry needed.
""")

    st.markdown("---")

    # ── Upload section ─────────────────────────────────────
    st.markdown("### 📂 Upload Groww Holdings Statement")

    col_up, col_cap = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload your Excel (.xlsx) from Groww",
            type=["xlsx"],
            help="Stocks Holdings Statement from Groww — always the same format"
        )
    with col_cap:
        new_investment = st.number_input(
            "New Investment Amount (₹)",
            min_value=0, max_value=10000000,
            value=0, step=5000, format="%d",
            help="Additional capital you plan to deploy — leave 0 if just reviewing current portfolio"
        )

    # ── Watchlist ──────────────────────────────────────────
    watchlist_input = st.text_input(
        "📋 Watchlist — stocks you're tracking but haven't bought",
        "NLCINDIA, MAZDOCK, TITAN, BAJAJ-AUTO",
        help="Comma-separated NSE tickers"
    )

    if uploaded_file is not None:
        # Parse the Excel
        with st.spinner("Parsing your Groww holdings..."):
            holdings, groww_summary, parse_skipped = parse_groww_excel(uploaded_file)

        if not holdings:
            if parse_skipped:
                st.error("Could not extract equity holdings. Details:")
                for s in parse_skipped:
                    st.write("•", s)
            else:
                st.error("Could not extract any equity holdings. The file may be empty or in an unexpected format.")
        else:
            # Set total capital from Excel + new investment
            portfolio_current_value = groww_summary.get("value", 0)
            total_capital_port = portfolio_current_value + new_investment

            # Show what was parsed
            st.success(f"✅ Found **{len(holdings)} equity positions** from your Groww statement")

            gs1, gs2, gs3, gs4 = st.columns(4)
            gs1.metric("Invested", f"₹{groww_summary.get('invested',0):,.0f}")
            gs2.metric("Current Value", f"₹{portfolio_current_value:,.0f}")
            gs3.metric("Unrealised P&L", f"₹{groww_summary.get('pnl',0):+,.0f}",
                       f"{groww_summary.get('pnl',0)/max(groww_summary.get('invested',1),1)*100:+.1f}%")
            gs4.metric("Total Capital", f"₹{total_capital_port:,.0f}",
                       f"+₹{new_investment:,.0f} new" if new_investment > 0 else "no new investment")

            if parse_skipped:
                with st.expander(f"ℹ️ {len(parse_skipped)} items filtered/skipped"):
                    for s in parse_skipped:
                        st.write("•", s)

            # Show parsed holdings table
            with st.expander("📋 Parsed holdings (verify tickers before analysing)"):
                preview_df = pd.DataFrame([{
                    "NSE Ticker": h["ticker"],
                    "Company":    h["name"],
                    "Shares":     int(h["shares"]),
                    "Avg Buy ₹":  h["buy_price"],
                } for h in holdings])
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
                st.caption("If any ticker looks wrong, let me know and I'll fix the mapping.")

            if st.button("🔍 Analyse Portfolio", type="primary", key="analyse_excel"):

                port_data = []
                errors    = []
                prog = st.progress(0)
                stat = st.empty()

                for i, h in enumerate(holdings):
                    ticker = h["ticker"]
                    stat.text(f"Analysing {ticker} ({h['name'][:30]})...")
                    if i > 0:
                        time.sleep(0.3)  # small delay to avoid Yahoo Finance rate limiting
                    df = fetch_data(ticker)

                    MIN_DAYS = 100
                    limited_data = len(df) < 252 if not df.empty else True
                    if df.empty or "Close" not in df.columns or len(df) < MIN_DAYS:
                        if len(df) > 0:
                            errors.append(f"{ticker} ({h['name'][:30]}) — only {len(df)} days of data (need {MIN_DAYS}+)")
                        else:
                            errors.append(f"{ticker} ({h['name'][:30]}) — no data returned (try again later)")
                        prog.progress((i+1)/len(holdings))
                        continue

                    try:
                        metrics       = compute_metrics(df, benchmark_df)
                        canslim_s, ct = compute_canslim_score(df)
                        _, mp, mpct   = compute_minervini_score(df)
                        mom_data      = compute_momentum_score(df)
                        risk_m        = compute_risk_metrics(df)
                        master, mrec  = compute_master_score(
                            metrics["Score"], ct, mpct,
                            mom_data["MomentumScore"], 50, market_regime
                        )

                        current_price = metrics["Price"]
                        invested      = h["shares"] * h["buy_price"]
                        current_val   = h["shares"] * current_price
                        pnl_inr       = current_val - invested
                        pnl_pct       = (current_price / h["buy_price"] - 1) * 100

                        signal, sig_color, sig_reason = compute_action_signal(
                            master, metrics, risk_m, pnl_pct, market_regime
                        )

                        port_data.append({
                            "Ticker":       ticker,
                            "Company":      h["name"],
                            "Sector":       get_sector(ticker),
                            "LimitedData":  limited_data,
                            "Shares":       int(h["shares"]),
                            "Buy ₹":        h["buy_price"],
                            "Current ₹":    current_price,
                            "Invested ₹":   round(invested, 0),
                            "Value ₹":      round(current_val, 0),
                            "P&L ₹":        round(pnl_inr, 0),
                            "P&L %":        round(pnl_pct, 1),
                            "Master Score": master,
                            "CANSLIM":      int(ct),
                            "Minervini/8":  mp,
                            "Signal":       signal,
                            "Signal Reason":sig_reason,
                            "RSI":          metrics["RSI"],
                            "3M Mom%":      metrics["Momentum3M"],
                            "Stop ₹":       risk_m["StopLoss"],
                            "2R Target ₹":  risk_m["Target2R"],
                            "Port %":       round(current_val / total_capital_port * 100, 1),
                        })
                    except Exception as e:
                        errors.append(f"{ticker} — {str(e)[:50]}")

                    prog.progress((i+1)/len(holdings))

                prog.empty()
                stat.empty()

                if not port_data:
                    st.error("Could not analyse any positions. Check ticker mappings.")
                else:
                    # ── Portfolio Summary ──────────────────────────────
                    total_invested = sum(r["Invested ₹"] for r in port_data)
                    total_value    = sum(r["Value ₹"]    for r in port_data)
                    total_pnl      = total_value - total_invested
                    total_pnl_pct  = (total_value / total_invested - 1) * 100 if total_invested > 0 else 0
                    avg_score      = round(sum(r["Master Score"] for r in port_data) / len(port_data))

                    st.markdown("### 📊 Portfolio Summary")
                    s1,s2,s3,s4,s5 = st.columns(5)
                    s1.metric("Total Invested",   f"₹{total_invested:,.0f}")
                    s2.metric("Current Value",    f"₹{total_value:,.0f}")
                    s3.metric("Total P&L",        f"₹{total_pnl:+,.0f}", f"{total_pnl_pct:+.1f}%")
                    s4.metric("Avg Master Score", f"{avg_score}/100")
                    s5.metric("Positions",        f"{len(port_data)}")

                    if market_regime == "Bear":
                        st.warning(
                            "⚠️ Bear Market Regime — Nifty is below its 200-day moving average. "
                            "All Master Scores are automatically reduced by 20% to reflect higher risk. "
                            "A stock scoring 40 here would score ~50 in a bull market. "
                            "Use Exit/Trim signals with extra caution — don't panic sell quality holdings."
                        )

                    # ── Action Signal Summary ──────────────────────────
                    st.markdown("### 🎯 Action Signals")
                    exits = [r for r in port_data if "EXIT"  in r["Signal"]]
                    trims = [r for r in port_data if "TRIM"  in r["Signal"] or "WATCH" in r["Signal"]]
                    adds  = [r for r in port_data if "ADD"   in r["Signal"]]
                    holds = [r for r in port_data if "HOLD"  in r["Signal"]]

                    ac1,ac2,ac3,ac4 = st.columns(4)
                    ac1.metric("🔴 Exit",  len(exits))
                    ac2.metric("🟠 Trim",  len(trims))
                    ac3.metric("🔵 Hold",  len(holds))
                    ac4.metric("🟢 Add",   len(adds))

                    if exits:
                        st.error("**Exit signals:** " + ", ".join(r["Ticker"] for r in exits))
                    if trims:
                        st.warning("**Trim/Watch:** " + ", ".join(r["Ticker"] for r in trims))
                    if adds:
                        st.success("**Add signals:** " + ", ".join(r["Ticker"] for r in adds))

                    # ── Sector Breakdown ───────────────────────────────
                    st.markdown("### 🏭 Sector Breakdown")
                    sectors = {}
                    for r in port_data:
                        s = r["Sector"]
                        if s not in sectors:
                            sectors[s] = {"tickers":[], "invested":0, "value":0, "pnl":0}
                        sectors[s]["tickers"].append(r["Ticker"])
                        sectors[s]["invested"] += r["Invested ₹"]
                        sectors[s]["value"]    += r["Value ₹"]
                        sectors[s]["pnl"]      += r["P&L ₹"]

                    sector_rows = []
                    for s, d in sorted(sectors.items(), key=lambda x: -x[1]["value"]):
                        pp = (d["value"]/d["invested"]-1)*100 if d["invested"]>0 else 0
                        cp = d["value"]/total_capital_port*100
                        sector_rows.append({
                            "Sector":    s,
                            "Stocks":    ", ".join(d["tickers"]),
                            "Invested":  f"₹{d['invested']:,.0f}",
                            "Value":     f"₹{d['value']:,.0f}",
                            "P&L":       f"₹{d['pnl']:+,.0f}",
                            "P&L %":     f"{pp:+.1f}%",
                            "Port %":    f"{cp:.1f}%",
                            "⚠️ Conc":   "⚠️ High" if cp > 30 else ("🟡 Med" if cp > 15 else "✅ OK")
                        })
                    st.dataframe(pd.DataFrame(sector_rows), use_container_width=True, hide_index=True)

                    # ── P&L Chart ──────────────────────────────────────
                    st.markdown("### 📊 P&L by Position")
                    sorted_data = sorted(port_data, key=lambda x: x["P&L ₹"])
                    fig_pnl = go.Figure(go.Bar(
                        x=[r["Ticker"] for r in sorted_data],
                        y=[r["P&L ₹"]  for r in sorted_data],
                        marker_color=["#2ecc71" if r["P&L ₹"]>=0 else "#e74c3c" for r in sorted_data],
                        text=[f"₹{r['P&L ₹']:+,.0f}\n({r['P&L %']:+.1f}%)" for r in sorted_data],
                        textposition="outside"
                    ))
                    fig_pnl.update_layout(
                        template="plotly_dark", height=340,
                        margin=dict(l=0,r=0,t=10,b=0),
                        yaxis_title="P&L (₹)", showlegend=False
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)

                    # ── Per-Position Detail ────────────────────────────
                    st.markdown("### 📋 Position Detail (by Sector)")
                    for sector in sorted(set(r["Sector"] for r in port_data)):
                        positions = [r for r in port_data if r["Sector"] == sector]
                        if not positions:
                            continue
                        st.markdown(f"#### 🏷️ {sector}")
                        for r in positions:
                            icon = "🟢" if r["P&L %"] >= 0 else "🔴"
                            data_flag = " · ⚠️ Limited Data" if r.get("LimitedData") else ""
                            with st.expander(
                                f"{r['Ticker']}  ·  {r['Signal']}  ·  "
                                f"{icon} {r['P&L %']:+.1f}%  (₹{r['P&L ₹']:+,.0f})  ·  "
                                f"Score {r['Master Score']}/100{data_flag}"
                            ):
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.markdown("**Position**")
                                    st.markdown(f"""
- Company: {r["Company"]}
- Shares: {r["Shares"]}
- Buy price: ₹{r["Buy ₹"]}
- Current: ₹{r["Current ₹"]}
- Invested: ₹{r["Invested ₹"]:,.0f}
- Value: ₹{r["Value ₹"]:,.0f}
- P&L: ₹{r["P&L ₹"]:+,.0f} ({r["P&L %"]:+.1f}%)
- Portfolio weight: {r["Port %"]}%
""")
                                with c2:
                                    st.markdown("**Framework Scores**")
                                    sc = r["Master Score"]
                                    col = "#2ecc71" if sc>=65 else ("#f39c12" if sc>=45 else "#e74c3c")
                                    st.markdown(f"""
- Master Score: **{sc}/100**
- CANSLIM: {r["CANSLIM"]}/100
- Minervini: {r["Minervini/8"]}/8
- RSI: {r["RSI"]:.0f}
- 3M Momentum: {r["3M Mom%"]:+.1f}%
""")
                                    st.markdown(
                                        f'<div style="background:#2e3250;border-radius:4px;height:10px;">'                                        f'<div style="background:{col};width:{sc}%;height:10px;border-radius:4px;"></div>'                                        f'</div><small>{sc}/100</small>',
                                        unsafe_allow_html=True
                                    )
                                with c3:
                                    st.markdown("**Risk Levels**")
                                    vs_stop = "⛔ BELOW STOP" if r["Current ₹"] < r["Stop ₹"] else f"₹{r['Current ₹']-r['Stop ₹']:.0f} above"
                                    vs_tgt  = "✅ TARGET HIT" if r["Current ₹"] >= r["2R Target ₹"] else f"₹{r['2R Target ₹']-r['Current ₹']:.0f} away"
                                    st.markdown(f"""
- Stop Loss: ₹{r["Stop ₹"]}
- 2R Target: ₹{r["2R Target ₹"]}
- vs Stop: {vs_stop}
- vs 2R Target: {vs_tgt}
""")
                                sig_bg = {"EXIT":"#3b0d0d","TRIM":"#3b2a0d",
                                           "WATCH":"#3b2a0d","HOLD":"#0d1b3b","ADD":"#0d3b1e"}
                                sig_key = next((k for k in sig_bg if k in r["Signal"]), "HOLD")
                                st.markdown(
                                    f'<div style="background:{sig_bg[sig_key]};padding:10px 14px;'                                    f'border-radius:8px;margin-top:8px;">'                                    f'<strong>{r["Signal"]}</strong> — {r["Signal Reason"]}'                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                    if errors:
                        with st.expander(f"⚠️ {len(errors)} position(s) could not be analysed"):
                            for e in errors:
                                st.write("•", e)

    else:
        st.info("👆 Upload your Groww Stocks Holdings Statement Excel file to get started.")
        st.markdown("""
**What this tab does once you upload:**
- Auto-reads all your equity holdings, quantities, and average buy prices
- Filters out mutual funds, ETFs, and gold bonds automatically
- Runs all 4 frameworks (CANSLIM, Minervini, Momentum, Master Score) on each position
- Shows P&L by position and by sector
- Gives Hold / Add / Trim / Exit signals with specific reasons
- Flags sector concentration risks
""")

    # ── WATCHLIST ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Watchlist — Entry Signal Tracker")
    st.caption("Stocks you're tracking but haven't bought yet")

    watch_tickers = [t.strip() for t in watchlist_input.split(",") if t.strip()]

    if watch_tickers and st.button("🔍 Check Watchlist", key="check_watchlist"):
        watch_rows  = []
        watch_prog  = st.progress(0)

        for i, ticker in enumerate(watch_tickers):
            df = fetch_data(ticker)
            if df.empty or len(df) < 252:
                watch_rows.append({
                    "Ticker": ticker,
                    "Entry Signal": "❓ No Data",
                    "Master Score": "-", "Minervini/8": "-",
                    "Price ₹": "-", "RSI": "-", "3M Mom%": "-",
                    "Reason": "Insufficient data"
                })
                watch_prog.progress((i+1)/len(watch_tickers))
                continue
            try:
                metrics       = compute_metrics(df, benchmark_df)
                canslim_s, ct = compute_canslim_score(df)
                _, mp, mpct   = compute_minervini_score(df)
                mom_data      = compute_momentum_score(df)
                risk_m        = compute_risk_metrics(df)
                master, _     = compute_master_score(
                    metrics["Score"], ct, mpct,
                    mom_data["MomentumScore"], 50, market_regime
                )
                if master>=70 and mp>=6 and metrics["RSI"]<70 and metrics["Momentum3M"]>0:
                    entry  = "🟢 Strong Entry"
                    reason = f"Score {master}, {mp}/8 Minervini, RSI {metrics['RSI']:.0f} — all conditions aligned"
                elif master>=55 and metrics["Price"]>metrics["SMA200"] and metrics["Momentum3M"]>0:
                    entry  = "🟡 Watch Closely"
                    reason = f"Score {master} — setup developing, not fully confirmed"
                elif metrics["RSI"]<40 and master>=45:
                    entry  = "🟡 Oversold — Wait"
                    reason = "Potentially oversold — wait for RSI recovery above 45"
                else:
                    entry  = "⏳ Not Ready"
                    reason = f"Score {master}, {mp}/8 Minervini — conditions not aligned yet"

                watch_rows.append({
                    "Ticker":       ticker,
                    "Entry Signal": entry,
                    "Master Score": master,
                    "Minervini/8":  mp,
                    "Price ₹":      f"₹{metrics['Price']}",
                    "RSI":          round(metrics["RSI"],1),
                    "3M Mom%":      f"{metrics['Momentum3M']:+.1f}%",
                    "Reason":       reason
                })
            except Exception as e:
                watch_rows.append({
                    "Ticker":ticker,"Entry Signal":"❌ Error",
                    "Master Score":"-","Minervini/8":"-",
                    "Price ₹":"-","RSI":"-","3M Mom%":"-","Reason":str(e)[:60]
                })
            watch_prog.progress((i+1)/len(watch_tickers))

        watch_prog.empty()
        if watch_rows:
            st.dataframe(pd.DataFrame(watch_rows), use_container_width=True, hide_index=True)
