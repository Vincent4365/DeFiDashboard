import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

# Data loading with Streamlit caching (auto-refresh every 3600s)
# Source: DefiLlama yields and historical APY data
@st.cache_data(ttl=3600)
def load_data():
    url = "https://yields.llama.fi/pools"
    response = requests.get(url)
    data = response.json()
    df = pd.json_normalize(data["data"])
    return df

@st.cache_data(ttl=3600)
def load_pool_chart(pool_id):
    url = f"https://yields.llama.fi/chart/{pool_id}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data["data"])
    return df

# Risk metrics based on an assigned score
def compute_risk_metrics(history_df, tvl_usd, baseline_apy):
    """
    Compute risk metrics from historical APY and TVL.
    history_df: dataframe with 'apy' and 'timestamp'
    tvl_usd: current TVL in USD (float)
    """
    hist = history_df.dropna(subset=["apy"]).copy()
    if hist.empty or len(hist) < 2:
        return None, None, None, None

    # Volatility of APY level (% points)
    apy_level_vol = hist["apy"].std()

    # Volatility of APY changes (day-to-day jumps, % change)
    hist = hist.sort_values("timestamp")
    hist["apy_change"] = hist["apy"].pct_change()
    apy_change_vol = hist["apy_change"].std()
    if apy_change_vol is not None and not np.isnan(apy_change_vol):
        apy_change_vol = apy_change_vol * 100.0
    else:
        apy_change_vol = None

    mean_apy = hist["apy"].mean()

    # TVL component (2-40)
    tvl_score = compute_tvl_score(tvl_usd)

    # APY level volatility component (5–25)
    if apy_level_vol is None or np.isnan(apy_level_vol):
        level_score = 10
    else:
        k = 5
        vol = max(apy_level_vol, 0)
        level_score = 5 + 20 * (1 - np.exp(-vol / k))

    # APY change volatility component (5–20)
    if apy_change_vol is None or np.isnan(apy_change_vol):
        change_score = 10
    else:
        k = 15
        vol = max(apy_change_vol, 0)
        change_score = 5 + 15 * (1 - np.exp(-vol / k))

    # Mean APY component
    if (
        mean_apy is None
        or np.isnan(mean_apy)
        or baseline_apy is None
        or np.isnan(baseline_apy)
    ):
        apy_score = 5
    else:
        
        excess = max(mean_apy - baseline_apy, 0.0)

        k = 2.0
        apy_score = 15.0 * (1.0 - np.exp(-excess / k))

    # Raw total from the four components
    raw_score = tvl_score + level_score + change_score + apy_score

    MIN_RAW = 12      # 2 + 5 + 5 + 0
    MAX_RAW = 100     # 40 + 25 + 20 + 15

    norm = (raw_score - MIN_RAW) / (MAX_RAW - MIN_RAW)
    norm = np.clip(norm, 0.0, 1.0)
    
    try:
        risk_score = int(round(norm * 100.0))

        if tvl_usd < 250_000:
            risk_score = max(risk_score, 95)
        elif tvl_usd < 500_000:
            risk_score = max(risk_score, 90)
        elif tvl_usd < 1_000_000:
            risk_score = max(risk_score, 85)
        elif tvl_usd < 5_000_000:
            risk_score = max(risk_score, 60)
    
    except Exception:
        risk_score = None

    if risk_score is None:
        risk_flag = "⚠️ Risk score unavailable (insufficient or invalid data)"
        return apy_level_vol, apy_change_vol, risk_flag, risk_score

    if risk_score < 35:
        risk_flag = "✅ Overall low risk (large TVL / relatively stable APY)"
    elif risk_score < 65:
        risk_flag = "⚠️ Medium risk (moderate volatility and/or mid TVL)"
    else:
        risk_flag = "⚠️ High risk (low liquidity and/or highly volatile APY)"

    return apy_level_vol, apy_change_vol, risk_flag, risk_score

def compute_tvl_score(tvl_usd: float) -> float:
    """
    TVL-based risk penalty.
    Small pools get a high score (more risky), large pools get a low score.
    Ranged between 2 and 40.
    """
    if tvl_usd is None or tvl_usd <= 0:
        # Treat unknown or zero TVL as maximum risk
        return 40.0

    tvl_clamped = max(tvl_usd, 500_000)

    log_tvl = np.log10(tvl_clamped)

    # Parameters fitted to risk level curve
    a = 128
    b = 15.5

    tvl_score = a - b * log_tvl

    # Keep score within range
    tvl_score = float(np.clip(tvl_score, 2.0, 40.0))
    return tvl_score
    
# App

st.title("DeFi Lending Rates")
st.markdown(
    "Real-time DeFi lending rates with simple risk metrics based on APY volatility and TVL. "
    "Data source: DefiLlama."
)

df = load_data()

# Baseline APY from dataset
baseline_apy = (
    float(df["apy"].dropna().mean())
    if "apy" in df.columns and not df["apy"].dropna().empty
    else 0.0
)

# Sidebar controls
with st.sidebar:
    st.header("Filters")

    tokens = df["symbol"].dropna().unique() if "symbol" in df.columns else []
    token_choice = st.multiselect(
        "Tokens",
        options=sorted(tokens),
        default=[t for t in ["USDC", "USDT", "DAI"] if t in tokens],
    )

    platforms = df["project"].dropna().unique() if "project" in df.columns else []
    platforms = sorted(platforms)

    default_platforms = []
    for platform in platforms:
        if platform.lower() in ["aave-v2", "aave-v3", "compound-v2", "compound-v3"]:
            default_platforms.append(platform)

    platform_choice = st.multiselect(
        "Platforms",
        options=platforms,
        default=default_platforms,
    )

    benchmark_rate = st.slider(
        "Benchmark rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.01,
    )

# Filter Data
filtered_df = df.copy()
if token_choice:
    filtered_df = filtered_df[filtered_df["symbol"].isin(token_choice)]
if platform_choice:
    filtered_df = filtered_df[filtered_df["project"].isin(platform_choice)]

# Display DataFrame sorted by APY
st.header("Lending Pools Data")

def color_tvls(val):
    try:
        v = float(val)
    except Exception:
        return ""

    if v > 50_000_000:
        return "color: green"          # Low risk
    elif v > 10_000_000:
        return "color: goldenrod"      # Medium
    elif v > 1_000_000:
        return "color: orangered"      # High
    else:
        return "color: red"            # Very High

def tvl_risk(tvl):
    try:
        v = float(tvl)
    except:
        return "Unknown"

    if v > 50_000_000:
        return "Low"
    elif v > 10_000_000:
        return "Medium"
    elif v > 1_000_000:
        return "High"
    else:
        return "Very High"
        
if not filtered_df.empty:
    filtered_df = filtered_df.copy()
    filtered_df["riskFlag"] = filtered_df["tvlUsd"].apply(tvl_risk)

    sorted_df = filtered_df[
        ["project", "chain", "symbol", "apy", "tvlUsd", "riskFlag"]
    ].sort_values(by="apy", ascending=False)

    # Rename only for display
    sorted_df = sorted_df.rename(columns={
        "tvlUsd": "Total Liquidity",
        "riskFlag": "TVL Risk",
    })

    # Style the table
    styled_df = (
        sorted_df.style
        .map(color_tvls, subset=["Total Liquidity"])
        .format({"apy": "{:.2f}", "Total Liquidity": "${:,.0f}"})
    )

    st.dataframe(styled_df, use_container_width=True)
else:
    st.warning("No pools match your filters. Try selecting more tokens or platforms.")

    # Historical APY Chart + Risk Metrics
    st.header("Historical APY & Risk Metrics")

    pool_id = selected_row["pool"]
    history_df = load_pool_chart(pool_id)

    if not history_df.empty and "timestamp" in history_df.columns:
        # Timestamp conversion
        if np.issubdtype(history_df["timestamp"].dtype, np.number):
            history_df["timestamp"] = pd.to_datetime(
                history_df["timestamp"], unit="s", errors="coerce"
            )
        else:
            history_df["timestamp"] = pd.to_datetime(
                history_df["timestamp"], errors="coerce"
            )

        # Drop rows where timestamp couldn't be parsed
        history_df = history_df.dropna(subset=["timestamp"])

        # Compute risk metrics
        tvl_usd = float(selected_row.get("tvlUsd", 0) or 0)
        apy_level_vol, apy_change_vol, risk_flag, risk_score = compute_risk_metrics(
            history_df, tvl_usd, baseline_apy
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current APY (%)", f"{apy:.2f}")
        with col2:
            if apy_level_vol is not None:
                st.metric("APY Level Volatility (std dev, % pts)", f"{apy_level_vol:.2f}")
        with col3:
            if apy_change_vol is not None:
                st.metric("APY Change Volatility (std dev, %)", f"{apy_change_vol:.2f}")
        with col4:
            if risk_score is not None:
                st.metric("Risk Score (0–100)", f"{risk_score:.0f}")

        # 30-day summary stats
        if not history_df.empty:
            max_ts = history_df["timestamp"].max()
            if pd.notna(max_ts):
                cutoff = max_ts - pd.Timedelta(days=30)
                last_30d = history_df[history_df["timestamp"] >= cutoff]
                if not last_30d.empty:
                    mean_30d = last_30d["apy"].mean()
                    min_apy = history_df["apy"].min()
                    max_apy = history_df["apy"].max()
                    st.write(
                        f"**Last 30 days average APY:** {mean_30d:.2f}%  |  "
                        f"**Historical min:** {min_apy:.2f}%  |  "
                        f"**max:** {max_apy:.2f}%"
                    )

        if risk_flag:
            st.info(f"Risk assessment: {risk_flag}")

        # Benchmark line
        history_df = history_df.sort_values("timestamp")
        history_df["benchmark"] = benchmark_rate

        fig = px.line(
            history_df,
            x="timestamp",
            y=["apy", "benchmark"],
            labels={
                "value": "APY (%)",
                "timestamp": "Date",
                "variable": "Series",
            },
            title=f"Historical APY vs Benchmark for {selected_pool}",
        )
        fig.update_traces(mode="lines")
        fig.update_layout(title_x=0.5, legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No historical data available for this pool.")

# Simulate Earnings
st.header("Simulate earnings")

if not filtered_df.empty:
    amount = st.number_input("Amount to lend (USD)", value=1000)
    days = st.slider("Duration (days)", 1, 365, 30)

    pool_labels = (
        filtered_df["project"] + " (" + filtered_df["chain"] + ") - " + filtered_df["symbol"]
    )
    selected_pool = st.selectbox("Select a pool for simulation", options=pool_labels)

    selected_row = filtered_df[
        (filtered_df["project"] + " (" + filtered_df["chain"] + ") - " + filtered_df["symbol"])
        == selected_pool
    ].iloc[0]

    apy = selected_row["apy"]
    earnings = amount * (apy / 100) * (days / 365)

    st.metric(label="Estimated Earnings ($)", value=f"{earnings:.2f}")
    
else:
    st.warning("No pool selected for simulation.")

# Methodology & Disclaimer
st.markdown("---")
st.subheader("Methodology & Disclaimer")
st.markdown(
    """
- **Data source:** DefiLlama lending pool API.  
- **APY Level Volatility:** Standard deviation of historical APY observations (percentage points).  
- **APY Change Volatility:** Standard deviation of day-to-day percentage changes in APY.  
- **Risk Score & Flags:** Heuristic rules combining TVL, APY level, and volatility; these are simplified indicators.  

This dashboard is for educational and research purposes only and does **not** constitute financial advice.
"""
)
