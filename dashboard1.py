import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

# 1. Load Lending Pools Data
@st.cache_data(ttl=3600)
def load_data():
    url = "https://yields.llama.fi/pools"
    response = requests.get(url)
    data = response.json()
    df = pd.json_normalize(data["data"])
    return df

# 2. Load Pool Historical Data
@st.cache_data(ttl=3600)
def load_pool_chart(pool_id):
    url = f"https://yields.llama.fi/chart/{pool_id}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data["data"])
    return df

# 3. Compute risk metrics
def compute_risk_metrics(history_df, tvl_usd):
    """
    Compute simple risk metrics from historical APY and TVL.
    history_df: dataframe with 'apy' and 'timestamp'
    tvl_usd: current TVL in USD (float)
    """
    hist = history_df.dropna(subset=["apy"]).copy()
    if hist.empty or len(hist) < 2:
        return None, None, None, None

    # Volatility of APY level (percentage points)
    apy_level_vol = hist["apy"].std()

    # Volatility of APY changes (day-to-day jumps, % change)
    hist = hist.sort_values("timestamp")
    hist["apy_change"] = hist["apy"].pct_change()
    apy_change_vol = hist["apy_change"].std()
    if apy_change_vol is not None and not np.isnan(apy_change_vol):
        apy_change_vol = apy_change_vol * 100  # convert to %
    else:
        apy_change_vol = None

    mean_apy = hist["apy"].mean()

    # Simple numeric risk score (0–100)
    risk_score = 30  # baseline

    if tvl_usd < 5_000_000:
        risk_score += 15
    if tvl_usd < 1_000_000:
        risk_score += 15
    if mean_apy is not None and mean_apy > 20:
        risk_score += 10
    if apy_level_vol is not None and not np.isnan(apy_level_vol) and apy_level_vol > 5:
        risk_score += 10
    if apy_change_vol is not None and apy_change_vol > 25:
        risk_score += 20

    risk_score = max(0, min(100, risk_score))

    # Text risk flag
    if tvl_usd < 1_000_000 and mean_apy > 15:
        risk_flag = "⚠️ Very high APY + low TVL (high risk)"
    elif apy_change_vol is not None and apy_change_vol > 25:
        risk_flag = "⚠️ Highly unstable APY (large jumps)"
    elif apy_level_vol is not None and not np.isnan(apy_level_vol) and apy_level_vol > 5:
        risk_flag = "⚠️ Highly volatile APY"
    elif tvl_usd < 5_000_000:
        risk_flag = "⚠️ Low liquidity (low TVL)"
    else:
        risk_flag = "✅ Moderate risk based on simple rules"

    return apy_level_vol, apy_change_vol, risk_flag, risk_score


# App

st.title("DeFi Lending Rates")
st.markdown(
    "Real-time DeFi lending rates with simple risk metrics based on APY volatility and TVL. "
    "Data source: DefiLlama."
)

# 4. Load Data
df = load_data()

# Sidebar controls
with st.sidebar:
    st.header("Filters")

    tokens = df["symbol"].dropna().unique()
    token_choice = st.multiselect(
        "Tokens",
        options=sorted(tokens),
        default=["USDC", "USDT", "DAI"],
    )

    platforms = df["project"].dropna().unique()
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
        "Benchmark rate (e.g. T-bill yield, %)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.25,
    )

# 6. Filter Data
filtered_df = df[
    (df["symbol"].isin(token_choice)) & (df["project"].isin(platform_choice))
]

# 7. Display DataFrame sorted by APY
st.header("Lending Pools Data")


def color_tvls(val):
    if val > 50_000_000:
        return "color: green"
    elif val > 10_000_000:
        return "color: yellow"
    elif val > 1_000_000:
        return "color: orange"
    else:
        return "color: red"


def basic_risk_flag(row):
    apy_val = row["apy"]
    tvl = row["tvlUsd"]
    if pd.isna(apy_val) or pd.isna(tvl):
        return "Unknown"

    if tvl < 1_000_000 and apy_val > 15:
        return "Very High"
    elif tvl < 5_000_000:
        return "High"
    elif apy_val > 20:
        return "High"
    elif apy_val < 3 and tvl > 20_000_000:
        return "Low"
    else:
        return "Medium"


if not filtered_df.empty:
    filtered_df = filtered_df.copy()
    filtered_df["riskFlag"] = filtered_df.apply(basic_risk_flag, axis=1)

    sorted_df = filtered_df[
        ["project", "chain", "symbol", "apy", "tvlUsd", "riskFlag"]
    ].sort_values(by="apy", ascending=False)

    styled_df = (
        sorted_df.style.map(color_tvls, subset=["tvlUsd"])
        .format({"apy": "{:.2f}", "tvlUsd": "${:,.0f}"})
    )

    st.dataframe(styled_df, width="stretch")
else:
    st.warning("No pools match your filters. Try selecting more tokens or platforms.")

# 8. Simulate Earnings
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

    # 9. Historical APY Chart + Risk Metrics
    st.header("Historical APY & Risk Metrics")

    pool_id = selected_row["pool"]
    history_df = load_pool_chart(pool_id)

    if not history_df.empty and "timestamp" in history_df.columns:
        # Robust timestamp conversion
        if not np.issubdtype(history_df["timestamp"].dtype, np.datetime64):
            if np.issubdtype(history_df["timestamp"].dtype, np.number):
                history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], unit="s")
            else:
                history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], errors="coerce")

        # Drop rows where timestamp couldn't be parsed
        history_df = history_df.dropna(subset=["timestamp"])

        # Compute risk metrics
        tvl_usd = float(selected_row.get("tvlUsd", 0) or 0)
        apy_level_vol, apy_change_vol, risk_flag, risk_score = compute_risk_metrics(
            history_df, tvl_usd
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
                last_30d = history_df[
                    history_df["timestamp"] >= max_ts - pd.Timedelta(days=30)
                ]
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
