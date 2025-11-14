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
    df = pd.json_normalize(data['data'])
    return df

# 2. Load Pool Historical Data
@st.cache_data(ttl=3600)
def load_pool_chart(pool_id):
    url = f"https://yields.llama.fi/chart/{pool_id}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['data'])
    return df

# 3. Compute risk metrics
def compute_risk_metrics(history_df, tvl_usd):
    """
    Compute simple risk metrics from historical APY and TVL.
    history_df: dataframe with 'apy' and 'timestamp'
    tvl_usd: current TVL in USD (float)
    """
    # Clean data
    hist = history_df.dropna(subset=['apy']).copy()
    if hist.empty:
        return None, None

    # Volatility of APY
    apy_vol = hist['apy'].std()  # standard deviation in percentage points

    # Simple risk flags
    if tvl_usd < 1_000_000 and hist['apy'].mean() > 15:
        risk_flag = "⚠️ Very high APY + low TVL (high risk)"
    elif apy_vol is not None and apy_vol > 5:
        risk_flag = "⚠️ Highly volatile APY"
    elif tvl_usd < 5_000_000:
        risk_flag = "⚠️ Low liquidity (low TVL)"
    else:
        risk_flag = "✅ Moderate risk based on simple rules"

    return apy_vol, risk_flag


# 3. App Layout
st.title('DeFi Lending rates')
st.markdown('Real-time lending rates. Data source: DefiLlama.')

# 4. Load Data
df = load_data()

# 5. Filters
tokens = df['symbol'].dropna().unique()
token_choice = st.multiselect(
    "Select tokens to view",
    options=sorted(tokens),
    default=['USDC', 'USDT', 'DAI']
)

platforms = df['project'].dropna().unique()
platforms = sorted(platforms)

default_platforms = []
for platform in platforms:
    if platform.lower() in ['aave-v2', 'aave-v3', 'compound-v2', 'compound-v3']:
        default_platforms.append(platform)

platform_choice = st.multiselect(
    "Select DeFi platforms",
    options=platforms,
    default=default_platforms
)

# 6. Filter Data
filtered_df = df[
    (df['symbol'].isin(token_choice)) &
    (df['project'].isin(platform_choice))
]

# 7. Display DataFrame sorted by APY
st.header('Lending Pools Data')

# Color-coding TVL based on new thresholds
def color_tvls(val):
    if val > 50_000_000:
        return 'color: green'
    elif val > 10_000_000:
        return 'color: yellow'
    elif val > 1_000_000:
        return 'color: orange'
    else:
        return 'color: red'

def basic_risk_flag(row):
    apy_val = row['apy']
    tvl = row['tvlUsd']
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
    filtered_df['riskFlag'] = filtered_df.apply(basic_risk_flag, axis=1)

    sorted_df = filtered_df[['project', 'chain', 'symbol', 'apy', 'tvlUsd', 'riskFlag']].sort_values(by='apy', ascending=False)
    styled_df = sorted_df.style.applymap(color_tvls, subset=['tvlUsd'])

    st.dataframe(
        styled_df,
        use_container_width=True
    )
else:
    st.warning("No pools match your filters. Try selecting more tokens or platforms.")


# 8. Simulate Earnings
st.header("Simulate earnings")

if not filtered_df.empty:
    amount = st.number_input("Amount to lend (USD)", value=1000)
    days = st.slider("Duration (days)", 1, 365, 30)

    # Show project and chain in simulation selector
    selected_pool = st.selectbox(
        "Select a pool for simulation",
        options=filtered_df['project'] + " (" + filtered_df['chain'] + ") - " + filtered_df['symbol']
    )

    # Adjust matching
    selected_row = filtered_df[
        (filtered_df['project'] + " (" + filtered_df['chain'] + ") - " + filtered_df['symbol']) == selected_pool
    ].iloc[0]

    apy = selected_row['apy']
    earnings = amount * (apy / 100) * (days / 365)

    st.metric(label="Estimated Earnings ($)", value=f"{earnings:.2f}")

     # 9. Historical APY Chart + Risk Metrics
    st.header("Historical APY & Risk Metrics")

    pool_id = selected_row['pool']
    history_df = load_pool_chart(pool_id)

    if not history_df.empty:
        # Make sure timestamp is in datetime
        if not np.issubdtype(history_df['timestamp'].dtype, np.datetime64):
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], unit='s', errors='ignore')

        # Compute risk metrics
        tvl_usd = float(selected_row.get('tvlUsd', 0) or 0)
        apy_vol, risk_flag = compute_risk_metrics(history_df, tvl_usd)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Current APY (%)",
                value=f"{apy:.2f}"
            )
        with col2:
            if apy_vol is not None:
                st.metric(
                    label="APY Volatility (std dev, % points)",
                    value=f"{apy_vol:.2f}"
                )

        if risk_flag:
            st.info(f"Risk assessment: {risk_flag}")

        # Plot historical APY
        fig = px.line(
            history_df,
            x='timestamp',
            y='apy',
            title=f"Historical APY for {selected_pool}",
            labels={"apy": "APY (%)", "timestamp": "Date"}
        )
        fig.update_traces(line_color='cyan')
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data available for this pool.")


else:
    st.warning("No pool selected for simulation.")
