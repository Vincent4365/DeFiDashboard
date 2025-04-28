import requests
import pandas as pd
import streamlit as st
import plotly.express as px

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

# 3. App Layout
st.title('DeFi Lending rates')
st.markdown('Real-time lending rates. Data source: DefiLlama. Final Project FA 591 Blockchain Technologies & Decentralized Finance.')

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

if not filtered_df.empty:
    sorted_df = filtered_df[['project', 'chain', 'symbol', 'apy', 'tvlUsd']].sort_values(by='apy', ascending=False)
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

    # 9. Historical APY Chart
    st.header("Historical APY Chart")

    pool_id = selected_row['pool']
    history_df = load_pool_chart(pool_id)

    if not history_df.empty:
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
