import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from typing import Optional, Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Reusable session for HTTP requests with retries
_session = requests.Session()
_DEFAULT_TIMEOUT = 8  # seconds

# Configure retries/backoff for transient HTTP errors
_retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"])
)
_adapter = HTTPAdapter(max_retries=_retry_strategy)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

def _get_json(url: str, timeout: int = _DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    """
    GET JSON with a shared session, tolerant to several payload shapes:
    - {"data": [...]}
    - [...]
    Returns a dict with "data" key on success, else None.
    Surfaces simple Streamlit warnings when payloads are unexpected.
    """
    try:
        resp = _session.get(url, timeout=timeout)
        status = resp.status_code
        # Keep some visible debug output so failures are not silent
        if status != 200:
            st.warning(f"GET {url} returned HTTP {status}")
        # Try parse JSON
        data = resp.json()
        # Case 1: dict with "data" key (expected)
        if isinstance(data, dict) and "data" in data:
            return data
        # Case 2: endpoint returns a plain list -> wrap into {"data": list}
        if isinstance(data, list):
            return {"data": data}
        # Unexpected shape: log a truncated version for debugging
        try:
            truncated = str(data)[:1000]
        except Exception:
            truncated = "<unprintable payload>"
        st.warning(f"Unexpected JSON payload from {url}: {truncated}")
        return None
    except requests.RequestException as e:
        st.warning(f"Request error for {url}: {e}")
        return None
    except ValueError as e:
        st.warning(f"JSON decode error for {url}: {e}")
        return None

# Data loading with Streamlit caching (auto-refresh every 3600s)
@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    url = "https://yields.llama.fi/pools"
    data = _get_json(url)
    if not data or "data" not in data:
        return pd.DataFrame()
    # dataset is often a list of flat dicts: normalize just in case
    try:
        df = pd.json_normalize(data["data"])
    except Exception:
        df = pd.DataFrame(data["data"])
    return df

@st.cache_data(ttl=3600)
def load_pool_chart(pool_id: str) -> pd.DataFrame:
    if not pool_id:
        return pd.DataFrame()
    url = f"https://yields.llama.fi/chart/{pool_id}"
    data = _get_json(url)
    if not data or "data" not in data:
        return pd.DataFrame()
    raw = data["data"]
    # Data could be list of primitives or nested dicts; normalize to DataFrame
    try:
        df = pd.json_normalize(raw)
    except Exception:
        try:
            df = pd.DataFrame(raw)
        except Exception:
            return pd.DataFrame()
    # Ensure typical columns are available; try some common alternatives
    # If 'apy' is nested (e.g., 'metrics.apy'), flatten will have created that column name already via json_normalize.
    if "apy" not in df.columns:
        # try to find a numeric-looking column and rename it to 'apy' as a fallback
        numeric_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
        # prefer columns that contain 'apy' or 'rate' in their name
        cand = [c for c in numeric_cols if "apy" in c.lower() or "rate" in c.lower()]
        if cand:
            df = df.rename(columns={cand[0]: "apy"})
        else:
            # nothing to do; let caller handle empty apy
            pass
    return df

# Risk metrics based on an assigned score.
def compute_risk_metrics(history_df, tvl_usd, baseline_apy):
    """
    Compute simple risk metrics from historical APY and TVL.
    Returns a tuple: (apy_level_vol, apy_change_vol, risk_flag, risk_score)
    """
    # Early guards
    if history_df is None or not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return None, None, None, None

    # Ensure 'apy' column exists and has at least one non-NA value
    if "apy" not in history_df.columns:
        return None, None, None, None

    apy_non_na = history_df["apy"].dropna()
    if apy_non_na.shape[0] == 0:
        return None, None, None, None

    # If only a single APY observation, provide reasonable fallbacks instead of bailing out.
    single_point = False
    if apy_non_na.shape[0] == 1:
        single_point = True

    hist = history_df.dropna(subset=["apy"]).copy()

    # Optional: sort by timestamp if present
    if "timestamp" in hist.columns:
        try:
            hist = hist.sort_values("timestamp")
        except Exception:
            pass

    # Volatility of APY level (% points)
    try:
        apy_level_vol = float(hist["apy"].std()) if not single_point else 0.0
    except Exception:
        apy_level_vol = None

    # Volatility of APY changes (day-to-day jumps, % change)
    hist["apy_change"] = hist["apy"].pct_change()
    try:
        # For single point, pct_change yields NaN; treat as 0.0 for reporting but we keep None semantics where appropriate
        apy_change_vol = float(hist["apy_change"].std()) if not single_point else 0.0
        if not np.isnan(apy_change_vol):
            apy_change_vol = apy_change_vol * 100.0  # convert to %
        else:
            apy_change_vol = None
    except Exception:
        apy_change_vol = None

    try:
        mean_apy = float(hist["apy"].mean())
    except Exception:
        mean_apy = None

    # TVL component (2-40)
    tvl_score = compute_tvl_score(tvl_usd)

    # APY level volatility component (5–25)
    if apy_level_vol is None or np.isnan(apy_level_vol):
        level_score = 10.0
    else:
        k = 5.0
        vol = max(apy_level_vol, 0.0)
        level_score = 5.0 + 20.0 * (1.0 - np.exp(-vol / k))

    # APY change volatility component (5–20)
    if apy_change_vol is None or np.isnan(apy_change_vol):
        change_score = 10.0
    else:
        k = 15.0
        vol = max(apy_change_vol, 0.0)
        change_score = 5.0 + 15.0 * (1.0 - np.exp(-vol / k))

    # Mean APY component (continuous 0–15); higher APY relative to baseline = more risk
    if mean_apy is None or baseline_apy is None or np.isnan(mean_apy) or np.isnan(baseline_apy):
        apy_score = 5.0
    else:
        excess = max(mean_apy - baseline_apy, 0.0)
        k = 2.0
        apy_score = 15.0 * (1.0 - np.exp(-excess / k))

    raw_score = tvl_score + level_score + change_score + apy_score

    #   tvl:   2–40
    #   level: 5–25
    #   change:5–20
    #   apy:   0–15
    MIN_RAW = 12.0      # 2 + 5 + 5 + 0
    MAX_RAW = 100.0     # 40 + 25 + 20 + 15

    norm = (raw_score - MIN_RAW) / (MAX_RAW - MIN_RAW)
    norm = np.clip(norm, 0.0, 1.0)

    try:
        risk_score = int(round(norm * 100.0))
    except Exception:
        risk_score = None

    # Risk flag
    if risk_score is None:
        risk_flag = "Unknown"
    elif risk_score < 35:
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

    # Ensure a minimum scale so tiny tvl doesn't blow up the log
    tvl_clamped = max(tvl_usd, 500_000)

    log_tvl = np.log10(tvl_clamped)

    # Parameters fitted to risk level curve
    a = 128
    b = 15.5

    tvl_score = a - b * log_tvl

    # Keep score within range
    tvl_score = float(np.clip(tvl_score, 2.0, 40.0))
    return tvl_score

# App UI
st.title("DeFi Lending Rates")
st.markdown(
    "Real-time DeFi lending rates with simple risk metrics based on APY volatility and TVL. "
    "Data source: DefiLlama."
)

# Sidebar: debug toggle + filters
with st.sidebar:
    st.header("Controls")
    debug_toggle = st.checkbox("Show debug logs", value=False)
    st.info("Enable 'Show debug logs' to expose API payload warnings in the UI (useful for debugging).")

# 4. Load Data
# If you want to clear cached results during debugging uncomment the next line:
# st.cache_data.clear()

df = load_data()

# Baseline APY from dataset
baseline_apy = float(df["apy"].dropna().mean()) if "apy" in df.columns and not df["apy"].dropna().empty else 0.0

# Sidebar filters (moved out of previous ordering for clarity)
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
        "Benchmark rate (e.g. T-bill yield, %)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.25,
    )

# 6. Filter Data
filtered_df = df.copy()
if token_choice:
    filtered_df = filtered_df[filtered_df["symbol"].isin(token_choice)]
if platform_choice:
    filtered_df = filtered_df[filtered_df["project"].isin(platform_choice)]

# 7. Display DataFrame sorted by APY
st.header("Lending Pools Data")

def color_tvls(val):
    try:
        v = float(val)
    except Exception:
        return ""
    if v > 50_000_000:
        return "color: green"
    elif v > 10_000_000:
        return "color: darkorange"
    elif v > 1_000_000:
        return "color: orange"
    else:
        return "color: red"

if not filtered_df.empty:
    filtered_df = filtered_df.copy()

    sorted_df = filtered_df[
        ["project", "chain", "symbol", "apy", "tvlUsd", "pool"]
    ].sort_values(by="apy", ascending=False)

    pool_ids = sorted_df["pool"].dropna().unique().tolist() if "pool" in sorted_df.columns else []
    pool_metrics = {}
    for pid in pool_ids:
        hist = load_pool_chart(pid)
        if hist is None or hist.empty:
            if debug_toggle:
                st.write(f"No history DataFrame for pool {pid} (empty or missing).")
            tvl_row = sorted_df.loc[sorted_df["pool"] == pid, "tvlUsd"]
            try:
                tvl_val = float(tvl_row.iloc[0]) if not tvl_row.empty else 0.0
            except Exception:
                tvl_val = 0.0
            # compute risk metrics with no history will return None values
            apy_level_vol, apy_change_vol, risk_flag, risk_score = compute_risk_metrics(pd.DataFrame(), tvl_val, baseline_apy)
        else:
            # confirm we have expected columns (debug)
            if debug_toggle:
                st.write(f"Pool {pid} history columns:", hist.columns.tolist())
                st.write(f"Pool {pid} sample rows:", hist.head(3).to_dict(orient='records'))
            try:
                tvl_row = sorted_df.loc[sorted_df["pool"] == pid, "tvlUsd"]
                tvl_val = float(tvl_row.iloc[0]) if not tvl_row.empty else 0.0
            except Exception:
                tvl_val = 0.0

            apy_level_vol, apy_change_vol, risk_flag, risk_score = compute_risk_metrics(
                hist, tvl_val, baseline_apy
            )

        pool_metrics[pid] = {
            "apy_level_vol": apy_level_vol,
            "apy_change_vol": apy_change_vol,
            "risk_flag": risk_flag,
            "risk_score": risk_score,
        }

    def _get_risk_score_for_pool(pid):
        if pid is None or pd.isna(pid):
            return None
        return pool_metrics.get(pid, {}).get("risk_score")

    def _get_risk_flag_for_pool(pid):
        if pid is None or pd.isna(pid):
            return None
        return pool_metrics.get(pid, {}).get("risk_flag")

    sorted_df["Risk Score"] = sorted_df["pool"].map(_get_risk_score_for_pool)
    sorted_df["Risk Assessment"] = sorted_df["pool"].map(_get_risk_flag_for_pool)

    sorted_df = sorted_df.rename(columns={
        "tvlUsd": "Total Liquidity",
        "Risk Assessment": "Risk Level",
    })

    display_cols = [
        "project",
        "chain",
        "symbol",
        "apy",
        "Total Liquidity",
        "Risk Level",
        "Risk Score",
    ]
    display_cols = [c for c in display_cols if c in sorted_df.columns]
    display_df = sorted_df[display_cols]

    styled_df = display_df.style.format(
        {
            "apy": "{:.2f}",
            "Total Liquidity": "${:,.0f}",
            "Risk Score": "{:.0f}",
        },
        na_rep="N/A",
    )
    if "Total Liquidity" in display_df.columns:
        styled_df = styled_df.applymap(color_tvls, subset=["Total Liquidity"])

    st.dataframe(styled_df, use_container_width=True)
else:
    st.warning("No pools match your filters. Try selecting more tokens or platforms.")

# 8. Simulate Earnings
st.header("Simulate earnings")

if not filtered_df.empty:
    amount = st.number_input("Amount to lend (USD)", value=1000)
    days = st.slider("Duration (days)", 1, 365, 30)

    def _label(row):
        proj = str(row.get("project", ""))
        chain = str(row.get("chain", ""))
        sym = str(row.get("symbol", ""))
        return f"{proj} ({chain}) - {sym}"

    pool_labels = filtered_df.apply(_label, axis=1).tolist()
    pool_ids_for_labels = filtered_df["pool"].tolist()

    selected_label = st.selectbox("Select a pool for simulation", options=pool_labels)

    try:
        selected_idx = pool_labels.index(selected_label)
    except ValueError:
        st.warning("No pool selected for simulation.")
        selected_idx = None

    if selected_idx is None:
        st.warning("No pool selected for simulation.")
    else:
        selected_row = filtered_df.iloc[selected_idx]
        apy = float(selected_row.get("apy") or 0.0)
        earnings = amount * (apy / 100.0) * (days / 365.0)
        st.metric(label="Estimated Earnings ($)", value=f"{earnings:.2f}")

        # 9. Historical APY Chart + Risk Metrics
        st.header("Historical APY & Risk Metrics")

        pool_id = pool_ids_for_labels[selected_idx] if selected_idx is not None else selected_row.get("pool")
        history_df = load_pool_chart(pool_id)

        if history_df.empty or "timestamp" not in history_df.columns or "apy" not in history_df.columns:
            st.info("No historical data available for this pool.")
            if debug_toggle:
                st.write("history_df empty or missing expected columns:", history_df.shape, history_df.columns.tolist())
        else:
            # Robust timestamp conversion
            try:
                if np.issubdtype(history_df["timestamp"].dtype, np.number):
                    # Try seconds first; if resulting dates are all in the far past/future, try milliseconds
                    conv = pd.to_datetime(history_df["timestamp"], unit="s", errors="coerce")
                    if conv.dropna().empty:
                        conv = pd.to_datetime(history_df["timestamp"], unit="ms", errors="coerce")
                    history_df["timestamp"] = conv
                else:
                    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], errors="coerce")
            except Exception:
                history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], errors="coerce")

            history_df = history_df.dropna(subset=["timestamp"])

            metrics = None
            if 'pool_metrics' in locals() and pool_id is not None and not pd.isna(pool_id):
                metrics = pool_metrics.get(pool_id)

            if metrics is None:
                tvl_usd = float(selected_row.get("tvlUsd", 0) or 0)
                apy_level_vol, apy_change_vol, risk_flag, risk_score = compute_risk_metrics(history_df, tvl_usd, baseline_apy)
            else:
                apy_level_vol = metrics.get("apy_level_vol")
                apy_change_vol = metrics.get("apy_change_vol")
                risk_flag = metrics.get("risk_flag")
                risk_score = metrics.get("risk_score")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current APY (%)", f"{apy:.2f}")
            with col2:
                st.metric("APY Level Volatility (std dev, % pts)", f"{apy_level_vol:.2f}" if apy_level_vol is not None else "N/A")
            with col3:
                st.metric("APY Change Volatility (std dev, %)", f"{apy_change_vol:.2f}" if apy_change_vol is not None else "N/A")
            with col4:
                st.metric("Risk Score (0–100)", f"{risk_score:.0f}" if risk_score is not None else "N/A")

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
                title=f"Historical APY vs Benchmark for {selected_label}",
            )
            fig.update_traces(mode="lines")
            fig.update_layout(title_x=0.5, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
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
