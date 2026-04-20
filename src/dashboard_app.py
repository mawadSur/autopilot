import streamlit as st
import pandas as pd
import requests
import time
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Autopilot Trader Dashboard", layout="wide")

# ----------------------------
# Settings
# ----------------------------
API_URL = "http://localhost:8000/api/state"
REFRESH_RATE = 1.0 # seconds

# ----------------------------
# Helper: Fetch Data
# ----------------------------
def fetch_state():
    try:
        resp = requests.get(API_URL, timeout=0.5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🚀 Autopilot Dashboard")
st.sidebar.info("Real-time monitoring for Live/Paper trading.")
status_placeholder = st.sidebar.empty()

# ----------------------------
# Main Layout
# ----------------------------
col1, col2, col3 = st.columns(3)
price_metric = col1.empty()
equity_metric = col2.empty()
pos_metric = col3.empty()

st.divider()

# Confidence Gauges
st.subheader("Model Confidence")
conf_col1, conf_col2, conf_col3 = st.columns(3)
short_gauge = conf_col1.empty()
hold_gauge = conf_col2.empty()
long_gauge = conf_col3.empty()

st.divider()

# Equity Curve
st.subheader("Equity Curve")
equity_chart = st.empty()

# Recent Trades
st.subheader("Recent Trades")
trade_table = st.empty()

# ----------------------------
# Live Update Loop
# ----------------------------
while True:
    state = fetch_state()
    
    if state:
        status_placeholder.success(f"Connected (Last: {datetime.now().strftime('%H:%M:%S')})")
        
        # Top Metrics
        price_metric.metric("Current Price", f"${state['price']:.2f}")
        equity_metric.metric("Current Equity", f"${state['equity']:,.2f}")
        pos_str = "FLAT"
        if state['position'] == 1: pos_str = "LONG 🟢"
        elif state['position'] == -1: pos_str = "SHORT 🔴"
        pos_metric.metric("Position", pos_str)
        
        # Confidence Gauges
        def create_gauge(title, val, color):
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = val * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': title, 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            return fig

        short_gauge.plotly_chart(create_gauge("Short P(0)", state['p_short'], "red"), use_container_width=True)
        hold_gauge.plotly_chart(create_gauge("Hold P(1)", state['p_hold'], "gray"), use_container_width=True)
        long_gauge.plotly_chart(create_gauge("Long P(2)", state['p_long'], "green"), use_container_width=True)
        
        # Equity Curve
        if state['equity_curve']:
            df_eq = pd.DataFrame(state['equity_curve'])
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=df_eq['ts'], y=df_eq['val'], mode='lines', name='Equity'))
            fig_eq.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
            equity_chart.plotly_chart(fig_eq, use_container_width=True)
            
        # Trades
        if state['recent_trades']:
            df_trades = pd.DataFrame(state['recent_trades'])
            # Reverse to show newest first
            trade_table.table(df_trades.iloc[::-1].head(20))
        else:
            trade_table.write("No trades recorded yet.")
            
    else:
        status_placeholder.error("Disconnected from State Server")
        
    time.sleep(REFRESH_RATE)
