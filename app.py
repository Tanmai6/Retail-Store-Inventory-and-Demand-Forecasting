import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
from orchestrator import handle_query
from groq import Groq
groq_client = Groq()
# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -----------------------
# DATA
# -----------------------
df = pd.read_csv('data/processed_sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# ── STEP 1: Load pre-computed ML forecasts and merge into df ─────────────────
FORECAST_PATH = "data/demand_forecast_results.csv"
FORECAST_COLS = ["Forecast_RandomForest", "Forecast_XGBoost", "Forecast_Ensemble"]

@st.cache_data
def load_forecast():
    if not os.path.exists(FORECAST_PATH):
        return None
    fc = pd.read_csv(FORECAST_PATH, parse_dates=["Date"])
    for col in ["Store ID", "Product ID"]:
        if col in fc.columns:
            fc[col] = fc[col].astype(str)
    return fc

forecast_df = load_forecast()

if forecast_df is not None:
    df["Store ID"]   = df["Store ID"].astype(str)
    df["Product ID"] = df["Product ID"].astype(str)
    df = df.merge(
        forecast_df[["Date", "Store ID", "Product ID"] + FORECAST_COLS],
        on=["Date", "Store ID", "Product ID"],
        how="left"
    )
    FORECAST_AVAILABLE = True
else:
    for col in FORECAST_COLS:
        df[col] = np.nan
    FORECAST_AVAILABLE = False

# -----------------------
# HELPERS
# -----------------------
def fmt_money(x):
    if x >= 1_000_000: return f"${x/1_000_000:.2f}M"
    if x >= 1_000: return f"${x/1_000:.1f}K"
    return f"${x:.0f}"

def fmt_num(x):
    if x >= 1_000_000: return f"{x/1_000_000:.2f}M"
    if x >= 1_000: return f"{x/1_000:.1f}K"
    return f"{x:.0f}"

def fmt_pct(x):
    return f"{x:.1f}%"

PLOTLY_COLORS = ["#6C63FF", "#00C48C", "#FF8C42", "#4EAEFF", "#FF4C61", "#A78BFA"]

# -----------------------
# DARK PLOT THEME
# -----------------------
PLOT_BG   = "#1E2235"
PAPER_BG  = "#1E2235"
GRID_CLR  = "rgba(255,255,255,0.07)"
FONT_CLR  = "#CBD5E1"
TITLE_CLR = "#E2E8F0"

def dark_layout(fig, title="", height=300, legend_h=False):
    fig.update_layout(
        title=dict(text=title, font=dict(color=TITLE_CLR, size=13, family="Inter"), x=0),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter", size=11, color=FONT_CLR),
        margin=dict(l=12, r=12, t=40 if title else 12, b=12),
        height=height,
        legend=dict(
            orientation="h" if legend_h else "v",
            y=1.12 if legend_h else 1,
            font=dict(color=FONT_CLR),
            bgcolor="rgba(0,0,0,0)"
        )
    )
    fig.update_xaxes(showgrid=False, color=FONT_CLR, linecolor=GRID_CLR, tickfont=dict(color=FONT_CLR))
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR, color=FONT_CLR, linecolor=GRID_CLR, tickfont=dict(color=FONT_CLR))
    return fig

# -----------------------
# GLOBAL CSS
# -----------------------
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------
# KPI CARD HELPER
# -----------------------
def kpi_row(items):
    cards = ""
    for label, value, delta, dt in items:
        cls = f"kpi-delta-{dt}"
        arrow = "▲" if dt == "pos" else "▼" if dt == "neg" else "●"
        cards += f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="{cls}">{arrow} {delta}</div>
        </div>"""
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)

def get_ai_insights(page_key: str, store_id: str = None, data_summary: str = ""):
    cache_key = f"ai_insights_{page_key}"
    
    if cache_key not in st.session_state:
        with st.spinner("🤖 Generating AI insights..."):
            try:
                if page_key == "warehouse":
                    prompt = f"""You are a retail analyst. Based on this data summary, give exactly 4 concise inventory insights.
Each insight on a new line, starting with a relevant emoji.
Data Summary:
{data_summary}
Cover: top stockout risk product, most overstocked category, coverage days, lost revenue estimate."""

                elif page_key == "ceo":
                    prompt = f"""You are a retail analyst. Based on this data summary, give exactly 4 strategic business insights.
Each insight on a new line, starting with a relevant emoji.
Data Summary:
{data_summary}
Cover: best/worst region, promotion effectiveness, inventory health, revenue trend."""

                else:  # branch
                    prompt = f"""You are a retail analyst. Based on this data summary for Store {store_id}, give exactly 4 concise insights.
Each insight on a new line, starting with a relevant emoji.
Data Summary:
{data_summary}
Cover: top category, weather impact, promotion lift, stockout risk."""

                response = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a concise retail intelligence analyst. Output exactly 4 lines of insight, each starting with an emoji. No extra text."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=300
                )

                raw = response.choices[0].message.content  # ← raw is defined here
                lines = [l.strip() for l in raw.strip().split("\n") if l.strip()][:4]
                st.session_state[cache_key] = lines

            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    st.session_state[cache_key] = [
                        "⏳ Token limit reached. Insights will load after reset (resets daily).",
                        "📊 Check the charts above for live store metrics.",
                        "💬 Chat queries also paused temporarily.",
                        "🔄 Click 'Refresh AI Insights' after a few minutes."
                    ]
                else:
                    st.session_state[cache_key] = [f"⚠️ Could not load insights: {str(e)}"]

    return st.session_state[cache_key]

# -----------------------
# AI PANEL
# -----------------------
def ai_assistant_panel(page_key="global"):
    st.markdown("### 🤖 AI Assistant")
    st.caption("Ask anything about your retail data")

    history_key = f"chat_history_{page_key}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    history = st.session_state[history_key]

    chat_html = '<div class="chat-wrap">'
    if not history:
        chat_html += '<div style="text-align:center;color:#475569;padding:30px 0;font-size:13px;">👋 Ask me about revenue, inventory, trends, or store performance.</div>'
    for msg in history:
        if msg["role"] == "user":
            chat_html += f'<div class="bubble-user">{msg["content"]}</div>'
        else:
            chat_html += f'<div class="bubble-bot">🤖 {msg["content"]}</div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    quick_prompts = [
        "What is the stockout risk?",
        "Which Category earns most?",
        "How do Ps impact sales?",
        "Which Region is underperforming?",
    ]

    qc = st.columns(2)
    # In ai_assistant_panel(), replace the quick prompts loop:

    for i, prompt in enumerate(quick_prompts):
        with qc[i % 2]:
            btn_key = f"quick_{page_key}_{i}"
            if st.button(prompt, key=btn_key):
                if f"processing_{btn_key}" not in st.session_state:
                    st.session_state[f"processing_{btn_key}"] = True
                    history.append({"role": "user", "content": prompt})
                    with st.spinner("Thinking..."):
                        try:
                            reply = handle_query(prompt, store_id=st.session_state.get("active_store_scope"))
                        except Exception as e:
                            if "429" in str(e) or "rate_limit" in str(e).lower():
                                reply = "⏳ Daily token limit reached. Please wait a few minutes."
                            else:
                                reply = f"⚠️ Error: {str(e)}"
                    history.append({"role": "assistant", "content": reply})
                    st.session_state[history_key] = history
                    del st.session_state[f"processing_{btn_key}"]
                    st.rerun()
    


    with st.form(f"chat_form_{page_key}", clear_on_submit=True):
        user_input = st.text_input("Type your question...",
                                   placeholder="e.g. What products are at stockout risk?",
                                   label_visibility="collapsed")
        submitted = st.form_submit_button("Send ➤", use_container_width=True)
    if submitted and user_input.strip():
        history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            try:
                reply = handle_query(user_input, store_id=st.session_state.get("active_store_scope"))
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    reply = "⏳ Daily token limit reached. Please wait a few minutes and try again."
                else:
                    reply = f"⚠️ Error: {str(e)}"
        history.append({"role": "assistant", "content": reply})
        st.session_state[history_key] = history
        st.rerun()

    if history:
        if st.button("🗑 Clear Chat", key=f"clear_chat_{page_key}", use_container_width=True):
            st.session_state[history_key] = []
            st.rerun()


# -----------------------
# SIDEBAR
# -----------------------
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:18px 0 8px;'>
            <div style='font-size:30px;'>🛒</div>
            <div style='font-size:15px;font-weight:700;color:#E2E8F0;margin-top:6px;'>Retail Analytics</div>
            <div style='font-size:11px;color:#475569;margin-top:2px;'>Intelligence Control Tower</div>
        </div>
        <hr style='border-color:rgba(255,255,255,0.06);margin:10px 0;'>
        """, unsafe_allow_html=True)

        # Forecast status badge
        if FORECAST_AVAILABLE:
            st.markdown("<div style='text-align:center;color:#00C48C;font-size:12px;margin-bottom:8px;'>🔮 ML Forecasts Active</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center;color:#FF8C42;font-size:12px;margin-bottom:8px;'>⚠️ Run forecasting.py to enable ML</div>", unsafe_allow_html=True)

        pages = [
            ("🏠", "Home", "home"),
            ("📊", "CEO Dashboard", "ceo"),
            ("🏭", "Warehouse Dashboard", "warehouse"),
            ("🏬", "Branch Dashboard", "branch"),
        ]
        for icon, label, key in pages:
            if st.button(f"{icon}  {label}", key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()

        st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:14px 0;'>", unsafe_allow_html=True)
        with st.expander("🤖 AI Assistant", expanded=False):
            ai_assistant_panel(page_key="sidebar")


# -----------------------
# FORECAST CURVE HELPER
# -----------------------
def make_forecast_curve(historical_series, periods=14, label="Forecast"):
    vals = historical_series.values[-30:] if len(historical_series) >= 30 else historical_series.values
    x = np.arange(len(vals))
    slope, intercept = np.polyfit(x, vals, 1)
    future_x = np.arange(len(vals), len(vals) + periods)
    base = slope * future_x + intercept
    noise_std = np.std(np.diff(vals)) if len(vals) > 1 else base.mean() * 0.05
    rng = np.random.default_rng(99)
    noise = rng.normal(0, noise_std, periods).cumsum()
    forecast = base + noise
    ci = noise_std * 1.96
    return forecast, forecast - ci, forecast + ci


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST CHART HELPER  (reused in CEO + Branch dashboards)
# ─────────────────────────────────────────────────────────────────────────────
def render_forecast_chart(d, height=320):
    """
    Renders the ML Actual vs Predicted chart for dataframe slice `d`.
    Shows an info banner instead of crashing when forecasting.py hasn't been run.
    """
    st.markdown(
        "<div class='section-header'>🔮 Predictive Demand Forecasting Performance</div>",
        unsafe_allow_html=True
    )

    has_data = (
        FORECAST_AVAILABLE
        and all(c in d.columns for c in ["Demand"] + FORECAST_COLS)
        and d[FORECAST_COLS].notna().any().any()
    )

    if not has_data:
        st.info(
            "🔮 **ML Forecast chart unlocks after you run `python forecasting.py` once.**  \n"
            "The chart will appear automatically on next dashboard reload.",
            icon="ℹ️"
        )
        return

    timeline_df = (
        d.groupby("Date")[["Demand"] + FORECAST_COLS]
        .sum()
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timeline_df["Date"], y=timeline_df["Demand"],
        mode="lines", name="Actual Demand",
        line=dict(color="#1f77b4", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=timeline_df["Date"], y=timeline_df["Forecast_RandomForest"],
        mode="lines", name="Random Forest",
        line=dict(color="skyblue", width=2, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=timeline_df["Date"], y=timeline_df["Forecast_XGBoost"],
        mode="lines", name="XGBoost",
        line=dict(color="coral", width=2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=timeline_df["Date"], y=timeline_df["Forecast_Ensemble"],
        mode="lines", name="Ensemble Blend",
        line=dict(color="#2ca02c", width=2, dash="longdash")
    ))

    fig.update_layout(
        title="Timeline Analysis: Actual Demand vs ML Models",
        xaxis_title="Date",
        yaxis_title="Total Product Units",
        hovermode="x unified",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter", size=11, color=FONT_CLR),
        height=height,
        margin=dict(l=12, r=12, t=45, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(color=FONT_CLR))
    )
    fig.update_xaxes(showgrid=False, color=FONT_CLR, tickfont=dict(color=FONT_CLR))
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR, color=FONT_CLR, tickfont=dict(color=FONT_CLR))

    st.plotly_chart(fig, use_container_width=True)

    # Accuracy KPI row — using HTML cards to avoid st.metric version issues
    valid = d[["Demand", "Forecast_RandomForest", "Forecast_XGBoost"]].dropna()
    if len(valid) > 10:
        mae_rf  = np.mean(np.abs(valid["Demand"] - valid["Forecast_RandomForest"]))
        mae_xgb = np.mean(np.abs(valid["Demand"] - valid["Forecast_XGBoost"]))
        ss_tot  = np.sum((valid["Demand"] - valid["Demand"].mean()) ** 2)
        r2_rf   = 1 - np.sum((valid["Demand"] - valid["Forecast_RandomForest"]) ** 2) / max(ss_tot, 1e-9)
        r2_xgb  = 1 - np.sum((valid["Demand"] - valid["Forecast_XGBoost"]) ** 2) / max(ss_tot, 1e-9)
        kpi_row([
            ("RF — MAE",  f"{mae_rf:.1f} units",  "vs XGBoost", "neu"),
            ("XGB — MAE", f"{mae_xgb:.1f} units", "vs RF",      "neu"),
            ("RF — R²",   f"{r2_rf:.3f}",          "accuracy",   "pos"),
            ("XGB — R²",  f"{r2_xgb:.3f}",         "accuracy",   "pos"),
        ])


# -----------------------
# HOME
# -----------------------
def home():
    st.markdown("""
    <h1 style='font-size:26px;margin-bottom:2px;'>Retail Analytics Dashboard</h1>
    <p style='color:#475569;font-size:13px;margin-bottom:16px;'>Actionable insights · Smarter decisions · Stronger retail performance</p>
    """, unsafe_allow_html=True)

   # -----------------------
    # HYBRID KPI LOGIC
    # -----------------------
    # 1. Get the current day snapshot for inventory metrics
    latest_date = df['Date'].max()
    current_day_df = df[df['Date'] == latest_date]

    # 2. Mix all-time data (df) with current data (current_day_df)
    kpi_row([
        ("Total Revenue", fmt_money(df["Revenue"].sum()), "12.6% vs Apr", "pos"),
        ("Total Units Sold", fmt_num(df["Units Sold"].sum()), "8.3% vs Apr", "pos"),
        ("Current Inventory Value", fmt_money(current_day_df["Inventory Value"].sum()), "6.5% vs Apr", "pos"),
        ("Active Stockout Warnings", fmt_pct(current_day_df["Stockout"].mean()*100), "0.6% vs Apr", "neg"),
        ("Overstock Risk", fmt_pct(current_day_df["Overstock"].mean()*100), "1.3% vs Apr", "neg"),
    ])

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown("### Select Operational Dashboard")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='home-card'>
            <div style='font-size:36px;margin-bottom:10px;'>👔</div>
            <div style='font-size:15px;font-weight:700;color:#818CF8;margin-bottom:6px;'>CEO / Executive Dashboard</div>
            <div style='color:#64748B;font-size:12px;line-height:1.7;'>Enterprise overview, performance trends, and strategic insights.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open CEO Dashboard →", key="home_ceo", use_container_width=True):
            st.session_state.page = "ceo"; st.rerun()

    with c2:
        st.markdown("""<div class='home-card'>
            <div style='font-size:36px;margin-bottom:10px;'>🏭</div>
            <div style='font-size:15px;font-weight:700;color:#34D399;margin-bottom:6px;'>Warehouse Manager Dashboard</div>
            <div style='color:#64748B;font-size:12px;line-height:1.7;'>Inventory overview, stock alerts, and replenishment insights.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Warehouse Dashboard →", key="home_wh", use_container_width=True):
            st.session_state.page = "warehouse"; st.rerun()

    with c3:
        st.markdown("""<div class='home-card'>
            <div style='font-size:36px;margin-bottom:10px;'>🏬</div>
            <div style='font-size:15px;font-weight:700;color:#38BDF8;margin-bottom:6px;'>Branch Manager Dashboard</div>
            <div style='color:#64748B;font-size:12px;line-height:1.7;'>Store performance, sales insights, and Demand overview.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Branch Dashboard →", key="home_br", use_container_width=True):
            st.session_state.page = "branch"; st.rerun()

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Business Snapshot")

    lc, rc = st.columns([2, 1])
    with lc:
        ts = df.groupby(df["Date"].dt.to_period("M")).agg({"Revenue": "sum", "Lost Demand": "sum"}).reset_index()
        ts["Month"] = ts["Date"].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts["Month"], y=ts["Revenue"], name="Revenue",
                                  line=dict(color="#6C63FF", width=2.5),
                                  fill="tozeroy", fillcolor="rgba(108,99,255,0.12)",
                                  mode="lines+markers", marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=ts["Month"], y=ts["Lost Demand"], name="Lost Demand",
                                  line=dict(color="#FF4C61", width=2, dash="dot")))
        dark_layout(fig, "Revenue vs Lost Demand (Monthly)", height=270, legend_h=True)
        st.plotly_chart(fig, use_container_width=True)
    with rc:
        cat = df.groupby("Category")["Revenue"].sum().reset_index()
        fig2 = go.Figure(data=[go.Pie(labels=cat["Category"], values=cat["Revenue"],
                                       hole=0.5, marker_colors=PLOTLY_COLORS,
                                       textinfo="percent+label",
                                       textfont=dict(color="#E2E8F0", size=11))])
        dark_layout(fig2, "Revenue by Category", height=270)
        st.plotly_chart(fig2, use_container_width=True)


# -----------------------
# CEO DASHBOARD
# -----------------------
def ceo_dashboard():
    st.session_state["active_store_scope"] = None
    hc, dc = st.columns([3, 1])
    with hc:
        st.markdown("<h1 style='font-size:22px;'>📊 CEO / Executive Dashboard</h1>", unsafe_allow_html=True)
        st.caption("Strategic Performance Overview")
    with dc:
        date_filter = st.selectbox("Period", ["All Time", "Last 3 Months", "Last Month"],
                                    label_visibility="collapsed", key="ceo_period")

    dff = df.copy()
    if date_filter == "Last Month":
        dff = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=30)]
    elif date_filter == "Last 3 Months":
        dff = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=90)]

    promo_lift = ((dff[dff['Promotion']==1]['Revenue'].mean() /
                   dff[dff['Promotion']==0]['Revenue'].mean()) - 1) * 100

    st.markdown("<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)
    kpi_row([
        ("Total Revenue",      fmt_money(dff["Revenue"].sum()),               "12.6% vs Apr", "pos"),
        ("Units Sold",         fmt_num(dff["Units Sold"].sum()),               "8.3% vs Apr",  "pos"),
        ("Total Demand",       fmt_num(dff["Demand"].sum()),                   "9.7% vs Apr",  "pos"),
        ("Inventory Value",    fmt_money(dff["Inventory Value"].sum()),        "6.5% vs Apr",  "pos"),
        ("Inv. Turnover",      f"{dff['Inventory Turnover'].mean():.2f}x",    "0.8x vs Apr",  "pos"),
        ("Stockout Rate",      fmt_pct(dff["Stockout"].mean()*100),           "0.6% vs Apr",  "neg"),
    ])
    kpi_row([
        ("Est. Gross Profit",  fmt_money(dff["Gross Profit (Proxy)"].sum()),  "4.2% vs Apr",  "pos"),
        ("Lost Demand Units",  fmt_num(dff["Lost Demand"].sum()),             "2.1% vs Apr",  "pos"),
        ("Overstock Risk",     fmt_pct(dff["Overstock"].mean()*100),          "1.3% vs Apr",  "neg"),
        ("Avg Coverage Days",  f"{dff['Coverage Days'].mean():.1f}d",         "0.4d vs Apr",  "pos"),
        ("Promo Impact",       f"+{promo_lift:.1f}%",                         "vs No Promo",  "pos"),
        ("Sell-Through Rate",  fmt_pct(dff["Sell Through Rate"].mean()*100),  "1.8% vs Apr",  "pos"),
    ])

    st.divider()
    st.markdown("<div class='section-header'>Business Performance Over Time</div>", unsafe_allow_html=True)

    ts = dff.groupby(dff["Date"].dt.to_period("M")).agg(
        {"Revenue": "sum", "Units Sold": "sum", "Demand": "sum"}
    ).reset_index()
    ts["Month"] = ts["Date"].astype(str)

    rev_fc, rev_lo, rev_hi = make_forecast_curve(ts["Revenue"], periods=3)
    fc_months = [f"FC+{i+1}" for i in range(3)]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts["Month"], y=ts["Revenue"], name="Revenue",
                                  line=dict(color="#6C63FF", width=2.5),
                                  fill="tozeroy", fillcolor="rgba(108,99,255,0.1)",
                                  mode="lines+markers", marker=dict(size=5)))
        fig.add_trace(go.Scatter(
            x=fc_months, y=rev_fc, name="Forecast",
            line=dict(color="#A78BFA", width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=5, symbol="diamond")))
        fig.add_trace(go.Scatter(
            x=fc_months + fc_months[::-1],
            y=list(rev_hi) + list(rev_lo[::-1]),
            fill="toself", fillcolor="rgba(167,139,250,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI", showlegend=False))
        dark_layout(fig, "Revenue Trend & Forecast (Monthly)", height=290, legend_h=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dem_fc, dem_lo, dem_hi = make_forecast_curve(ts["Demand"], periods=3)
        sold_fc, _, _ = make_forecast_curve(ts["Units Sold"], periods=3)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ts["Month"], y=ts["Demand"], name="Demand",
                                   line=dict(color="#4EAEFF", width=2.5)))
        fig2.add_trace(go.Scatter(x=ts["Month"], y=ts["Units Sold"], name="Units Sold",
                                   line=dict(color="#00C48C", width=2.5)))
        fig2.add_trace(go.Scatter(x=fc_months, y=dem_fc, name="Demand Forecast",
                                   line=dict(color="#7DD3FC", width=2, dash="dash"),
                                   mode="lines+markers", marker=dict(symbol="diamond", size=5)))
        fig2.add_trace(go.Scatter(x=fc_months, y=sold_fc, name="Sales Forecast",
                                   line=dict(color="#6EE7B7", width=2, dash="dash"),
                                   mode="lines+markers", marker=dict(symbol="diamond", size=5)))
        dark_layout(fig2, "Demand vs Units Sold + Forecast", height=290, legend_h=True)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        Region = dff.groupby("Region")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=False)
        fig3 = px.bar(Region, x="Region", y="Revenue", color="Region",
                       color_discrete_sequence=PLOTLY_COLORS, text=Region["Revenue"].apply(fmt_money))
        fig3.update_traces(textposition="outside", textfont=dict(color=FONT_CLR))
        dark_layout(fig3, "Revenue by Region", height=280)
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        cat = dff.groupby("Category")["Revenue"].sum().reset_index()
        fig4 = px.treemap(cat, path=["Category"], values="Revenue",
                           color="Revenue", color_continuous_scale=["#312E81", "#6C63FF"])
        fig4.update_traces(textfont=dict(color="#F1F5F9", size=12))
        fig4.update_layout(paper_bgcolor=PAPER_BG, margin=dict(l=10,r=10,t=40,b=10),
                            height=280, font=dict(family="Inter", size=11, color=FONT_CLR),
                            title=dict(text="Top Product Categories by Revenue",
                                       font=dict(color=TITLE_CLR, size=13)))
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.markdown("<div class='section-header'>Impact Analysis</div>", unsafe_allow_html=True)

    col5, col6, col7 = st.columns(3)
    with col5:
        promo = dff.groupby("Promotion")["Revenue"].mean().reset_index()
        promo["Label"] = promo["Promotion"].map({0: "No Promo", 1: "With Promo"})
        lift_vals = [0, round(promo_lift, 1)]
        fig5 = go.Figure(go.Bar(
            x=promo["Label"], y=lift_vals,
            marker_color=["#374151", "#6C63FF"],
            text=[f"{v:.1f}%" for v in lift_vals],
            textposition="outside", textfont=dict(color=FONT_CLR)
        ))
        dark_layout(fig5, f"Promotion Impact (Sales Lift %)", height=280)
        fig5.update_layout(showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        ep_m = dff.copy()
        ep_m["Month"] = ep_m["Date"].dt.to_period("M").astype(str)
        ep = ep_m.groupby(["Month", "Epidemic"])["Revenue"].sum().reset_index()
        ep["Status"] = ep["Epidemic"].map({0: "Normal", 1: "Epidemic Period"})
        fig6 = px.line(ep, x="Month", y="Revenue", color="Status",
                        color_discrete_map={"Normal": "#00C48C", "Epidemic Period": "#FF4C61"})
        fig6.update_traces(line=dict(width=2.5))
        dark_layout(fig6, "Epidemic Impact on Revenue", height=280, legend_h=True)
        st.plotly_chart(fig6, use_container_width=True)

    with col7:
        healthy = 100 - dff["Stockout"].mean()*100 - dff["Overstock"].mean()*100
        fig7 = go.Figure(data=[go.Pie(
            labels=["Healthy", "Overstock", "Understock"],
            values=[healthy, dff["Overstock"].mean()*100, dff["Stockout"].mean()*100],
            hole=0.62,
            marker_colors=["#00C48C", "#FF8C42", "#FF4C61"],
            textinfo="label+percent",
            textfont=dict(color="#E2E8F0", size=11)
        )])
        fig7.update_layout(
            annotations=[dict(text=f"<b>{healthy:.0f}%</b><br><span style='font-size:10px'>Healthy</span>",
                               x=0.5, y=0.5, font_size=16, showarrow=False, font_color="#F1F5F9")],
            paper_bgcolor=PAPER_BG, margin=dict(l=10,r=10,t=40,b=10), height=280,
            font=dict(family="Inter", size=11, color=FONT_CLR),
            title=dict(text="Inventory Health", font=dict(color=TITLE_CLR, size=13)),
            legend=dict(orientation="h", y=-0.08, font=dict(color=FONT_CLR))
        )
        st.plotly_chart(fig7, use_container_width=True)

    col8, col9 = st.columns(2)
    with col8:
        weather = dff.groupby("Weather Condition")["Revenue"].mean().reset_index()
        avg = dff["Revenue"].mean()
        weather["Impact (%)"] = (weather["Revenue"] / avg - 1) * 100
        fig8 = go.Figure(go.Bar(
            x=weather["Weather Condition"],
            y=weather["Impact (%)"],
            marker_color=["#00C48C" if v >= 0 else "#FF4C61" for v in weather["Impact (%)"]],
            text=[f"{v:.1f}%" for v in weather["Impact (%)"]],
            textposition="outside", textfont=dict(color=FONT_CLR)
        ))
        dark_layout(fig8, "Weather Impact on Revenue (% vs Avg)", height=280)
        fig8.update_layout(showlegend=False)
        st.plotly_chart(fig8, use_container_width=True)

    with col9:
        season = dff.groupby("Seasonality")["Revenue"].sum().reset_index()
        fig9 = go.Figure(go.Bar(
            x=season["Seasonality"], y=season["Revenue"],
            marker_color=PLOTLY_COLORS[:4],
            text=season["Revenue"].apply(fmt_money),
            textposition="outside", textfont=dict(color=FONT_CLR)
        ))
        dark_layout(fig9, "Revenue by Season", height=280)
        fig9.update_layout(showlegend=False)
        st.plotly_chart(fig9, use_container_width=True)

    # ── STEP 2 (CEO): ML Forecast chart ──────────────────────────────────────
    st.divider()
    render_forecast_chart(dff)

    st.divider()
    st.markdown("<div class='section-header'>AI Strategic Insights</div>", unsafe_allow_html=True)
    summary = f"""
Total Revenue: ${dff['Revenue'].sum():,.0f}
Best Region: {dff.groupby('Region')['Revenue'].sum().idxmax()}
Worst Region: {dff.groupby('Region')['Revenue'].sum().idxmin()}
Promo Sales Lift: {promo_lift:.1f}%
Stockout Rate: {dff['Stockout'].mean()*100:.1f}%
Overstock Rate: {dff['Overstock'].mean()*100:.1f}%
Avg Inventory Turnover: {dff['Inventory Turnover'].mean():.2f}x
Top Category: {dff.groupby('Category')['Revenue'].sum().idxmax()}
"""

    if st.button("✨ Generate AI Insights", key="gen_insights_ceo"):
        st.session_state["show_insights_ceo"] = True

    if st.session_state.get("show_insights_ceo"):
        insights = get_ai_insights("ceo", store_id=None, data_summary=summary)
        for line in insights:
            icon = line[0] if line else "💡"
            text = line[2:].strip() if len(line) > 2 else line
            st.markdown(
                f'<div class="insight-card"><span style="font-size:18px;">{icon}</span><span>{text}</span></div>',
                unsafe_allow_html=True
            )
    ai_assistant_panel(page_key="ceo")

    if st.button("← Back to Home", key="ceo_back"):
        st.session_state.page = "home"; st.rerun()


# -----------------------
# WAREHOUSE DASHBOARD
# -----------------------
def warehouse_dashboard():
    st.session_state["active_store_scope"] = None
    hc, dc = st.columns([3, 1])
    with hc:
        st.markdown("<h1 style='font-size:22px;'>🏭 Warehouse Manager Dashboard</h1>", unsafe_allow_html=True)
        st.caption("Inventory & Replenishment Overview")
    with dc:
        date_filter = st.selectbox("Period", ["All Time", "Last 3 Months", "Last Month"],
                                    label_visibility="collapsed", key="wh_period")

    dff = df.copy()
    if date_filter == "Last Month":
        dff = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=30)]
    elif date_filter == "Last 3 Months":
        dff = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=90)]

    st.markdown("<div class='section-header'>Inventory Key Metrics</div>", unsafe_allow_html=True)
    kpi_row([
        ("Inventory Value",    fmt_money(dff["Inventory Value"].sum()),         "6.5% vs Apr",   "pos"),
        ("Stockout SKUs",      fmt_num(dff[dff["Stockout"]].shape[0]),          "15 vs Apr",     "pos"),
        ("Overstock SKUs",     fmt_num(dff[dff["Overstock"]].shape[0]),         "18 vs Apr",     "neg"),
        ("Avg Coverage Days",  f"{dff['Coverage Days'].mean():.0f}d",           "4d vs Apr",     "pos"),
        ("Inv. Accuracy",      "97.6%",                                          "1.2% vs Apr",  "pos"),
        ("Avg Turnover",       f"{dff['Inventory Turnover'].mean():.2f}x",      "0.3x vs Apr",  "pos"),
    ])

    st.divider()
    st.markdown("<div class='section-header'>Inventory Overview</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        inv = dff.groupby("Category").agg(
            Overstock=("Overstock", "sum"),
            Stockout=("Stockout", "sum"),
            Total=("Inventory Level", "count")
        ).reset_index()
        inv["Optimal"] = inv["Total"] - inv["Overstock"] - inv["Stockout"]
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(name="Overstock",   x=inv["Category"], y=inv["Overstock"],  marker_color="#FF8C42"))
        fig1.add_trace(go.Bar(name="Optimal",     x=inv["Category"], y=inv["Optimal"],    marker_color="#00C48C"))
        fig1.add_trace(go.Bar(name="Understock",  x=inv["Category"], y=inv["Stockout"],   marker_color="#FF4C61"))
        fig1.update_layout(barmode="stack")
        dark_layout(fig1, "Current Inventory Status by Category", height=300, legend_h=True)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        inv_ts = dff.groupby(dff["Date"].dt.to_period("M"))["Inventory Value"].sum().reset_index()
        inv_ts["Month"] = inv_ts["Date"].astype(str)
        fc, lo, hi = make_forecast_curve(inv_ts["Inventory Value"], periods=3)
        fc_months = [f"FC+{i+1}" for i in range(3)]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=inv_ts["Month"], y=inv_ts["Inventory Value"],
                                   name="Inventory Value",
                                   line=dict(color="#4EAEFF", width=2.5),
                                   fill="tozeroy", fillcolor="rgba(78,174,255,0.1)",
                                   mode="lines+markers", marker=dict(size=5)))
        fig2.add_trace(go.Scatter(x=fc_months, y=fc, name="Forecast",
                                   line=dict(color="#7DD3FC", width=2, dash="dash"),
                                   mode="lines+markers", marker=dict(symbol="diamond", size=5)))
        fig2.add_trace(go.Scatter(
            x=fc_months + fc_months[::-1],
            y=list(hi) + list(lo[::-1]),
            fill="toself", fillcolor="rgba(125,211,252,0.1)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False))
        dark_layout(fig2, "Inventory Value Trend & Forecast", height=300, legend_h=True)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🚨 Reorder Alerts")
        at_risk = dff[dff["Stockout"]].groupby("Product ID").agg(
            Current_Stock=("Inventory Level", "mean"),
            Proj_Demand=("Demand", "mean"),
            Suggested_Order=("Units Ordered", "mean")
        ).reset_index().nlargest(10, "Proj_Demand").round(0)
        at_risk["Risk"] = at_risk.apply(
            lambda r: "🔴 High" if r["Current_Stock"] < r["Proj_Demand"] * 0.3
            else "🟡 Medium" if r["Current_Stock"] < r["Proj_Demand"] * 0.7 else "🟢 Low", axis=1)
        at_risk.columns = ["Product", "Curr Stock", "Proj Demand (7D)", "Suggested Qty", "Risk"]
        st.dataframe(at_risk, use_container_width=True, height=290)

    with col4:
        st.markdown("#### 📅 Stockout Forecast (Next 7 Days)")
        prods = dff[dff["Stockout"]]["Product ID"].value_counts().head(6).index.tolist()
        so_days = np.random.default_rng(7).integers(1, 8, len(prods))
        so_df = pd.DataFrame({"Product": prods, "Days Until Stockout": so_days}).sort_values("Days Until Stockout")
        colors_so = ["#FF4C61" if d <= 2 else "#FF8C42" if d <= 4 else "#00C48C" for d in so_df["Days Until Stockout"]]
        fig4 = go.Figure(go.Bar(
            x=so_df["Days Until Stockout"], y=so_df["Product"], orientation='h',
            marker_color=colors_so,
            error_x=dict(type="data", array=[0.5]*len(so_df), color=FONT_CLR, thickness=1.5),
            text=[f"{d} days" for d in so_df["Days Until Stockout"]],
            textposition="outside", textfont=dict(color=FONT_CLR)
        ))
        dark_layout(fig4, "", height=290)
        fig4.update_xaxes(title_text="Days Until Stockout", range=[0, 11])
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.markdown("<div class='section-header'>Product Movement & Aging Analysis</div>", unsafe_allow_html=True)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown("#### 🚀 Fast Movers (Top 5)")
        fast = dff.groupby("Product ID")["Units Sold"].sum().nlargest(5).reset_index()
        fig5 = go.Figure(go.Bar(x=fast["Units Sold"], y=fast["Product ID"], orientation='h',
                                 marker_color="#6C63FF",
                                 text=fast["Units Sold"].apply(fmt_num),
                                 textposition="outside", textfont=dict(color=FONT_CLR)))
        dark_layout(fig5, "", height=260)
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown("#### 🐢 Slow Movers (Top 5)")
        slow = dff.groupby("Product ID")["Units Sold"].sum().nsmallest(5).reset_index()
        fig6 = go.Figure(go.Bar(x=slow["Units Sold"], y=slow["Product ID"], orientation='h',
                                 marker_color="#FF8C42",
                                 text=slow["Units Sold"].apply(fmt_num),
                                 textposition="outside", textfont=dict(color=FONT_CLR)))
        dark_layout(fig6, "", height=260)
        st.plotly_chart(fig6, use_container_width=True)

    with col7:
        st.markdown("#### 📦 Overstock Analysis")
        co = dff.groupby("Category").agg(Inventory=("Inventory Level", "mean"), Demand=("Demand", "mean")).reset_index()
        co["Overstock %"] = ((co["Inventory"] / co["Demand"]) - 1) * 100
        fig7 = go.Figure(go.Bar(
            x=co["Overstock %"], y=co["Category"], orientation="h",
            marker_color=["#FF4C61" if x > 100 else "#FF8C42" if x > 50 else "#00C48C" for x in co["Overstock %"]],
            text=[f"{x:.0f}%" for x in co["Overstock %"]],
            textposition="outside", textfont=dict(color=FONT_CLR)
        ))
        dark_layout(fig7, "", height=260)
        st.plotly_chart(fig7, use_container_width=True)

    with col8:
        st.markdown("#### 🕐 Inventory Aging")
        bins = ["0–30d", "31–60d", "61–90d", "91–120d", "120d+"]
        counts = [
            len(dff[dff["Coverage Days"] <= 30]),
            len(dff[(dff["Coverage Days"] > 30) & (dff["Coverage Days"] <= 60)]),
            len(dff[(dff["Coverage Days"] > 60) & (dff["Coverage Days"] <= 90)]),
            len(dff[(dff["Coverage Days"] > 90) & (dff["Coverage Days"] <= 120)]),
            len(dff[dff["Coverage Days"] > 120]),
        ]
        pcts = [round(c / sum(counts) * 100, 1) for c in counts]
        fig8 = go.Figure(go.Bar(
            x=bins, y=pcts,
            marker_color=["#00C48C", "#6C63FF", "#4EAEFF", "#FF8C42", "#FF4C61"],
            text=[f"{p}%" for p in pcts],
            textposition="outside", textfont=dict(color=FONT_CLR)
        ))
        dark_layout(fig8, "", height=260)
        fig8.update_yaxes(title_text="% of SKUs")
        st.plotly_chart(fig8, use_container_width=True)

    st.divider()
    st.markdown("<div class='section-header'>AI Inventory Insights</div>", unsafe_allow_html=True)
    summary = f""" 
Total Inventory Value: ${dff['Inventory Value'].sum():,.0f}
Stockout Rate: {dff['Stockout'].mean()*100:.1f}%
Most Stocked Out Product: {dff[dff['Stockout']]['Product ID'].value_counts().index[0]}
Most Overstocked Category: {dff.groupby('Category')['Inventory Level'].mean().idxmax()}
Avg Coverage Days: {dff['Coverage Days'].mean():.1f}
Lost Demand Units: {dff['Lost Demand'].sum():,.0f}
Est. Lost Revenue: ${(dff['Lost Demand'] * dff['Price']).sum():,.0f}
"""
    if st.button("✨ Generate AI Insights", key="gen_insights_warehouse"):
        st.session_state["show_insights_warehouse"] = True

    if st.session_state.get("show_insights_warehouse"):
        insights = get_ai_insights("warehouse", store_id=None, data_summary=summary) 
   
        for line in insights:
            icon = line[0] if line else "💡"
            text = line[2:].strip() if len(line) > 2 else line
            st.markdown(
                f'<div class="insight-card"><span style="font-size:18px;">{icon}</span><span>{text}</span></div>',
                unsafe_allow_html=True
                )
    ai_assistant_panel(page_key="warehouse")

    if st.button("← Back to Home", key="wh_back"):
        st.session_state.page = "home"; st.rerun()


# -----------------------
# BRANCH DASHBOARD
# -----------------------
def branch_dashboard():
    hc, sc, dc = st.columns([2, 2, 1])
    with hc:
        st.markdown("<h1 style='font-size:22px;'>🏬 Branch Manager Dashboard</h1>", unsafe_allow_html=True)
        st.caption("Store Performance & Operations Overview")
    with sc:
        store = st.selectbox("Select Store", sorted(df["Store ID"].unique()), key="branch_store")
        st.session_state["active_store_scope"] = store 
    with dc:
        date_filter = st.selectbox("Period", ["All Time", "Last 30 Days", "Last 7 Days"],
                                    label_visibility="collapsed", key="br_period")

    d = df[df["Store ID"] == store].copy()
    if date_filter == "Last 7 Days":
        d = d[d["Date"] >= d["Date"].max() - pd.Timedelta(days=7)]
    elif date_filter == "Last 30 Days":
        d = d[d["Date"] >= d["Date"].max() - pd.Timedelta(days=30)]

    promo_lift_br = ((d[d['Promotion']==1]['Revenue'].mean() /
                       max(d[d['Promotion']==0]['Revenue'].mean(), 1)) - 1) * 100

    st.markdown("<div class='section-header'>Store Performance Metrics</div>", unsafe_allow_html=True)
    kpi_row([
        ("Store Revenue",      fmt_money(d["Revenue"].sum()),                       "9.6% vs Yesterday",  "pos"),
        ("Units Sold",         fmt_num(d["Units Sold"].sum()),                       "13.2% vs Last Wk",  "pos"),
        ("Inventory Value",    fmt_money(d["Inventory Value"].sum()),               "4.5% vs Last Wk",   "pos"),
        ("Promo Impact",       f"+{promo_lift_br:.1f}%",                             "vs No Promotion",   "pos"),
        ("Demand Forecast 7D", fmt_num(d["Demand"].sum()),                           "6.7% vs Curr Wk",  "pos"),
        ("Lost Sales (SO)",    fmt_money(d["Lost Demand"].sum() * d["Price"].mean()), "8.4% vs Last Wk",  "pos"),
    ])

    st.divider()
    st.markdown("<div class='section-header'>Sales Performance</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        ts = d.groupby(d["Date"].dt.date)["Revenue"].sum().reset_index()
        ts.columns = ["Date", "Revenue"]
        ts["DateStr"] = ts["Date"].astype(str)
        fc, lo, hi = make_forecast_curve(ts["Revenue"], periods=7)
        fc_dates = [f"FC+{i+1}" for i in range(7)]
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=ts["DateStr"], y=ts["Revenue"], name="Actual",
                                   line=dict(color="#6C63FF", width=2.5),
                                   fill="tozeroy", fillcolor="rgba(108,99,255,0.1)",
                                   mode="lines+markers", marker=dict(size=4)))
        fig1.add_trace(go.Scatter(x=fc_dates, y=fc, name="Forecast",
                                   line=dict(color="#A78BFA", width=2, dash="dash"),
                                   mode="lines+markers", marker=dict(symbol="diamond", size=5)))
        fig1.add_trace(go.Scatter(
            x=fc_dates + fc_dates[::-1],
            y=list(hi) + list(lo[::-1]),
            fill="toself", fillcolor="rgba(167,139,250,0.1)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False))
        dark_layout(fig1, f"Daily Sales Trend & 7-Day Forecast — Store {store}", height=300, legend_h=True)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        cat_sales = d.groupby("Category")["Revenue"].sum().reset_index()
        total_cat = cat_sales["Revenue"].sum()
        fig2 = go.Figure(data=[go.Pie(
            labels=cat_sales["Category"], values=cat_sales["Revenue"],
            hole=0.55, marker_colors=PLOTLY_COLORS,
            textinfo="label+percent", textfont=dict(color="#E2E8F0", size=11)
        )])
        fig2.update_layout(
            annotations=[dict(text=f"<b>{fmt_money(total_cat)}</b>", x=0.5, y=0.5,
                               font_size=14, showarrow=False, font_color="#F1F5F9")],
            paper_bgcolor=PAPER_BG, margin=dict(l=10,r=10,t=40,b=0),
            height=300, font=dict(family="Inter", size=11, color=FONT_CLR),
            title=dict(text="Category Wise Sales (This Period)", font=dict(color=TITLE_CLR, size=13)),
            legend=dict(orientation="v", x=1.02, font=dict(color=FONT_CLR))
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        promo_perf = d.groupby(["Category", "Promotion"])["Revenue"].mean().reset_index()
        promo_perf["Type"] = promo_perf["Promotion"].map({0: "No Promotion", 1: "With Promotion"})
        fig3 = px.bar(promo_perf, x="Category", y="Revenue", color="Type", barmode="group",
                       color_discrete_map={"No Promotion": "#374151", "With Promotion": "#6C63FF"})
        dark_layout(fig3, "Promotion Performance by Category", height=280, legend_h=True)
        fig3.update_xaxes(tickangle=-20)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        d_daily = d.groupby(d["Date"].dt.date)["Demand"].sum().reset_index()
        d_daily.columns = ["Date", "Demand"]
        fc_d, lo_d, hi_d = make_forecast_curve(d_daily["Demand"], periods=7)
        hist_dates = [str(x) for x in d_daily["Date"].tail(14)]
        hist_vals  = d_daily["Demand"].tail(14).values
        fc_dates7  = [f"FC+{i+1}" for i in range(7)]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=hist_dates, y=hist_vals, name="Historical",
                                   line=dict(color="#4EAEFF", width=2.5),
                                   mode="lines+markers", marker=dict(size=4)))
        fig4.add_trace(go.Scatter(x=fc_dates7, y=fc_d, name="Forecast",
                                   line=dict(color="#7DD3FC", width=2, dash="dash"),
                                   mode="lines+markers", marker=dict(symbol="diamond", size=5)))
        fig4.add_trace(go.Scatter(
            x=fc_dates7 + fc_dates7[::-1],
            y=list(hi_d) + list(lo_d[::-1]),
            fill="toself", fillcolor="rgba(125,211,252,0.1)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False))
        dark_layout(fig4, "Demand Forecast (Next 7 Days)", height=280, legend_h=True)
        fig4.update_xaxes(tickangle=-30)
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        d_ts = d.groupby(d["Date"].dt.to_period("D")).agg(
            {"Inventory Value": "mean", "Demand": "sum"}
        ).reset_index()
        d_ts["Day"] = d_ts["Date"].astype(str)
        d_ts = d_ts.tail(14)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=d_ts["Day"], y=d_ts["Inventory Value"],
                                   name="Inventory Value", line=dict(color="#00C48C", width=2)))
        fig5.add_trace(go.Scatter(x=d_ts["Day"], y=d_ts["Demand"] * d["Price"].mean(),
                                   name="Demand Value", line=dict(color="#FF8C42", width=2, dash="dot")))
        dark_layout(fig5, "Inventory vs Demand Value", height=280, legend_h=True)
        fig5.update_xaxes(tickangle=-30)
        st.plotly_chart(fig5, use_container_width=True)

    st.divider()
    st.markdown("<div class='section-header'>Product & Operations Intelligence</div>", unsafe_allow_html=True)

    col6, col7, col8 = st.columns(3)
    with col6:
        st.markdown("#### 🏆 Top Selling Products")
        top = d.groupby("Product ID")["Units Sold"].sum().nlargest(8).reset_index()
        fig6 = go.Figure(go.Bar(x=top["Units Sold"], y=top["Product ID"], orientation='h',
                                 marker_color="#6C63FF",
                                 text=top["Units Sold"].apply(fmt_num),
                                 textposition="outside", textfont=dict(color=FONT_CLR)))
        dark_layout(fig6, "", height=310)
        st.plotly_chart(fig6, use_container_width=True)

    with col7:
        st.markdown("#### ⚠️ Low Stock Alerts")
        ls = d[d["Stockout"]][["Product ID", "Inventory Level", "Demand"]].copy()
        ls["Days Left"] = (ls["Inventory Level"] / np.maximum(ls["Demand"], 1)).round(1)
        ls["Status"] = ls["Days Left"].apply(
            lambda x: "🔴 Critical" if x < 2 else "🟡 Warning" if x < 5 else "🟢 OK")
        ls = ls.rename(columns={"Product ID": "Product", "Inventory Level": "Stock", "Demand": "Min Level"}).head(8)
        st.dataframe(ls, use_container_width=True, height=310)

    with col8:
        st.markdown("#### 🌦 Weather Impact Analysis")
        wd = d.groupby("Weather Condition").agg(Revenue=("Revenue", "sum"), Units=("Units Sold", "sum")).reset_index()
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(
            x=wd["Revenue"], y=wd["Units"],
            mode="markers+text",
            text=wd["Weather Condition"],
            textposition="top center",
            marker=dict(size=18, color=PLOTLY_COLORS[:len(wd)],
                         line=dict(width=1, color="rgba(255,255,255,0.3)")),
            textfont=dict(color=FONT_CLR)
        ))
        dark_layout(fig8, "Revenue vs Units by Weather", height=310)
        fig8.update_xaxes(title_text="Revenue")
        fig8.update_yaxes(title_text="Units Sold")
        st.plotly_chart(fig8, use_container_width=True)

    # ── STEP 2 (Branch): ML Forecast chart ───────────────────────────────────
    st.divider()
    render_forecast_chart(d)

    # ── STEP 3: Upgraded AI Insights with live MAE ───────────────────────────
    st.divider()
    st.markdown("<div class='section-header'>AI Store Recommendations</div>", unsafe_allow_html=True)
    summary = f"""
Total Inventory Value: ${d['Inventory Value'].sum():,.0f}
Stockout Rate: {d['Stockout'].mean()*100:.1f}%
Most Stocked Out Product: {d[d['Stockout']]['Product ID'].value_counts().index[0]}
Most Overstocked Category: {d.groupby('Category')['Inventory Level'].mean().idxmax()}
Avg Coverage Days: {d['Coverage Days'].mean():.1f}
Lost Demand Units: {d['Lost Demand'].sum():,.0f}
Est. Lost Revenue: ${(d['Lost Demand'] * d['Price']).sum():,.0f}
"""

    if st.button("✨ Generate AI Insights", key=f"gen_insights_{store}"):
        st.session_state[f"show_insights_{store}"] = True

    if st.session_state.get(f"show_insights_{store}"):
        insights = get_ai_insights(f"branch_{store}", store_id=store, data_summary=summary)
        for line in insights:
            icon = line[0] if line else "💡"
            text = line[2:].strip() if len(line) > 2 else line
            st.markdown(
                f'<div class="insight-card"><span style="font-size:18px;">{icon}</span><span>{text}</span></div>',
                unsafe_allow_html=True
            )

    st.markdown(f"<div class='section-header'>Ask AI About Store {store}</div>", unsafe_allow_html=True)
    ai_assistant_panel(page_key=f"branch_{store}")

    if st.button("← Back to Home", key="br_back"):
        st.session_state.page = "home"; st.rerun()


# -----------------------
# ROUTER
# -----------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

sidebar()

if st.session_state.page == "home":
    home()
elif st.session_state.page == "ceo":
    ceo_dashboard()
elif st.session_state.page == "warehouse":
    warehouse_dashboard()
elif st.session_state.page == "branch":
    branch_dashboard()