import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Retail Inventory Forecasting",
    layout="wide"
)

tab1, tab2, tab3 = st.tabs([
    "Dashboard",
    "Forecast",
    "AI Assistant"
])

# ---------------- Dashboard Tab ----------------
with tab1:
    st.title("Dashboard")

    # Fake data with all columns + product + season
    data = {
        "date": pd.date_range("2026-01-01", periods=30),
        "region": ["North", "South", "East", "West"] * 7 + ["North", "South"],
        "product": ["Apples", "Bananas", "Oranges", "Milk"] * 7 + ["Bread", "Eggs"],
        "sales": [100, 200, 150, 180] * 7 + [120, 210],
        "inventory": [500, 450, 600, 550] * 7 + [480, 470],
        "low_stock_pct": [0.1, 0.2, 0.05, 0.15] * 7 + [0.12, 0.18],
        "promotion": ["Yes", "No", "No", "Yes"] * 7 + ["No", "Yes"],
        "weather": ["Sunny", "Rainy", "Cloudy", "Sunny"] * 7 + ["Rainy", "Sunny"],
        "competitor": [90, 85, 70, 95] * 7 + [88, 92],
        "epidemic": [0, 0, 1, 0] * 7 + [0, 0],
        "season": ["Winter", "Spring", "Summer", "Autumn"] * 7 + ["Winter", "Spring"]
    }
    df = pd.DataFrame(data)

    # Sidebar filters
    st.sidebar.header("Filters")
    selected_region = st.sidebar.selectbox("Select Region", df["region"].unique())
    selected_date = st.sidebar.date_input("Select Date", df["date"].min())

    # Apply filters for KPIs/tables
    filtered_df = df.copy()
    if selected_region:
        filtered_df = filtered_df[filtered_df["region"] == selected_region]
    if selected_date:
        filtered_df = filtered_df[filtered_df["date"] == pd.to_datetime(selected_date)]

    # KPIs row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${filtered_df['sales'].sum():,.0f}")
    col2.metric("Total Inventory", f"{filtered_df['inventory'].sum():,.0f} units")
    col3.metric("Low Stock %", f"{filtered_df['low_stock_pct'].mean()*100:.1f}%")
    col4.metric("Competitor Index", f"{filtered_df['competitor'].mean():.0f}")

    # Regional Sales (all-time)
    st.subheader("Regional Sales Performance (All-Time)")
    fig_region = px.bar(df, x="region", y="sales", color="region")
    st.plotly_chart(fig_region, use_container_width=True)

    # Datewise Sales Trend (all-time)
    st.subheader("Datewise Sales Trend (All-Time)")
    fig_date = px.line(df, x="date", y="sales", color="region")
    st.plotly_chart(fig_date, use_container_width=True)

    # Two-column section
    left, right = st.columns(2)
    with left:
        st.subheader("Promotion Impact")
        fig_promo = px.bar(df, x="promotion", y="sales", color="region")
        st.plotly_chart(fig_promo, use_container_width=True)

    with right:
        st.subheader("Weather Conditions Impact")
        fig_weather = px.bar(df, x="weather", y="sales", color="region")
        st.plotly_chart(fig_weather, use_container_width=True)

    # Competitor vs Sales
    st.subheader("Competitor vs Sales")
    fig_comp = px.scatter(df, x="competitor", y="sales", color="region", title="Competitor Index vs Sales")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Product-wise Inventory
    st.subheader("Product-wise Inventory Levels")
    fig_prod_inv = px.bar(df, x="product", y="inventory", color="region", title="Inventory by Product and Region")
    st.plotly_chart(fig_prod_inv, use_container_width=True)

    # Seasonal Reports
    st.subheader("Seasonal Sales & Inventory Report")
    fig_season = px.bar(df, x="season", y="sales", color="region", title="Sales by Season")
    st.plotly_chart(fig_season, use_container_width=True)

    fig_season_inv = px.bar(df, x="season", y="inventory", color="region", title="Inventory by Season")
    st.plotly_chart(fig_season_inv, use_container_width=True)

    # Low Stock Products
    st.subheader("Low Stock Products")
    low_stock_df = df[df["low_stock_pct"] > 0.15][["date", "region", "product", "inventory", "low_stock_pct"]]
    st.dataframe(low_stock_df)

    # Alerts
    st.subheader("Alerts")
    alerts = []
    if filtered_df['low_stock_pct'].mean() > 0.15:
        alerts.append("⚠️ Average low stock percentage is high.")
    for alert in alerts:
        st.warning(alert)

# ---------------- Forecasting Tab ----------------
with tab2:
    st.title("Demand Forecasting")

    # Forecast settings (only visible in tab2)
    st.subheader("Forecast Settings")
    use_promotion = st.checkbox("Include Promotion Data", value=True)

    # Fake forecast data (replace with forecasting.py outputs later)
    forecast_data = {
        "date": pd.date_range("2026-02-01", periods=15),
        "forecast_sales": [120, 130, 140, 150, 160, 170, 180, 175, 165, 155, 145, 135, 125, 115, 110]
    }
    forecast_df = pd.DataFrame(forecast_data)

    # Apply toggles (for now just annotate, later pass to model)
    factors = []
    if use_promotion:
        factors.append("Promotion")

    st.write(f"Forecast generated with factors: {', '.join(factors) if factors else 'None'}")

    # Forecast chart
    st.subheader("Forecasted Demand (Next 15 Days)")
    fig_forecast = px.line(forecast_df, x="date", y="forecast_sales", title="Demand Forecast")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Forecast table
    st.subheader("Forecast Data Table")
    st.dataframe(forecast_df)

# ---------------- AI Assistant Tab ----------------
with tab3:
    st.title("AI Assistant")

    st.write("Ask questions about your sales, inventory, and external factors (promotion, competitor, epidemic, season).")

    # Input box for user query
    user_query = st.text_input("Enter your question:")
