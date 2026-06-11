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
    st.title("Inventory & Demand Dashboard")

    # Fake data with category + demand forecast
    data = {
        "date": pd.date_range("2026-01-01", periods=30),
        "region": ["North", "South", "East", "West"] * 7 + ["North", "South"],
        "product": ["Apples", "Bananas", "Oranges", "Milk"] * 7 + ["Bread", "Eggs"],
        "category": ["Fruits", "Fruits", "Fruits", "Dairy"] * 7 + ["Bakery", "Dairy"],
        "inventory": [500, 450, 600, 550] * 7 + [480, 470],
        "low_stock_pct": [0.1, 0.2, 0.05, 0.15] * 7 + [0.12, 0.18],
        "promotion": ["Yes", "No", "No", "Yes"] * 7 + ["No", "Yes"],
        "weather": ["Sunny", "Rainy", "Cloudy", "Sunny"] * 7 + ["Rainy", "Sunny"],
        "competitor": [90, 85, 70, 95] * 7 + [88, 92],
        "season": ["Winter", "Spring", "Summer", "Autumn"] * 7 + ["Winter", "Spring"],
        "demand_forecast": [120, 130, 140, 150, 160, 170, 180, 175, 165, 155, 145, 135, 125, 115, 110] * 2
    }
    df = pd.DataFrame(data)

   # Sidebar filters
    st.sidebar.header("Filters")

    # Region multiselect
    selected_regions = st.sidebar.multiselect("Select Regions", df["region"].unique())

    # Date filter
    selected_date = st.sidebar.date_input("Select Date", df["date"].min())

    # Mutually exclusive filters for category vs product
    selected_products = st.sidebar.multiselect("Select Products", df["product"].unique())
    if selected_products:
        selected_categories = []  # disable categories if products chosen
    else:
        selected_categories = st.sidebar.multiselect("Select Categories", df["category"].unique())

    # Apply filters
    filtered_df = df.copy()
    if selected_regions:
        filtered_df = filtered_df[filtered_df["region"].isin(selected_regions)]
    if selected_date:
        filtered_df = filtered_df[filtered_df["date"] == pd.to_datetime(selected_date)]
    if selected_products:
        filtered_df = filtered_df[filtered_df["product"].isin(selected_products)]
    elif selected_categories:
        filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]

    # KPIs row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Inventory", f"{filtered_df['inventory'].sum():,.0f} units")
    col2.metric("Low Stock %", f"{filtered_df['low_stock_pct'].mean()*100:.1f}%")
    col3.metric("Forecasted Demand", f"{filtered_df['demand_forecast'].mean():,.0f} units")

    # Inventory vs Demand Trend (with filters)
    st.subheader("Inventory vs Demand Trend")
    fig_demand_inv = px.line(filtered_df, x="date", y=["inventory", "demand_forecast"],
                             title="Inventory vs Forecasted Demand")
    st.plotly_chart(fig_demand_inv, use_container_width=True)

    # Seasonal Inventory & Demand Report
    st.subheader("Seasonal Inventory & Demand Report")
    fig_season = px.bar(df, x="season", y=["inventory", "demand_forecast"], barmode="group",
                        title="Seasonal Inventory vs Demand")
    st.plotly_chart(fig_season, use_container_width=True)

    # Demand by Promotion
    st.subheader("Demand by Promotion")
    fig_demand_promo = px.bar(df, x="promotion", y="demand_forecast", color="region",
                              title="Forecasted Demand by Promotion")
    st.plotly_chart(fig_demand_promo, use_container_width=True)

    # Demand by Weather
    st.subheader("Demand by Weather")
    fig_demand_weather = px.bar(df, x="weather", y="demand_forecast", color="region",
                                title="Forecasted Demand by Weather")
    st.plotly_chart(fig_demand_weather, use_container_width=True)

    # Low Stock Products
    st.subheader("Low Stock Products")
    low_stock_df = df[df["low_stock_pct"] > 0.15][["date", "region", "product", "inventory", "low_stock_pct"]]
    st.dataframe(low_stock_df)

    # Alerts
    st.subheader("Alerts")
    alerts = []
    if filtered_df['low_stock_pct'].mean() > 0.15:
        alerts.append("⚠️ Average low stock percentage is high.")
    if (filtered_df['inventory'] < filtered_df['demand_forecast']).any():
        alerts.append("⚠️ Inventory may not meet forecasted demand.")
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
