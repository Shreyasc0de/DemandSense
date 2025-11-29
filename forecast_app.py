import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error

# ------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------------
st.set_page_config(page_title="DemandSense AI", layout="wide")

st.title("🍦 DemandSense: AI Inventory Optimizer")
st.markdown("""
**Role:** Senior Supply Chain Analyst.  
**Goal:** Predict daily inventory demand based on historical sales, weekly seasonality, and weather forecasts.
""")

# ------------------------------------------------------------------------
# 2. INTELLIGENT DATA GENERATION
# ------------------------------------------------------------------------
@st.cache_data
def generate_data():
    # Create a date range for 2 years
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    n = len(dates)
    
    # 1. Base Trend (Sales slightly increasing over time)
    trend = np.linspace(50, 100, n)
    
    # 2. Seasonality (Weekends are busy)
    # 0=Mon, 6=Sun. Multiplier: 1.4x on Sat/Sun, 1.0x on weekdays
    weekly_seasonality = np.where(dates.dayofweek >= 5, 1.4, 1.0)
    
    # 3. Weather Impact (Temperature)
    # Simulate temp: Cold in Jan, Hot in July
    # Sine wave centered at 65 degrees, swinging +/- 25 degrees
    temp_pattern = 65 + 25 * np.sin(2 * np.pi * dates.dayofyear / 365)
    # Add randomness to weather
    temp = temp_pattern + np.random.normal(0, 5, n)
    
    # Sales increase by 2 units for every degree over 60F
    weather_effect = np.maximum(0, (temp - 60) * 2)
    
    # Final Sales Calculation + Random Noise
    sales = (trend * weekly_seasonality) + weather_effect + np.random.normal(0, 10, n)
    
    # Create DataFrame
    df = pd.DataFrame({'ds': dates, 'y': sales, 'temp': temp})
    
    # Ensure no negative sales
    df['y'] = df['y'].clip(lower=0)
    
    return df

df = generate_data()

# ------------------------------------------------------------------------
# 3. PROPHET MODEL TRAINING
# ------------------------------------------------------------------------
@st.cache_resource
def train_model(data):
    # Initialize Prophet with a custom regressor
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.add_regressor('temp')
    
    # Fit the model
    m.fit(data)
    return m

model = train_model(df)

# ------------------------------------------------------------------------
# 4. DASHBOARD SIDEBAR (SCENARIO PLANNING)
# ------------------------------------------------------------------------
st.sidebar.header("🔮 Scenario Planner")
st.sidebar.write("Adjust the weather forecast for the next 30 days:")

expected_temp = st.sidebar.slider("Avg Expected Temperature (°F)", min_value=10, max_value=100, value=75)

# Calculate MAE for Sidebar
y_true = df['y']
y_pred = model.predict(df)['yhat']
mae = mean_absolute_error(y_true, y_pred)
st.sidebar.markdown("---")
st.sidebar.metric("Model Error (MAE)", f"{mae:.1f} units")

# ------------------------------------------------------------------------
# 5. FUTURE FORECASTING
# ------------------------------------------------------------------------
# Create future dataframe for 30 days
future = model.make_future_dataframe(periods=30)

# We need to fill the 'temp' column for the future dates
# For historical rows, use actual temp. For future rows, use the Slider value.
future['temp'] = df['temp'].tolist() + [expected_temp] * 30

# Predict
forecast = model.predict(future)

# ------------------------------------------------------------------------
# 6. VISUALIZATION & BUSINESS LOGIC
# ------------------------------------------------------------------------

# Section 1: Inventory & Stockout Logic
st.subheader("📦 Inventory Run-Out Simulator")

col_a, col_b = st.columns([1, 2])

with col_a:
    # User Input: How much stock do we have right now?
    current_stock = st.number_input(
        "Current Warehouse Stock (Units)",
        min_value=0,
        value=2000,
        step=100,
        help="Enter the amount of inventory you currently have on hand."
    )

with col_b:
    # 1. Get the forecast for the next 30 days
    forecast_30 = forecast.tail(30).copy()
    # 2. Calculate Cumulative Demand (Day 1 + Day 2 + Day 3...)
    forecast_30['cumulative_demand'] = forecast_30['yhat'].cumsum()
    # 3. Calculate Remaining Stock day-by-day
    forecast_30['remaining_stock'] = current_stock - forecast_30['cumulative_demand']
    # 4. Find the EXACT date we go below 0
    stockout_row = forecast_30[forecast_30['remaining_stock'] < 0]
    
    if not stockout_row.empty:
        # Get the first date where we run out
        run_out_date = pd.to_datetime(stockout_row.iloc[0]['ds']).strftime('%B %d, %Y')
        st.error(f"⚠️ **CRITICAL ALERT:** Based on this weather forecast, you will run out of stock on **{run_out_date}**.")
    else:
        st.success(f"✅ **Safe:** Stock levels are sufficient for the next 30 days. You will have {int(forecast_30.iloc[-1]['remaining_stock'])} units left.")

st.markdown("---")

# Section 2: Interactive Chart
st.subheader("📈 Forecast vs. Historicals")
st.markdown("Drag the slider on the left to see how weather changes the forecast (the blue line at the end).")

# Plotly Chart
fig = go.Figure()

# Actual Sales (Black dots)
fig.add_trace(go.Scatter(
    x=df['ds'], y=df['y'], mode='markers', name='Actual Sales',
    marker=dict(color='black', size=4, opacity=0.5)
))

# Predicted Trend (Blue Line)
fig.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast',
    line=dict(color='#007BFF', width=2)
))

# Confidence Interval (Upper/Lower bounds)
fig.add_trace(go.Scatter(
    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
    y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
    fill='toself', fillcolor='rgba(0,123,255,0.2)', line=dict(color='rgba(255,255,255,0)'),
    name='Confidence Interval'
))

fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Units Sold")
st.plotly_chart(fig, use_container_width=True)

# Section 3: Model Components
with st.expander("🔎 View Model Components (Seasonality & Weather Impact)"):
    st.write("This shows how Day of Week and Temperature affect sales separately.")
    fig2 = plot_plotly(model, forecast)
    st.plotly_chart(fig2)

# Section 4: Export
st.subheader("💾 Export Analysis")
csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'temp']].to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Forecast to CSV",
    data=csv,
    file_name='demand_forecast.csv',
    mime='text/csv',
)