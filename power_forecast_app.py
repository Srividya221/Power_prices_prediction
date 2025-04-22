import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib

# Page settings
st.set_page_config(page_title="Power Trading Forecast", layout="wide")

# Sidebar for DB credentials (optional)
st.sidebar.header("Database Credentials (Optional)")
db_user = st.sidebar.text_input("User", value="your_user")
db_password = st.sidebar.text_input("Password", value="your_password", type="password")
db_host = st.sidebar.text_input("Host", value="localhost")
db_port = st.sidebar.text_input("Port", value="5432")
db_name = st.sidebar.text_input("Database Name", value="power_trading_db")

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\navya\Downloads\xg_model_files\daily_iex_data.csv")
    df = df.rename(columns={
        'Purchase Bid (MWh)': 'Purchase_Bid',
        'Sell Bid (MWh)': 'Sell_Bid',
        'MCV (MWh)': 'MCV',
        'Final Scheduled Volume (MWh)': 'Final_Scheduled_Volume',
        'MCP (Rs/MWh) *': 'MCP',
        'Weighted MCP (Rs/MWh)': 'Weighted_MCP'
    })
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    train = pd.read_csv(r"C:\Users\navya\Downloads\xg_model_files\Train.csv", parse_dates=['Date'], index_col='Date')
    test = pd.read_csv(r"C:\Users\navya\Downloads\xg_model_files\Test.csv", parse_dates=['Date'], index_col='Date')

    return df, train, test

@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\navya\Downloads\xg_model_files\xgboost_best_model.pkl")

df, Train, Test = load_data()
best_model = load_model()

# Header
st.title("🔌 Power Trading Forecast - Weighted MCP")
st.write("Forecasting next 30 days using lag features and XGBoost with 90% confidence interval.")

# Settings
forecast_days = 30
lags = 7
np.random.seed(42)  # Optional: set global seed for reproducibility

# Prepare data
for data in [df, Train, Test]:
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

# Read feature columns from saved X_train
X_train = pd.read_csv(r"C:\Users\navya\Downloads\xg_model_files\X_train.csv")
feature_cols = [col for col in X_train.columns if col != 'Date']

# Forecast setup
last_date = df.index.max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
forecast_series = pd.Series(index=future_dates, dtype='float64')
lower_bound = pd.Series(index=future_dates, dtype='float64')
upper_bound = pd.Series(index=future_dates, dtype='float64')

history = df.copy().iloc[-(lags + 30):].copy() 
 # Keep enough data for lagging
 

# Dynamic Forecast Loop
for date in future_dates:
    new_row = {}
    
    # Simulate new values based on the last day's data
    last = history.iloc[-1]
    new_row['Purchase_Bid'] = last['Purchase_Bid'] + np.random.normal(0, 5)
    new_row['Sell_Bid'] = last['Sell_Bid'] + np.random.normal(0, 5)
    new_row['MCV'] = last['MCV'] + np.random.normal(0, 10)
    new_row['Final_Scheduled_Volume'] = last['Final_Scheduled_Volume'] + np.random.normal(0, 8)
    new_row['MCP'] = last['MCP'] + np.random.normal(0, 2)

    # Create lag features from history
    for col in ['Weighted_MCP', 'Purchase_Bid', 'Sell_Bid']:
        for lag in range(1, lags + 1):
            lag_date = date - pd.Timedelta(days=lag)
            if lag_date in history.index:
                new_row[f"{col}_lag_{lag}"] = history.loc[lag_date, col]
            else:
                new_row[f"{col}_lag_{lag}"] = np.nan

    # Create input for prediction
    new_X = pd.DataFrame([new_row], index=[date])

    # Skip prediction if missing lag data
    if new_X[feature_cols].isnull().any().any():
        continue

    bootstrap_preds = []
    for _ in range(100):  # 100 bootstrapped predictions
        noisy_X = new_X[feature_cols].copy()
        # Add slight noise to numeric lag features only
        for col in noisy_X.columns:
            noisy_X[col] += np.random.normal(0, 0.01 * noisy_X[col].mean()) if noisy_X[col].mean() != 0 else 0
        pred = best_model.predict(noisy_X)[0]
        bootstrap_preds.append(pred)

    pred_mean = np.mean(bootstrap_preds)
    forecast_series[date] = pred_mean
    lower_bound[date] = np.percentile(bootstrap_preds, 5)
    upper_bound[date] = np.percentile(bootstrap_preds, 95)


    # Update history recursively with predicted and simulated values
    history.loc[date] = {
        'Weighted_MCP': pred_mean,
        'Purchase_Bid': new_row['Purchase_Bid'],
        'Sell_Bid': new_row['Sell_Bid'],
        'MCV': new_row['MCV'],
        'Final_Scheduled_Volume': new_row['Final_Scheduled_Volume'],
        'MCP': new_row['MCP']
    }
 

# Save forecast history
history.to_csv("future_predictions.csv")

# Plotting
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index, df['Weighted_MCP'], label='Actual', color='gray')
ax.plot(Train.index, Train['Weighted_MCP'], label='Train', color='blue')
ax.plot(Test.index, Test['Weighted_MCP'], label='Test', color='orange')
ax.plot(forecast_series.index, forecast_series, label='Forecast (Next 30 days)', color='green', linestyle='--')
ax.fill_between(forecast_series.index, lower_bound, upper_bound, color='green', alpha=0.2, label='90% Confidence Interval')
ax.axvline(x=df.index.max(), color='red', linestyle=':', label='Forecast Start')
ax.set_title("Weighted MCP Forecast with 90% Confidence Interval")
ax.set_xlabel("Date")
ax.set_ylabel("Weighted_MCP")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Table
st.subheader("📈 Forecast Table (Next 30 Days)")
forecast_df = pd.DataFrame({
    'Forecast': forecast_series,
    'Lower_90%': lower_bound,
    'Upper_90%': upper_bound
}).dropna()
st.dataframe(forecast_df.style.format("{:.2f}"))

# Download Button
st.download_button(
    label="📥 Download Forecast CSV",
    data=forecast_df.to_csv().encode('utf-8'),
    file_name='forecast_next_30_days.csv',
    mime='text/csv'
)


