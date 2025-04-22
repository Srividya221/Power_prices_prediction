'''Business Probem : In Power Trading Price volatility and demand-supply imbalances creates
 signigicant financial risks for market participants.The challenge lies in integrating
 multiple data sources,handling market uncertainities and optimizing tarding startegies  
 in real time.

 Business Objective : Minimize financial and procurement costs.
 Business Constraint : Maximize trading returns
 
 ##### Success Criteria #####
 Business Success Criteria : Atleast 15-20% reduction in financial risks due to price volatility,
 leading to increased profitability for power traders and Utilities.
 ML Success critetia : Acheive a Mean Absolute Percentage Error(MAPE) below 10% compared to
 baseline models.
 Economic Success Criteria: Optimized trading strategies leading to 5-10% savings in Energy 
 Procurement Costs and a 10-15% increase in returns from power Trading. '''
# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')
import joblib


# Load the data
df = pd.read_csv( r"C:\Users\navya\Downloads\daily_iex_data.csv")

# Parse date and sort
df['Date'] = pd.to_datetime(df['Date'] , format = '%d-%m-%Y')
df.set_index('Date', inplace=True)
df = df.sort_values('Date')


# Change the column name
df =df.rename(columns={'Purchase Bid (MWh)':'Purchase_Bid',
                         'Sell Bid (MWh)': 'Sell_Bid',
                         'MCV (MWh)' : 'MCV',
                         'Final Scheduled Volume (MWh)':'Final_Scheduled_Volume',
                         'MCP (Rs/MWh) *':'MCP',
                         'Weighted MCP (Rs/MWh)' : 'Weighted_MCP'})
# Function to drop duplicates
def drop_duplicates(df):
    return df[~df.index.duplicated(keep='first')]

# Function to forward fill missing values
def forward_fill(df):
    return df.ffill()

# Combine into a preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[
    ('drop_duplicates', FunctionTransformer(drop_duplicates)),
    ('forward_fill', FunctionTransformer(forward_fill))
])

# Apply preprocessing pipeline
df = preprocessing_pipeline.fit_transform(df)

# Save the pipeline
joblib.dump(preprocessing_pipeline, 'iex_preprocessing_pipeline.pkl')


# Target variable
target = 'Weighted_MCP'

# Create lag features
def create_lags(df, columns, lags):
    for col in columns:
        for lag in range(1, lags + 1):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df

lags = 7
columns_to_lag = ['Weighted_MCP', 'Purchase_Bid', 'Sell_Bid']
df = create_lags(df, columns_to_lag, lags)
df
# Drop missing values caused by lagging
df.dropna(inplace=True)

# Train-test split 
Train = df[df.index < df.index.max() - pd.DateOffset(years=1)]
Test = df[df.index >= df.index.max() - pd.DateOffset(years=1)]

X_train = Train.drop(columns=[target])
y_train = Train[target]
X_test = Test.drop(columns=[target])
y_test = Test[target]

# Define XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
'''Parameter	Default	Description
eta or learning_rate	0.3	Step size shrinkage used to prevent overfitting. Lower = slower but more accurate.
max_depth	6	Maximum depth of trees. Higher = more complex model, but can overfit.
min_child_weight	1	Minimum sum of instance weight (Hessian) needed in a child node. Controls complexity.
gamma	0	Minimum loss reduction required to make a split. Higher = more conservative.
max_delta_step	0	Limits weight updates. Use positive values (like 1) for imbalanced classes.
subsample	1	Fraction of data rows used to grow each tree. Reduces overfitting.
colsample_bytree	1	Fraction of columns used to build each tree.
colsample_bylevel	1	Fraction of columns used at each level (layer) of the tree.
colsample_bynode	1	Fraction of columns used at each node (split) in the tree.
lambda (L2)	1	L2 regularization term on weights. Prevents overfitting.
alpha (L1)	0	L1 regularization term on weights. Can lead to feature selection.
tree_method	'auto'	Method used for tree construction. Options: 'auto', 'exact', 'approx', 'hist', 'gpu_hist'.
scale_pos_weight	1	Controls balance of positive and negative weights — useful for imbalanced data.'''


# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}
from sklearn.model_selection import TimeSeriesSplit 
tscv = TimeSeriesSplit(n_splits=5)

# Grid Search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           cv=tscv, scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
best_model.get_params()
# Predict on test data
y_pred = best_model.predict(X_test)

# Evaluate
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Test MAPE: {mape:.4f}")

y_test_pred = best_model.predict(X_test)
y_train_pred = best_model.predict(X_train)
print("Test_xgd_MAPE: ", mean_absolute_percentage_error(y_test_pred, y_test))
print("Train_xgd_MAPE: ", mean_absolute_percentage_error(y_train_pred, y_train))

# Save the best model
joblib.dump(best_model, 'xgboost_model_iex.pkl')


# Combine predictions and actual values into a single DataFrame for plotting
plot_df = pd.DataFrame(index=df.index)
plot_df['Actual'] = df[target]
plot_df['Predicted'] = np.nan

# Fill in predictions for train and test sets
plot_df.loc[X_train.index, 'Predicted'] = y_train_pred
plot_df.loc[X_test.index, 'Predicted'] = y_test_pred

# Plot
plt.figure(figsize=(15, 6))
sns.lineplot(data=plot_df[['Actual', 'Predicted']], linewidth=2)

# Vertical line to mark train-test split
split_date = X_test.index.min()
plt.axvline(x=split_date, color='red', linestyle='--', label='Train/Test Split')

# Add labels and legend
plt.title('Actual vs Predicted Weighted MCP (XGBoost)')
plt.xlabel('Date')
plt.ylabel('Weighted MCP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


########### Forecasting for next 30 days ###############
print("End date of test data:", Test.index.max().date())
print(best_model.get_booster().feature_names)


# ---------------- Forecast Next 30 Days ---------------- #

# Parameters
forecast_days = 30
lags = 7

# 1. Last known date and data
last_date = df.index.max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

# 2. Copy the last 60 days to get enough lag history
history = df.copy().iloc[-60:].copy()

# 3. Get original features used during training
feature_cols = X_train.columns.tolist()

# 4. Create an empty DataFrame to store predictions
forecast_series = pd.Series(index=future_dates, dtype='float64')

# 5. Forecast one step at a time
# Forecast loop
for date in future_dates:
    new_row = {}
    
    # Include required non-lag features (using assumptions or last known values)
    new_row['Purchase_Bid'] = history['Purchase_Bid'].iloc[-1]
    new_row['Sell_Bid'] = history['Sell_Bid'].iloc[-1]
    new_row['MCV'] = history['MCV'].iloc[-1]
    new_row['Final_Scheduled_Volume'] = history['Final_Scheduled_Volume'].iloc[-1]
    new_row['MCP'] = history['MCP'].iloc[-1]

    # Generate lag features
    for col in ['Weighted_MCP', 'Purchase_Bid', 'Sell_Bid']:
        for lag in range(1, lags + 1):
            lag_date = date - pd.Timedelta(days=lag)
            if lag_date in history.index:
                new_row[f"{col}_lag_{lag}"] = history.loc[lag_date, col]
            else:
                new_row[f"{col}_lag_{lag}"] = np.nan
    
    # Convert to DataFrame
    new_X = pd.DataFrame([new_row], index=[date])
    
    # Drop rows with NaN (in case of insufficient lag history)
    if new_X.isnull().any().any():
        continue

    # Predict and store forecast
    forecast_value = best_model.predict(new_X)[0]
    forecast_series.loc[date] = forecast_value

    # Update history with prediction for next step
    history.loc[date] = {
        'Weighted_MCP': forecast_value,
        'Purchase_Bid': new_row['Purchase_Bid'],
        'Sell_Bid': new_row['Sell_Bid'],
        'MCV': new_row['MCV'],
        'Final_Scheduled_Volume': new_row['Final_Scheduled_Volume'],
        'MCP': new_row['MCP']
    }
forecast_series.info()
history.info()
history.to_csv("future predictions.csv")


import matplotlib.pyplot as plt
import seaborn as sns


# Plot everything
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Weighted_MCP'], label='Actual')
plt.plot(Train.index, Train['Weighted_MCP'], label='Train', color='blue')
plt.plot(Test.index, Test['Weighted_MCP'], label='Test', color='orange')
plt.plot(forecast_series.index, forecast_series, label='Forecast (Next 30 days)', color='green', linestyle='--')

# Optional: Show where forecast starts
plt.axvline(x=df.index.max(), color='gray', linestyle=':', label='Forecast Start')

plt.title("Weighted MCP Forecast")
plt.xlabel("Date")
plt.ylabel("Weighted_MCP")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()




    











