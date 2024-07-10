import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import warnings
import os
import time
from scipy import stats

def plot_historical_trend(temp_data):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(temp_data.index, temp_data['LandAverageTemperature'], label='Temperature')
    
    # Add trend line
    z = np.polyfit(range(len(temp_data)), temp_data['LandAverageTemperature'], 1)
    p = np.poly1d(z)
    ax.plot(temp_data.index, p(range(len(temp_data))), "r--", label='Trend')
    
    ax.set_title('Historical Temperature Trend', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_figure(fig, 'historical_trend.png')

def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = results_df['Model']
    x_pos = np.arange(len(x))
    width = 0.35
    
    ax.bar(x_pos - width/2, results_df['R2'], width, label='R2 Score', color='skyblue')
    ax.bar(x_pos + width/2, results_df['MSE'], width, label='MSE', color='lightgreen')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, 'model_comparison.png')

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Temperature (°C)', fontsize=12)
    ax.set_ylabel('Predicted Temperature (°C)', fontsize=12)
    ax.set_title(f'Actual vs Predicted Temperature ({model_name})', fontsize=16)
    plt.tight_layout()
    save_figure(fig, f'actual_vs_predicted_{model_name}.png')

def plot_residual_analysis(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(residuals, bins=30)
    ax1.set_xlabel('Residuals')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of Residuals')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title("Normal Q-Q plot")
    
    fig.suptitle(f'Residual Analysis for {model_name}', fontsize=16)
    plt.tight_layout()
    save_figure(fig, f'residual_analysis_{model_name}.png')

def plot_seasonal_patterns(temp_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Month', y='LandAverageTemperature', data=temp_data, ax=ax)
    ax.set_title('Temperature Distribution by Month', fontsize=16)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    plt.tight_layout()
    save_figure(fig, 'seasonal_patterns.png')

def save_figure(fig, filename):
    fig.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

warnings.filterwarnings('ignore')

print("Script started.")

# paths
input_csv_path = 'C:\\Users\\baqer\\Desktop\\VS coder\\GlobalTemperatures.csv'  
output_csv_path = 'improved_temperature_analysis_results.csv'
plots_dir = 'improved_temperature_analysis_plots'
os.makedirs(plots_dir, exist_ok=True)

print("Directories set up.")

# plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("deep")



print("Loading data...")
# Load and preprocess data
df = pd.read_csv(input_csv_path)
df['dt'] = pd.to_datetime(df['dt'])
df.set_index('dt', inplace=True)

# Focusing on LandAverageTemperature
temp_data = df[['LandAverageTemperature']].dropna()

print("Data loaded and preprocessed.")

print("Starting EDA...")
# EDA
print(temp_data.describe())

print("Generating time series plot...")
# Time series plot
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(temp_data.index, temp_data['LandAverageTemperature'])
ax.set_title('Land Average Temperature Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (°C)')
ax.xaxis.set_major_locator(mdates.YearLocator(20))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()
save_figure(fig, 'land_avg_temp_time_series.png')

print("Performing seasonal decomposition...")
# Seasonal decomposition
result = seasonal_decompose(temp_data['LandAverageTemperature'], model='additive', period=12)
fig = result.plot()
fig.set_size_inches(15, 10)
fig.suptitle('Seasonal Decomposition of Land Average Temperature')
save_figure(fig, 'seasonal_decomposition.png')

print("EDA completed.")

print("Performing feature engineering...")
# Feature engineering
temp_data['Year'] = temp_data.index.year
temp_data['Month'] = temp_data.index.month
temp_data['Season'] = temp_data.index.month.map({1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 
                                                 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 
                                                 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'})

print("Splitting data...")
# Split data
train_data = temp_data[:'2010-12-31']
test_data = temp_data['2011-01-01':]

X_train = train_data[['Year', 'Month']]
y_train = train_data['LandAverageTemperature']
X_test = test_data[['Year', 'Month']]
y_test = test_data['LandAverageTemperature']

print("Training Linear Regression model with polynomial features...")
# Linear Regression with polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_poly, y_train)
lr_pred = lr_model.predict(X_test_poly)

lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print("Training Random Forest model with hyperparameter tuning...")
# Random Forest with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'oob_score': [True, False],
    'random_state': [42],
    'criterion': ['squared_error', 'absolute_error', 'poisson'],
    'max_leaf_nodes': [None, 10, 20],
    'min_impurity_decrease': [0.0, 0.1],
    'ccp_alpha': [0.0, 0.1]
}

rf_base = RandomForestRegressor()
rf_grid = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=5, 
                       scoring=make_scorer(r2_score), n_jobs=-1)
rf_grid.fit(X_train, y_train)

print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best R2 score: {rf_grid.best_score_:.4f}")

rf_model = rf_grid.best_estimator_
rf_pred = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("Finding optimal SARIMA parameters...")
# SARIMA
# Finding optimal SARIMA parameters
start_time = time.time()
try:
    auto_arima_model = auto_arima(train_data['LandAverageTemperature'], seasonal=True, m=12,
                                  suppress_warnings=True, error_action="ignore", trace=False,
                                  stepwise=True, max_order=5, max_p=2, max_q=2, max_d=1,
                                  max_P=1, max_Q=1, max_D=1)
    print(f"Best SARIMA parameters: {auto_arima_model.order} {auto_arima_model.seasonal_order}")
except Exception as e:
    print(f"Auto ARIMA failed: {str(e)}")
    auto_arima_model = None
print(f"Time taken for auto_arima: {time.time() - start_time:.2f} seconds")

print("Fitting SARIMA model...")
# Fit SARIMA model
sarima_success = False
if auto_arima_model is not None:
    try:
        sarima_model = SARIMAX(train_data['LandAverageTemperature'], 
                               order=auto_arima_model.order, 
                               seasonal_order=auto_arima_model.seasonal_order)
        sarima_results = sarima_model.fit(disp=False)
        sarima_success = True
    except Exception as e:
        print(f"SARIMA model fitting failed: {str(e)}")

if not sarima_success:
    print("Trying simpler SARIMA model...")
    try:
        sarima_model = SARIMAX(train_data['LandAverageTemperature'], order=(1,1,1), seasonal_order=(1,1,1,12))
        sarima_results = sarima_model.fit(disp=False)
        sarima_success = True
    except Exception as e:
        print(f"Simpler SARIMA model fitting also failed: {str(e)}")

print("Making SARIMA predictions...")
# Make predictions
if sarima_success:
    sarima_pred = sarima_results.forecast(steps=len(test_data))
    sarima_mse = mean_squared_error(test_data['LandAverageTemperature'], sarima_pred)
    sarima_r2 = r2_score(test_data['LandAverageTemperature'], sarima_pred)
else:
    print("SARIMA modeling failed. Proceeding without SARIMA results.")
    sarima_pred = None
    sarima_mse = None
    sarima_r2 = None

print("Generating prediction plots...")
# Plot predictions
fig, ax = plt.subplots(figsize=(15, 8))

# Focus on the last 20 years of historical data
focus_start = test_data.index[0] - pd.DateOffset(years=10)
ax.plot(train_data.loc[focus_start:].index, train_data.loc[focus_start:, 'LandAverageTemperature'], label='Training Data')
ax.plot(test_data.index, test_data['LandAverageTemperature'], label='Test Data')
ax.plot(test_data.index, lr_pred, label='Linear Regression')
ax.plot(test_data.index, rf_pred, label='Random Forest')
if sarima_success:
    ax.plot(test_data.index, sarima_pred, label='SARIMA')

ax.set_title('Model Predictions vs Actual Data (Last 20 Years)', fontsize=16)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()

# Set y-axis limits to focus on the range of the data
y_min = min(train_data.loc[focus_start:, 'LandAverageTemperature'].min(), test_data['LandAverageTemperature'].min(), lr_pred.min(), rf_pred.min())
y_max = max(train_data.loc[focus_start:, 'LandAverageTemperature'].max(), test_data['LandAverageTemperature'].max(), lr_pred.max(), rf_pred.max())
if sarima_success:
    y_min = min(y_min, sarima_pred.min())
    y_max = max(y_max, sarima_pred.max())
ax.set_ylim(y_min - 0.5, y_max + 0.5)

plt.tight_layout()
save_figure(fig, 'model_predictions_focused.png')

print("Generating feature importance plot...")
# Feature importance for Random Forest
feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
ax.set_title('Feature Importance (Random Forest)')
save_figure(fig, 'feature_importance.png')

print("Making future predictions...")
# Future predictions
future_dates = pd.date_range(start=temp_data.index[-1] + pd.Timedelta(days=1), periods=120, freq='M')
future_df = pd.DataFrame(index=future_dates, columns=['Year', 'Month'])
future_df['Year'] = future_df.index.year
future_df['Month'] = future_df.index.month

future_df_poly = poly.transform(future_df)
lr_future = lr_model.predict(future_df_poly)
rf_future = rf_model.predict(future_df)
if sarima_success:
    sarima_future = sarima_results.forecast(steps=120)
else:
    sarima_future = None

fig, ax = plt.subplots(figsize=(15, 8))

# Focus on the last 10 years of historical data and future predictions
focus_start = temp_data.index[-120]
ax.plot(temp_data.loc[focus_start:].index, temp_data.loc[focus_start:, 'LandAverageTemperature'], label='Historical Data')
ax.plot(future_dates, lr_future, label='Linear Regression')
ax.plot(future_dates, rf_future, label='Random Forest')
if sarima_success:
    ax.plot(future_dates, sarima_future, label='SARIMA')

ax.set_title('Future Temperature Predictions (Next 10 Years)', fontsize=16)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()

# Set y-axis limits to focus on the range of the data
y_min = min(temp_data.loc[focus_start:, 'LandAverageTemperature'].min(), lr_future.min(), rf_future.min())
y_max = max(temp_data.loc[focus_start:, 'LandAverageTemperature'].max(), lr_future.max(), rf_future.max())
if sarima_success:
    y_min = min(y_min, sarima_future.min())
    y_max = max(y_max, sarima_future.max())
ax.set_ylim(y_min - 0.5, y_max + 0.5)

plt.tight_layout()
save_figure(fig, 'future_predictions_focused.png')

print("Saving results...")
# Save results
results = {
    'Model': ['Linear Regression', 'Random Forest'],
    'MSE': [lr_mse, rf_mse],
    'R2': [lr_r2, rf_r2]
}
if sarima_success:
    results['Model'].append('SARIMA')
    results['MSE'].append(sarima_mse)
    results['R2'].append(sarima_r2)

results_df = pd.DataFrame(results)
print(results_df)

results_df.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")
plot_historical_trend(temp_data)
plot_model_comparison(results_df)
plot_actual_vs_predicted(y_test, rf_pred, 'Random Forest')
plot_residual_analysis(y_test, rf_pred, 'Random Forest')
plot_seasonal_patterns(temp_data)
print(f"All plots saved in the '{plots_dir}' directory")

print("Script completed.")