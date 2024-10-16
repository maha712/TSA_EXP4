# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

REG NO: 212222240057
NAME: MAHALAKSHMI K
# Date: 



### AIM:

To implement ARMA model in python.

### ALGORITHM:

1. Import necessary libraries.

2. Set up matplotlib settings for figure size.

3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.

5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

# Load the data
url = '/content/archive (6).zip'
data = pd.read_csv(url)

# Convert the date column to datetime
data['Date_reported'] = pd.to_datetime(data['Date_reported'])
data.set_index('Date_reported', inplace=True)

# Aggregate data by date (e.g., daily new cases)
daily_cases = data.groupby('Date_reported')['New_cases'].sum()

# Check for stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    return result[1]  # p-value

p_value = check_stationarity(daily_cases)

# Difference the data if not stationary
if p_value > 0.05:
    daily_cases_diff = daily_cases.diff().dropna()
else:
    daily_cases_diff = daily_cases

# Fit ARMA model (for example, with p=1, q=1)
model = ARIMA(daily_cases, order=(1, 1, 1))  # Change orders based on ACF and PACF
model_fit = model.fit()

# Plot ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sm.tsa.plot_acf(daily_cases_diff, lags=20, ax=ax[0])
ax[0].set_title('ACF Plot')
sm.tsa.plot_pacf(daily_cases_diff, lags=20, ax=ax[1])
ax[1].set_title('PACF Plot')
plt.show()

# Plot actual vs fitted values
plt.figure(figsize=(12, 6))
plt.plot(daily_cases.index, daily_cases, label='Actual Cases', color='blue')
plt.plot(daily_cases.index, model_fit.fittedvalues, label='Fitted Values', color='red')
plt.title('Actual vs Fitted Values')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
plt.show()

# Plot residuals
residuals = model_fit.resid
plt.figure(figsize=(12, 6))
plt.plot(residuals, color='green')
plt.title('Residuals of the ARMA Model')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.axhline(0, color='black', linestyle='--', lw=2)
plt.show()
```
OUTPUT:

SIMULATED ARMA(1,1) PROCESS:

![Screenshot (640)](https://github.com/user-attachments/assets/6b88c12b-5195-4483-b17c-6fbccde7d6b8)

ACF for ARMA(1,1):

![Screenshot (641)](https://github.com/user-attachments/assets/d5e658ce-071a-4b3c-a38d-50033f762c5c)

PACF for ARMA(1,1):

![Screenshot (642)](https://github.com/user-attachments/assets/17174ab9-a257-4ed6-8da9-2e31e2ce3e5a)

SIMULATED ARMA(2,2) PROCESS:

![Screenshot (643)](https://github.com/user-attachments/assets/9c8e8db5-8dec-4236-9ce9-38bf7f691600)

ACF for ARMA(2,2):

![Screenshot (644)](https://github.com/user-attachments/assets/862efe80-f5b7-43f7-9e3f-3ca1115c1c8c)

PACF for ARMA(2,2):

![Screenshot (645)](https://github.com/user-attachments/assets/877b36b4-27b9-476d-a01f-e37ea0a201ab)

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
