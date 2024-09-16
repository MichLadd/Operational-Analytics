### VARMAX is an extension of VARMA that includes the modeling of exogenous variables (at the same time steps as the original series)

# The method is suitable for multivariate time series without trend and seasonal components with exogenous variables.
from statsmodels.tsa.statespace.varmax import VARMAX 
import numpy as np
data = list()
for i in range(100): 
    v1 = np.random.random() # artificial dataset with dependency 
    v2 = v1 + np.random.random() 
    row = [v1, v2] 
    data.append(row)
from random import random

data_exog = [x + random() for x in range(100)] 
# fit model 
model = VARMAX(data, exog=data_exog, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction 
data_exog2 = [[100]] 
yhat = model_fit.forecast(exog=data_exog2)
print(yhat)


### SARIMAX 
# Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX) is an extension of SARIMA that also includes the modeling of exogenous variables.
from statsmodels.tsa.statespace.sarimax import SARIMAX 
# artificial dataset 
data1 = [x + np.random.random() for x in range(1, 100)] 
data2 = [x + np.random.random() for x in range(101, 200)] 
# fit model 
model = SARIMAX(data1, exog=data2, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)) 
model_fit = model.fit(disp=False) 
# make prediction 
exog2 = [200 + np.random.random()] 
yhat = model_fit.predict(len(data1), len(data1), exog=[exog2]) 
print(yhat)


### Simple Exponential Smoothing (SES)
# Simple Exponential Smoothing (SES) is a time series forecasting method for univariate data without a trend or seasonality.
# mModels the next time step as an exponentially weighted linear function of observations at prior time steps. It requires a single parameter, α, called the smoothing factor or smoothing coefficient.

# The parameter controls the rate at which the influence of the observations at prior time steps decay exponentially
# Values close to 1 mean that the model pays attention mainly to the most recent observations, whereas values close to 0 mean that more of the history is taken into account. 
#
# The method is suitable for univariate time series without trend and seasonal components.

import pandas as pd
import matplotlib.pyplot as plt

def simple_exp_smooth(data, nforecasts=1, alpha=0.4): 
    n = len(data)
    f = np.full(n + nforecasts, np.nan) # Forecast array
    data = np.append(data, [np.nan] * nforecasts) # forecast placeholders 
    f[1] = data[0] # initialization of first forecast 
    # predictions
    for t in range(2, n+1): 
        f[t] = alpha * data[t - 1] + (1 - alpha) * f[t - 1] # forecast
    for t in range(n+1,n+nforecasts): 
        f[t] = alpha * f[t - 1] + (1 - alpha) * f[t - 2]
    return pd.DataFrame.from_dict({"Data": data, "Forecast": f, "Error": data - f})

sales = pd.read_csv("data/FilRouge.csv",usecols=["sales"]).T
sales = np.array(sales).flatten()
df = simple_exp_smooth(sales, nforecasts=4,alpha=0.5)
MAE = df["Error"].abs().mean() 
print("MAE:", round(MAE, 2))
RMSE = np.sqrt((df["Error"] ** 2).mean())
print("RMSE:", round(RMSE, 2)) 
df.index.name = "Periods" 
plt.figure(figsize=(8, 3))
plt.plot(df[["Data"]],label="data")
plt.plot(df[["Forecast"]],label="Simple smoothing") # see last part of forecast, has lost the trend Smoothed contributions
plt.legend() 
plt.title("Simple Exponential Smoothing")
plt.show()


### Holt Winter’s Exponential Smoothing (HWES)
# Holt Winter’s Exponential Smoothing (HWES) is a time series forecasting method for univariate data with a trend and/or seasonality.
# Like SES but with two additional components: trend and seasonality.

# Parameters: α, level smoothing coefficient, β trend coefficient, γ seasonal coefficient. Trend and seasonality may be modeled as either additive or multiplicative (for a linear or exponential changements).

# The method is suitable for univariate time series with trend and/or seasonal components.
'''
from statsmodels.tsa.holtwinters import ExponentialSmoothing

 # fit model
model = ExponentialSmoothing(train, seasonal_periods=4,trend="add", seasonal="mul",
                               damped_trend=True, use_boxcox=True, initialization_method="estimated")
hwfit = model.fit() # make forecast 
yfore = hwfit.predict(len(train), len(train)+3) 
print(yfore)
'''
def holtwinters(x, m, nfor, alpha = 0.5, beta = 0.5, gamma = 0.5): 
    Y = x[:]
    initial_values = np.array([0.0, 1.0, 0.0]) 
    boundaries = [(0, 1), (0, 1), (0, 1)]
    type = 'multiplicative' # could have been additive
    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
    s = [Y[i] / a[0] for i in range(m)] 
    y = [(a[0] + b[0]) * s[0]] 
    rmse = 0
    for i in range(len(Y) + nfor): 
        if i == len(Y):
            Y.append((a[-1] + b[-1]) * s[-m])
        a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i])) 
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i]) 
        s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i]) 
        y.append((a[i + 1] + b[i + 1]) * s[i + 1])
    rmse = np.sqrt(sum([(m-n)**2 for m, n in zip(Y[:-nfor], y[:-nfor-1])]) / len(Y[:-nfor])) 
    return Y, rmse

# Filrouge eample
sales = pd.read_csv("data/FilRouge.csv",usecols=["sales"]).T 
sales = np.array(sales).flatten() 
hwforecasts,rmse = holtwinters(sales.tolist(),4,4,alpha=0.6, beta=0.5, gamma=0.3)
print("RMSE HW:", round(RMSE, 2)) 
plt.figure(figsize=(8, 3)) 
plt.plot(sales,linewidth=3,label="data") 
plt.plot(hwforecasts,label="Holt-Winters") 
plt.title("Holt-Winters Exponential Smoothing")
plt.legend() 
plt.show()


### BATS and TBATS
# BATS and TBATS are time series forecasting methods that can handle multiple seasonalities.
# BATS is an acronym for Box-Cox transformation, ARMA errors, Trend and Seasonal components.
# TBATS is an acronym for Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components.

# The methods are suitable for univariate time series with multiple seasonalities.
# TBATS tries to fit various alternative models (quite slow), such as: 
# • with and without Box-Cox transformation. 
# • with and without trend 
# • with and without ARMA(p,q) process used to model residuals 
# • with and without seasonality 
# • multiple frequencies, with different amplitudes, to model multiple seasonalities

from tbats import TBATS
from random import random
sales = pd.read_csv("data/FilRouge.csv",usecols=["sales"]).T
sales = np.array(sales).flatten()
# fit model
model = TBATS(seasonal_periods=(4,12))
tbfit = model.fit(sales)
# make forecast
yfore = tbfit.forecast(steps=4)
print(yfore)
plt.figure(figsize=(8, 3))
plt.plot(sales,linewidth=3,label="data")
plt.plot(yfore, label="TBATS")
plt.title("TBATS Forecast")
plt.legend()
plt.show()
