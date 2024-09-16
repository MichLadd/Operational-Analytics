# Forecast, selected algorithms
# Time series forecasting classical methods: 
# • Autoregression (AR) 
# • Vector Autoregression (VAR) 
# • Moving Average (MA) 
# • Autoregressive Moving Average (ARMA) 
# • Vector Autoregression Moving-Average (VARMA) 
# • Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)
# • Autoregressive Integrated Moving Average (ARIMA) 
# • Seasonal Autoregressive Integrated Moving-Average (SARIMA) 
# • Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)
# • Simple Exponential Smoothing (SES) 
# • Holt and Winter’s Exponential Smoothing (HWES) 
# • BATS (and TBATS)



### Autoregression (AR)
# Autoregression (AR) is a time series model that uses observations from previous time steps as input to a regression equation to predict the value at the next time step.
import numpy as np
from statsmodels.tsa.ar_model import AutoReg 
# artificial dataset 
data = [x + np.random.random() for x in range(1, 100)] 
# fit model 
model = AutoReg(data, lags=1) 
model_fit = model.fit() 
# make prediction 
yhat = model_fit.predict(len(data), len(data)) 
print("AR model")
print(yhat)

### Vector Autoregression (VAR)
# Vector Autoregression (VAR) is a multivariate version of autoregression. It models the next step in each time series using an AR model.

from statsmodels.tsa.api import VAR
data = list() 
# artificial data with internal dependency 
for i in range(100): 
    v1 = i + np.random.random() 
    v2 = v1 + np.random.random() 
    row = [v1, v2]
    data.append(row)
# fit model 
model = VAR(data)
model_fit = model.fit() 
# make prediction 
yhat = model_fit.forecast(model_fit.endog, steps=1)
print("VAR model")
print(yhat)

### Moving Average (MA)
# Moving Average (MA) is a time series model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.

from statsmodels.tsa.arima.model import ARIMA # questa è deprecata: from statsmodels.tsa.arima.model import ARIMA
# artificial dataset
data = [x + np.random.random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(0, 0, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print("MA model")
print(yhat)

### Autoregressive Moving Average (ARMA)
# Autoregressive Moving Average (ARMA) is a time series model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
# It combines both AR and MA models. ARMA models can be used only on stationary processes, without trend or seasonality (see Arima for non stationary).

from statsmodels.tsa.arima.model import ARIMA
# artificial dataset
data = [x + np.random.random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(2, 0, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print("ARMA model")
print(yhat)

### Vector Autoregression Moving-Average (VARMA)
# Vector Autoregression Moving-Average (VARMA) is a multivariate version of ARMA.
# The method is suitable for multivariate time series without trend and seasonal components.
from statsmodels.tsa.statespace.varmax import VARMAX
data = list()
# artificial data with internal dependency
for i in range(100):
    v1 = i + np.random.random()
    v2 = v1 + np.random.random()
    row = [v1, v2]
    data.append(row)
# fit model
model = VARMAX(data, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.forecast()
print("VARMA model")
print(yhat)

### Autoregressive Integrated Moving Average (ARIMA)
# Autoregressive Integrated Moving Average (ARIMA) is a time series model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
# It is suitable for univariate time series with trend and without seasonality. 
# The model uses differencing (I of AR I MA as Integrated) of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.

import os 
import pandas as pd, matplotlib.pyplot as plt 
from statsmodels.tsa.arima.model import ARIMA
# Import data 
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
df = pd.read_csv('data/BoxJenkins.csv', usecols=[1], names=['value'], header=0)
# 1,1,2 ARIMA Model (p,d,q) 
model = ARIMA(df.value, order=(1,1,2))
model_fit = model.fit()
print(model_fit.summary()) 
# diff=1 
pred = model_fit.predict(1, len(df), typ="levels") 
plt.plot(df.value) 
plt.plot(pred) 
plt.show() 
plt.title('ARIMA model')
# Plot residual errors 
residuals = pd.DataFrame(model_fit.resid) 
fig, ax = plt.subplots(1,2) 
residuals.plot(title="Residuals", ax=ax[0]) 
residuals.plot(kind='kde', title='Density', ax=ax[1]) 
plt.show()
