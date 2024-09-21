import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Seasonal Autoregressive Integrated Moving-Average (SARIMA)
# Seasonal Autoregressive Integrated Moving-Average (SARIMA) is an extension of ARIMA which can be applied also to data with seasonality
# A seasonal diff is included.
#  Parameters: 
# • p,d,q are ARIMA parameters referring to periods (ex., week), 
# • P,D,Q the same parameters referring to seasons (ex., trimester)
# • m number of period in one season (ex., weeks in a trimester. Multiple seasonalities will be addressed by an extension at the end of this slide pack).

# Sarima Utilities (if needed):

# - Preprocessing: log transform
df = pd.read_csv('data/FilRouge.csv') 
npa = df['sales'].to_numpy() 
logdata = np.log(npa) 
plt.plot(npa, color = 'blue', marker = "o") 
plt.plot(logdata, color = 'red', marker = "o") 
plt.title("numpy.log()") 
plt.xlabel("x")
plt.ylabel("logdata") 
plt.show()

# - Autocorrelation
from statsmodels.tsa.stattools import acf 
df = pd.read_csv('data/FilRouge.csv') 
diffdata = df['sales'].diff() 
diffdata[0] = df['sales'][0] 
#reset 1st elem 
acfdata = acf(diffdata,adjusted=True,nlags=8) 
plt.bar(np.arange(len(acfdata)),acfdata) 
plt.title("Autocorrelation")
plt.show()
# otherwise 
import statsmodels.api as sm 
sm.graphics.tsa.plot_acf(diffdata, lags=8) 
plt.show()

#May be needed, may be not. Check your case.
# Preprocessing, log - diff transform 
df = pd.read_csv('data/FilRouge.csv', header=0) 
aSales = df['sales'].to_numpy() 
logdata = np.log(aSales) # array of sales data # log transform
logdiff = pd.Series(logdata).diff() # logdiff transform Preprocessing, train and test set 
cutpoint = int(0.7*len(logdiff))
# example, cut where needed
train = logdiff[:cutpoint] 
test = logdiff[cutpoint:]
#Postprocessing, reconstruction (here very pythonic, otherwise plain loops) 
train[0] = 0
# set first entry 
reconstruct = np.exp(np.r_[train,test].cumsum()+logdata[0])


# Parameter fitting 
import pmdarima as pm 
# pip install pmdarima 
import pandas as pd 
df = pd.read_csv('data/FilRouge.csv', names=['sales'], header=0)
ds = df.sales 
model = pm.auto_arima(ds.values, start_p=1, start_q=1, 
                      test='adf', max_p=3, max_q=3, m=4, 
                      start_P=0, seasonal=True, d=None, D=1, 
                      trace=True, error_action='ignore', 
                      suppress_warnings=True, stepwise=True) # stepwise=False full grid
print(model.summary()) 
morder = model.order
mseasorder = model.seasonal_order # p,d,q # P,D,Q,m
fitted = model.fit(ds) 
yfore = fitted.predict(n_periods=4) # forecast 
ypred = fitted.predict_in_sample() 
plt.plot(ds.values)
plt.plot(ypred) 
plt.plot([None for i in ypred] + [x for x in yfore]) 
plt.xlabel('time')
plt.ylabel('sales')
plt.title('SARIMA auto_arima')
plt.legend(["Data", "Prediction", "Forecast"])
plt.show()

# SARIMA forecast
data = ds.values 
n_periods = 4 
fitted, confint = model.predict(n_periods=n_periods, return_conf_int=True)
tindex = np.arange(df.index[-1]+1,df.index[-1]+n_periods+1)
# series, for plotting 
fitted_series = pd.Series(fitted, index=tindex)
lower_series = pd.Series(confint[:, 0], index=tindex)
upper_series = pd.Series(confint[:, 1], index=tindex)
# Plot plt.plot(data) 
plt.plot(fitted_series, color='darkgreen') 
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
plt.title("SARIMAX Forecast")
plt.show()


### Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)
# Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX) is an extension of SARIMA that also includes the modeling of exogenous variables.
# Exogenous variables are also called covariates and can be thought of as parallel input sequences that have observations at the same time steps as the original series.
# The observations for exogenous variables are included in the model directly at each time step and are not modeled in the same way as the primary endogenous sequence (e.g. a time series).

from statsmodels.tsa.statespace.sarimax import SARIMAX
sarima_model = SARIMAX(ds, order=(0,2,2), seasonal_order=(0,1,0,4))
sfit = sarima_model.fit()
sfit.plot_diagnostics(figsize=(10, 6))
plt.title("SARIMAX diagnostics")
plt.show()

#Predictions in-sample:
ypred = sfit.predict(start=0,end=len(df))
plt.plot(df.sales)
plt.plot(ypred)
plt.xlabel('time')
plt.ylabel('sales')
plt.title('SARIMAX prediction')
plt.show()


#Predictions out-of-sample (forecast):
forewrap = sfit.get_forecast(steps=4)
forecast_ci = forewrap.conf_int() 
forecast_val = forewrap.predicted_mean

forecast_ci = pd.DataFrame(forecast_ci)

plt.plot(df.sales, label='Actual Sales')
plt.fill_between(np.arange(len(df), len(df) + 4), forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='k', alpha=.25) # questo serve per fare l'ombreggiatura
plt.plot(np.arange(len(df), len(df) + 4), forecast_val, label='Forecast')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.title('SARIMAX Out-of-Sample Prediction')
plt.legend()
plt.show()