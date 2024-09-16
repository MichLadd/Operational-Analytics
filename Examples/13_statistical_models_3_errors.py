import numpy as np, pandas as pd 
from statsmodels.tsa.stattools import acf 

# Accuracy metrics 
def forecast_accuracy(forecast, actual): 


    forecast = np.array(forecast)
    actual = np.array(actual)

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual)) 
    # MAPE
    me = np.mean(forecast - actual) # ME 
    mae = np.mean(np.abs(forecast - actual)) # MAE
    mpe = np.mean((forecast - actual)/actual) # MPE 
    rmse = np.mean((forecast - actual)**2)**.5 # RMSE 
    corr = np.corrcoef(forecast, actual)[0,1] # correlation coeff
    mins = np.amin(np.hstack([forecast[:,None], actual[:,None]]), axis=1) 
    maxs = np.amax(np.hstack([forecast[:,None], actual[:,None]]), axis=1) 
    minmax = 1 - np.mean(mins/maxs) # minmax 
    acf1 = acf(forecast-actual)[1] # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 'corr':corr, 'minmax':minmax})


# Example

from statsmodels.tsa.statespace.sarimax import SARIMAX
# Caricare i dati
df = pd.read_csv('data/FilRouge.csv')
ds = df['sales']

# Preprocessing: log transform
logdata = np.log(ds)
logdiff = logdata.diff().dropna()

# Dividere i dati in train e test set
cutpoint = int(0.7 * len(logdiff))
train = logdiff[:cutpoint]
test = logdiff[cutpoint:]

# Adattare il modello SARIMAX
sarima_model = SARIMAX(train, order=(0, 2, 2), seasonal_order=(0, 1, 0, 4))
sfit = sarima_model.fit(disp=False)

# Previsioni in-sample
ypred = sfit.predict(start=0, end=len(train)-1)

# Previsioni out-of-sample
forewrap = sfit.get_forecast(steps=len(test))
forecast_val = forewrap.predicted_mean
forecast_ci = forewrap.conf_int()

print( forecast_accuracy(forecast_val, test.values) )