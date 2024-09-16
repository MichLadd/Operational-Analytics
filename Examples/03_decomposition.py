from statsmodels.tsa.seasonal import seasonal_decompose 
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 6.0) 
df = pd.read_csv('data/BoxJenkins.csv',usecols=["Passengers"]) 
ds = df[df.columns[0]] 
# converts to series 

# Modello moltiplicativo
result_multiplicative = seasonal_decompose(ds, model='multiplicative', period=12)
result_multiplicative.plot()
plt.show()

# Modello additivo
result_additive = seasonal_decompose(ds, model='additive', period=12)
result_additive.plot()