import os 
import numpy as np, pandas as pd, matplotlib.pyplot as plt 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
# Import data
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
df = pd.read_csv('data/traffico16.csv', usecols=[0], names=['value'], engine='python', header=0) 
# Original Series 
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120}) 
fig, axes = plt.subplots(2, 2, sharex=True) 
axes[0, 0].plot(df.value); 
axes[0, 0].set_title('Original Series') 
plot_acf(df.value, ax=axes[0, 1]) 

# 1st Differencing 
axes[1, 0].plot(df.value.diff()); 
axes[1, 0].set_title('1st Order Differencing') 
plot_acf(df.value.diff().dropna(), ax=axes[1, 1]) 
plt.show()