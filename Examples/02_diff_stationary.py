import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# (1 - Predictive Analytics)

ds = pd.read_csv('data/BoxJenkins.csv', header=0)
data = ds.Passengers.values

# diff data to make it stationary
diffData = np.diff(data, 1)

plt.plot(data)
plt.plot(diffData)

plt.legend(["Data", "Data differenziati"])
plt.show()