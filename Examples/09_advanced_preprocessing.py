# Advanced preprocessing
# More sophisticated data preprocessing techniques include: 
# • Missing values insertions 
# • Outliers detection 
# • Feature extraction 
# • Dimensionality reduction 
# • Noise reduction 
# • Data augmentation



#### Missing values insertions
# Imputation (missing data insertion) does not necessarily give better results.
# Imputation algorithms can be divided into simple but fast approaches (like mean imputation) and more advanced algorithms that need more computation time (like Kalman smoothing)

# Time-Series specific method • Last observation carried forward (LOCF) • Next observation carried backward (NOCB) • Linear interpolation • Spline interpolation

# Using pandas, simple imputations 
'''
ds = pd.Series(y)
ds1=ds.fillna(method="ffill")
ds2=ds.fillna(method="bfill")
ds3=ds.fillna(ds.mean()) 
ds4=ds.interpolate()
'''

#### Outliers detection
# Description: Outliers detection (aka anomaly detection) is a crucial step in time series analysis. 
# Outliers can be caused by different factors, such as errors in data collection, changes in the underlying data generation process, or simply by random fluctuations.
# 
# Possible decomposition approaches: 
# • STL decomposition: seasonal-trend decomposition procedure. This technique gives you an ability to split your time series signal into three parts: seasonal, trend and residue.
# If you analyze deviation of residue and introduce some threshold for it, you’ll get an anomaly detection algorithm. See forward. Similarly with ARIMA or exponential smoothing

# • Classification and regression trees. use supervised learning to teach trees to classify anomaly and non-anomaly data points.
# Caution: in certain applications outliers are data of special interest, not data to be deleted.
#  
# In this script, we will use the seasonal_decompose function from the statsmodels library to detect outliers in a time series. 
# The function decomposes a time series into its trend, seasonal, and residual components. 
# We will then plot the residuals and highlight the outliers using the 1.5 standard deviation rule. 
# Outliers are defined as data points that are more than 1.5 standard deviations away from the mean.
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("data/traffico16.csv") 
col = "ago2" 
series = df[col].fillna(value=df[col].mean()) 
result = seasonal_decompose(series, model='additive',period=7) 
observed=result.observed 
trend = result.trend 
seasonal=result.seasonal 
resid=result.resid 
std = resid.std()
plt.figure(figsize=(10,5)) 
plt.title("Outliers detection")
plt.plot(resid,"o",label="datapoints") 
plt.hlines(0,0,len(resid)) 
plt.hlines(1.5*std,0,len(resid),color="red",label="std limits") 
plt.hlines(-1.5*std,0,len(resid),color="red") 
plt.legend() 
plt.show()



#### Dimensionality reduction
# Description: Dimensionality reduction is a technique used to reduce the number of input variables in a dataset.
# Dimensionality reduction can be done in two different ways:
# - By only keeping the most relevant variables from the original dataset (this technique is properly called feature selection)
# - By finding a smaller set of new variables, each being a combination of the input variables, containing basically the same information as the input variables (this technique is sometimes called dimensionality reduction itself)

# Some techniques for dimensionality reduction:
# - Principal Component Analysis (PCA)
# - Single Value Decomposition (SVD)
# - Random forest (very popular in machine learning)
# - Low variance filter
# - High correlation filter
# - Missing values ratio filter

# PCA example
'''from sklearn.decomposition import PCA
sk_model = PCA(n_components=10)
sk_model.fit_transform(features_ndarray)
print(sk_model.explained_variance_ratio_.cumsum())'''


#### Noise reduction
# Description: Noise reduction is a technique used to remove noise from a dataset.
# Denoising, can be heavily borrowed from signal processing techniques to improve the signal-to-noise ratio (SNR).
# • Low pass filter: It passes signals with a frequency lower than a certain cut-off frequency and attenuates signals with frequencies higher than that. In time series, a simple moving average (SMA) is a low pass filter.
# • High pass filter: It passes signals with a frequency higher than a certain cut-off frequency and attenuates signals with frequencies lower than that. For time series, either linear filters (such as SMA) or non-linear filters (such as median filter) can be used. Consider also Kalman filter.
# • Frequency domain: apply to the series Fourier Transform or Wavelet Transform and subsequently an appropriate filter

# Example: Autoencoders
'''data = np.zeros(shape=(1,len(y))) 
data[0] = measurements.T
model = Sequential()
model.add(Dense(len(measurements)/10, input_dim=len(measurements), activation='relu'))
model.add(Dense(len(measurements)))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)
model.fit(data,data,callbacks=[monitor],verbose=1,epochs=50) 
predictions = model.predict(data)
'''

# Example: Kalman filter
'''
from pykalman import KalmanFilter 
... 
measurements = y.to_numpy() 
kf = KalmanFilter(transition_matrices=[1], 
observation_matrices=[1], 
initial_state_mean=measurements[0], 
initial_state_covariance=1, 
observation_covariance=10, 
transition_covariance=10) 
state_means, state_covariances = kf.filter(measurements)
'''


#### Data augmentation
# Description: Data augmentation is a technique used to increase the size of a dataset by adding slightly modified copies of the original data.
# Some of the common techniques used in data augmentation, especially for time series classification, include:
# - Extrapolation: the relevant fields are filled with values based on heuristics.
# - Tagging: common records are tagged to a group.
# - Aggregation: values are estimated using mathematical values such as averages and means
# - Probability: values are populated based on the probability of events


#### Feature extraction
# Feature extraction creates new features, typically less than the original ones, from an initial set of data with the objective of enhancing machine learning by finding characteristics in the data that help solve a particular problem.
