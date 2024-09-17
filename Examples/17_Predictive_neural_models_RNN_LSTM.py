# Recurrent NN
# The most effective neural model for forecasting so far. 
# • They have at least one backward connection (feedback loop) 
# • They can keep activations even with no input → they have an internal state.
# • Even with one single neuron, we can input a time series and get a series in the output.
# • They can approximate any dynamical system. 
# • Very complex mathematical analysis. 
# • Very complex learning algorithms, hard to control. The most used one is Backpropagation Through Time (BTT).

# Their effect is not function approximation, but process modeling.

# RNN, activation functions
# • Sigmoid: output in [0,1]
# • Tanh: output in [-1,1]
# • ReLU: output in [0,∞]

# RNN disadvantages
# • They are hard to train (vanishing gradient problem) and slow.

# RNN Architectures
# One-to-one: standard feedforward neural network
# One-to-many: music generation
# Many-to-one: sentiment analysis
# Many-to-many: translation


### LSTM
# LSTM is a special kind of RNN, capable of learning long-term dependencies.

# An LSTM is a complicated, but single layer network

# They do not have proper neurons, but memory blocks connecting layers.
# A block is more complex than a normal neuron, it contains “gates” to maintain Internal state and output
# • Forget gate: decides what information to forget
# • Input gate: decides what information to store
# • Output gate: decides what to output
'''
# Example
import pandas as pd, numpy as np, os 
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
df = pd.read_csv('data/gioiellerie.csv', header=0) 
df["period"] = df["year"].map(str) +"-" + df["month"].map(str) 
df["period"] = pd.to_datetime(df["period"], format="%Y-%m").dt.to_period("M") 
# df = df.set_index(‘period’) 
aSales = df["sales"].to_numpy() # array of sales data 
logdata = np.log(aSales) # log transform 
data = pd.Series(logdata) # convert to pandas series
plt.rcParams["figure.figsize"] = (10,8) # redefines figure size 
plt.plot(data.values)
plt.title("Log sales data")
plt.show() # data plot

# train and test set 
train = data[:-12] 
test = data[-12:]
reconstruct = np.exp(np.r_[train,test]) # simple reconstruction

# ------------------------------------------------- neural forecast 
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler() 
scaler.fit_transform(train.values.reshape(-1, 1)) 
scaled_train_data = scaler.transform(train.values.reshape(-1, 1)) 
scaled_test_data = scaler.transform(test.values.reshape(-1, 1))


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator 
n_input = 12
n_features = 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)


from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM
lstm_model = Sequential() 
lstm_model.add(LSTM(20, activation="relu", input_shape=(n_input, n_features), dropout=0.05)) 
lstm_model.add(Dense(1)) 
lstm_model.compile(optimizer="adam", loss="mse") 
lstm_model.fit(generator,epochs=25) 
lstm_model.summary()

losses_lstm = lstm_model.history.history['loss'] 
plt.xticks(np.arange(0,21,1)) # convergence trace 
plt.plot(range(len(losses_lstm)),losses_lstm)
plt.title("LSTM Loss")
plt.show()

lstm_predictions_scaled = list() 
batch = scaled_train_data[-n_input:] 
curbatch = batch.reshape((1, n_input, n_features)) # 1 dim more 
for i in range(len(test)): 
    lstm_pred = lstm_model.predict(curbatch)[0] 
    lstm_predictions_scaled.append(lstm_pred) 
    curbatch = np.append(curbatch[:,1:,:],[[lstm_pred]],axis=1)
lstm_forecast = scaler.inverse_transform(lstm_predictions_scaled) 
yfore = np.transpose(lstm_forecast).squeeze()
# recostruction 
expdata = np.exp(train)
# unlog
expfore = np.exp(yfore) 
plt.plot(df.sales, label="sales") 
plt.plot(expdata,label="expdata") 
plt.plot([None for x in expdata]+[x for x in expfore], label='forecast') 
plt.legend()
plt.title("LSTM Forecast") 
plt.show()

### Gated Recurrent Units
#Gated Recurrent Units (GRUs) are another type of RNN, similar to LSTM but with fewer parameters: they do not have an output gate but a Update gate and a Reset gate.

'''
### Diebold Mariano Test
# The Diebold-Mariano test is a statistical test for comparing the forecast accuracy of two models.
# The null hypothesis is that the two models have the same forecast accuracy.
# The test statistic is the difference between the two models' forecast errors.
# The test statistic is compared to a critical value from the standard normal distribution.
# If the test statistic is greater than the critical value, the null hypothesis is rejected.

# from dm_test import dm_test 
from dm_test import dm_test
if __name__ == '__main__': 
    MLP = [417.662,381.815,423.952,469.055, 471.502, 547.532, 639.609,
           596.040, 477.471, 445.010, 368.674,414.231] 
    LSTM= [416.999,390.999,418.999,461.000, 471.999, 535.000, 622.000,
           606.000, 507.999, 461.000, 390.000, 431.999] 
    actual = [417,391,419,461,472,535,622,606,508,461,390,432] 
    rt = dm_test(actual, MLP, LSTM, h=1, crit="MSE") # h=1, one step ahead forecast, MSE Mean squared error loss
    print(rt)

# Riultato: dm_return(DM=2.933300449662608, p_value=0.013609843048263027)
# DM = 2.93 --> DM > 1.96, reject H0
# p_value = 0.0136 p_value < 0.05, reject H0