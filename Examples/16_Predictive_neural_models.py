### NN Neural Networks
# NN identify a model from input data, essentially defining a model of the generating process.
# NN are used in time series forecasting, classification, and regression problems.
# • NN can learn patterns in linear time series 
# • NN can learn patterns in nonlinear time series 
# • NN can generalize linear and nonlinear patterns

# Time−lagged NN
# The net is structured according to feedforward scheme, where inputs are associated to past data and the output is the forecast.

# When the input of hidden and output neurons sees only well-defined subsets of the output of the neurons of the previous layer, 
# the net is named CONVONUTIONAL.
# These are networks most effective for classification, less so for regression.

# Sliding window
# Learning: 
# 1. A window on past data is shown in input. 
# 2. The corresponding output is computed. 
# 3. The output is compared with its corresponding actual value.
# 4. Backpropagation: weights are changed in order to reduce prediction error.
# 5. The window is shifted forward, presenting the following input.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Transform a data series into a learning dataset where the next datapoint of each record is the unknown.

# converts an array of values into two np arrays 
def create_dataset(dataset, look_back=1): 
    dataX, dataY = [], [] 
    for i in range(len(dataset) - look_back): 
        a = dataset[i:(i + look_back)] 
        dataX.append(a) 
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# converts a dataframe, 1 col of values, into a windowed dataframe 
def create_dataset2(df, look_back=1,colname='value'): 
    cols = ['x{}'.format(x) for x in range(look_back)] + ['y'] 
    df_win = [] 
    for i in range(look_back, df.shape[0]): 
        df_win.append(df.loc[i-look_back:i, colname].tolist())
    df_win = pd.DataFrame(df_win, columns= cols) 
    return df_win

# NN are able to approximate any function.

# Only one neuron →AR(p), nonlinear 
# Feedforward NN (MLP) → combination of AR(p) 
# Recurrent NN (Elman, Jordan) →ARMA(p,q) nonlinear


# Python (keras), MLP
# Sliding window MLP, Airline Passengers dataset (predicts t+1) 
import os, math 
import numpy as np, pandas as pd 
import matplotlib.pyplot as plt 
from keras.models import Sequential # pip install keras 
from keras.layers import Dense
# pip install tensorflow (as administrator)
# from series of values to windows matrix 
def create_dataset(dataset, look_back=1): 
    dataX, dataY = [], [] 
    for i in range(len(dataset) - look_back):
         a = dataset[i:(i + look_back), 0]
         dataX.append(a) 
         dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY) 

np.random.seed(550) # for reproducibility
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
df = pd.read_csv('data/BoxJenkins.csv', usecols=[1], names=['Passengers'], header=0) 
dataset = df.values
# time series values 
dataset = dataset.astype('float32') # needed for MLP input

# split into train and test sets 
train_size = int(len(dataset) - 12) 
test_size = len(dataset) - train_size 
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:] 
print("Len train={0}, len test={1}".format(len(train), len(test)))
# sliding window matrices (look_back = window width); dim = n - look_back - 1
look_back = 2
testdata = np.concatenate((train[-look_back:],test))
trainX, trainY = create_dataset(train, look_back) 
testX, testY = create_dataset(testdata, look_back)
# Multilayer Perceptron model 
loss_function = 'mean_squared_error' 
model = Sequential() 
model.add(Dense(8, input_dim=look_back, activation='relu')) # 8 hidden neurons 
model.add(Dense(1))
# 1 output neuron
model.compile(loss=loss_function, optimizer='adam') 
model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)


# Estimate model performance 
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(trainScore, math.sqrt(trainScore))) 
testScore = model.evaluate(testX, testY, verbose=0) 
print('Test Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(testScore, math.sqrt(testScore))) # generate predictions for training and forecast for plotting 
trainPredict = model.predict(trainX)
testForecast = model.predict(testX)
plt.plot(dataset) 
plt.plot(np.concatenate((np.full(look_back-1, np.nan), trainPredict[:,0]))) 
plt.plot(np.concatenate((np.full(len(train)-1, np.nan), testForecast[:, 0]))) 
plt.title('Neural Forecasting (MLP)')
plt.legend(['Original', 'Train', 'Test'], loc='upper left')
plt.show()


### Modelling process Phases for the definition of a neural forecast model :
# 1) Preprocessing (the same as statistical models) 
#   • Transforms (diff, log, ...) 
#   • Scaling, if different variables they must have comparable values 
#   • Normalization, to [0,1] or [-1,1]
# 2) Choice of the architecture of the NN 
#   • Number of neurons, input, hidden, output 
#   • Number of hidden layers (one: MLP, many: deep network) 
#   • Processing at the nodes (activation function: lin, logistic, ...) 
#   • Connections between layers
# 3) Training 
#   • Weights initialization (and reinitialization?) 
#   • Training algorithm (backpropagation, which one?, GLT …) 
#   • Parameter setting
# 4) Use of the NN 
# 5) Validation 
#   • Choice of the dataset 
#   • Validation criteria



### Data preprocessing
# Needed to improve forecast effectiveness. Procedures include: 
# • Verification, correction, editing (errors in the data, etc.) 
# • Re-encoding of variables (from categorical to numeric, one-hot) 
# • Scaling of variables (es. Linear interval scaling,
# • Normalization 
# • Selection of independent variables (PCA, … ) 
# • Removal of the outliers 
# • Insertion of missing values (means, regression, default, ... )

#  of the input vector is a mandatory requirement for the application of MLPs.
# As the sigmoid activation functions in the hidden nodes are only defined in the interval of ]-1, 1[ for the hyperbolic tangent or ]0, 1[ for the logistic function, 
# input data must be scaled to facilitate learning.
# 
# Some authors recommend linear scaling of data into smaller intervals, e.g. [0.2, 0.8], to avoid saturation effects at the asymptotic bounds of the activation functions

# Models, heuristic choices
# • Number of hidden layers: one, two, many
# Activation function: logistic, hyperbolic tangent, linear (only in output nodes), ReLU, ...

# Suggestions (MLP): 
# • The size of the test set should be about 10% to 30% of that of the training set.
# • To avoid overfitting, the test set size should be at least 5 times the number of network weights.
# • In theory, a single hidden layer is enough to approximate any continuous function, but in practice a further hidden layer often helps. More than 4 layers (an input, an output and two hidden) can work wonders but usually only in very controlled contexts.