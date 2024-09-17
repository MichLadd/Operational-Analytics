# Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task.

# Several ML models can be applied to forecasting: neural networks, but also decision trees, support vector machines (SVMs), Bayesian networks, and others.

### Support Vector Machines (SVM)
# SVM are machine learning classifiers which, given labeled training data (supervised learning), compute an optimal hyperplane which separates (categorizes) the examples.

### Support Vector Regression (SVR) with ε-insensitive loss function (Vapnik, 1995) allows a tolerance degree to errors not greater than ε.

# SVR in sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Dati di esempio
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
Y = np.array([45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000])

# Trasformazione dei dati
sc_X = StandardScaler() 
sc_y = StandardScaler()
X = sc_X.fit_transform(X) 
Y = Y.reshape(-1, 1)  # Trasforma Y in un array 2D
y = sc_y.fit_transform(Y)

# Modello SVR
regressor = SVR(kernel='rbf', C=250.0, gamma=20, epsilon=0.2)
regressor.fit(X, y.flatten())

# Predizione
y_pred = regressor.predict(sc_X.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred.reshape(-1, 1))  # Trasforma y_pred in un array 2D

print(y_pred)


### Decision Trees, ID3
# Decision Tree: It is a tree-like structure where at every node we make a decision and continue doing it till we reach a conclusion.
# Decision trees are a popular method for various machine learning tasks. They are easy to interpret and can handle both numerical and categorical data.

# ID3 (Iterative Dichotomiser 3) is a decision tree algorithm that uses entropy and information gain to build a decision tree. 
# Information Entropy is a measure of impurity in a set of examples. The entropy of a set is zero when all its members belong to the same class.
# Information Gain is the measure of the effectiveness of a feature in classifying the data.


## Building Ensemble Classifiers
# Ensemble methods are techniques that create multiple models and then combine them to produce improved results.
# - Boosting: Training a bunch of models sequentially. Each model learns from the mistakes of the previous mode
#       -AdaBoosting, Gradient Boosting, XGBoosting
# - Bagging: Training a bunch of models in parallel. Each model learns from a random subset of the data, where the dataset is same size as original but is randomly sampled with replacement (bootstrapped)

### BOOSTING 

## Gradient Boosting Decision Trees (GBDT)
# combine many weak learners to come up with one strong learner
# Due to this sequential connection, boosting algorithms are usually slow to learn, but also highly accurate.

## Boosting is a numerical optimization problem where the objective is to minimize the loss of the model by adding weak learners using a gradient descent like procedure.
## Algorithms using boosting are described as stage-wise additive models: a new weak learner is added at a time and existing weak learners in the model are frozen and left unchanged.
##
## Gradient boosting involves three elements: 
# • A loss function to be optimized. 
# • A weak learner to make predictions. 
# • An additive model to add weak learners to minimize the loss function.

## AdaBoost
# AdaBoost is a boosting algorithm that combines multiple weak classifiers to create a strong classifier.
# AdaBoost assigns weights to the data points and adjusts the weights at each iteration to classify the data correctly.

## Construct Weak Classifiers
## Using Different Data Distribution 
## • Start with uniform weighting 
## • During each step of learning 
## • Increase weights of the examples which are not correctly learned by the weak learner
## • Decrease weights of the examples which are correctly learned by the weak 
#
## Combine Weak Classifiers
## Weighted Voting 
## • Construct strong classifier by weighted voting of the weak classifiers
## Idea 
## • Better weak classifier gets a larger weight 
## • Iteratively add weak classifiers 
## • Increase accuracy of the combined classifier through minimization of a cost function


### XGBoost
# XGBoost (Extreme Gradient Boosting) is a scalable and accurate implementation of gradient boosting machines.

from xgboost import XGBRegressor 
import pandas as pd
import matplotlib.pyplot as plt
# rolling window dataset (see MLP) 
lookback = 12 
dataset = pd.DataFrame() 
df = pd.read_csv('data/BoxJenkins.csv', usecols=[1], engine='python')
for i in range(lookback, 0, -1): 
    dataset['t-' + str(i)] = df.Passengers.shift(i) # build dataset by columns
dataset['t'] = df.values 
dataset = dataset[lookback:] # removes the first lookback columns
x = dataset.iloc[:, :-1] 
y = dataset.iloc[:, -1] 
x_train, xtest = x[:-12], x[-12:] 
y_train, ytest = y[:-12], y[-12:] # fit model 
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(x_train, y_train) # make a one-step prediction 
yhat = model.predict(xtest)
plt.plot(ytest.values, label='Actual')
plt.plot(yhat, label='Predicted')
plt.legend()
plt.title('XGBoost forecast')
plt.show()


### BAGGING (parallel)
# Bagging is a parallel ensemble technique that combines the predictions from multiple independent models.
# Decision trees are sensitive to the specific data used to train the model, to overcome this, we can use a technique called bootstrap aggregating or bagging.
# Bagging involves training multiple models on different subsets of the training data and combining their predictions.

# Random Forest is a bagging method that use a randomly sampled subset of the training data to train multiple decision trees.
# It also uses a random subset of features at each split to reduce the correlation between the trees.
# Can be used both for classification and regression problems

# Random Forest Example

from sklearn.ensemble import RandomForestRegressor 
from sklearn.feature_selection import RFE 
from sklearn.metrics import mean_absolute_error
# rolling window dataset (see MLP) 
lookback = 12 
dataset = pd.DataFrame()
df = pd.read_csv('data/BoxJenkins.csv', usecols=[1], engine='python')
for i in range(lookback, 0, -1):
     dataset['t-'+str(i)] = df.Passengers.shift(i) # build dataset by cols
dataset['t'] = df.values
dataset = dataset[lookback:] # removes the first lookback columns 
X = dataset.iloc[:, :-1] 
y = dataset.iloc[:, -1] 
x_train, xtest = X[:-12], X[-12:] 
y_train, ytest = y[:-12], y[-12:]


# define and fit the model 
RFmodel = RandomForestRegressor(n_estimators=500, random_state=1) 
RFmodel.fit(x_train, y_train)
# forecast testset 
pred = RFmodel.predict(xtest)
mse = mean_absolute_error(ytest, pred)
print("MSE={}".format(mse)) 
pred = pd.Series(pred,
index=ytest.index) # from array to series
plt.figure() 
plt.plot(y_train.index,y_train.values, label='train') 
plt.plot(ytest.index, ytest.values,'-o', label='Actual') 
plt.plot(pred.index, pred.values,'-o', label='Forecast')
plt.legend() 
plt.title('Random Forest forecast')
plt.show()

# Stats about the trees in random forest 
n_nodes = [] 
max_depths = [] 
for ind_tree in RFmodel.estimators_: 
    n_nodes.append(ind_tree.tree_.node_count) 
    max_depths.append(ind_tree.tree_.max_depth)
print(f'Average number of nodes {int(np.mean(n_nodes))}') 
print(f'Average maximum depth {int(np.mean(max_depths))}')
# plot first tree (index 0) 
from sklearn.tree import plot_tree 
fig = plt.figure(figsize=(15, 10)) 
plot_tree(RFmodel.estimators_[0], max_depth=2, feature_names=dataset.columns[:-1], #class_names=dataset.columns[-1],  Il parametro class_names è utilizzato per i modelli di classificazione, non per i modelli di regressione.
          filled=True, impurity=True, rounded=True)
plt.legend()
plt.title('First tree in Random Forest')
plt.show()