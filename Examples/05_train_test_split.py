import pandas as pd
import matplotlib.pyplot as pyplot
ds = pd.read_csv('data/BoxJenkins.csv', header=0) 
X = ds.Passengers.values 
train_size = int(len(X) * 0.66) 
train, test = X[0:train_size], X[train_size:len(X)] 
print('Observations: %d' % (len(X))) 
print('Training Observations: %d' % (len(train))) 
print('Testing Observations: %d' % (len(test))) 
pyplot.plot(train) 
pyplot.plot([None for i in train] + [x for x in test]) 
pyplot.show()