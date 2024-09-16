# example of normalization (python) 
from sklearn.preprocessing import MinMaxScaler 
from numpy import array 
# define dataset 
data = [x for x in range(1, 10)] 
data = array(data).reshape(len(data), 1) 
print("original data {}".format(data)) 
# fit transform (rescaling between 0 and 1)
transformer = MinMaxScaler() 
transformer.fit(data) 
# difference transform 
transformed = transformer.transform(data)
print("tranformed data {}".format(transformed)) 
# invert difference
inverted = transformer.inverse_transform(transformed) 
print("inverted data {}".format(inverted))