# example of a difference transform (python) 
# difference dataset 
def difference(data, interval): 
    return [data[i] - data[i - interval] for i in range(interval, len(data))] 
# invert difference 
def invert_difference(orig_data, diff_data, interval): 
    return [diff_data[i-interval] + orig_data[i-interval] for i in range(interval, len(orig_data))] 
# define dataset 
data = [x for x in range(1, 10)] 
print("original data {}".format(data)) 
# difference transform 
transformed = difference(data, 1) 
print("transformed data {}".format(transformed))
# invert difference 
inverted = invert_difference(data, transformed, 1) 
print("inverted data {}".format(inverted))



# example of power transform and inversion (python) 
from math import log 
from math import exp 
from scipy.stats import boxcox 
# invert a boxcox transform for one value 
def invert_boxcox(value, lam): 
    # log case 
    if lam == 0: 
        return exp(value) 
    # all other cases 
    return exp(log(lam * value + 1) / lam)
print("power transform and inversion")
# define dataset 
data = [x for x in range(1, 10)] 
print("original data {}".format(data))
# power transform 
transformed, lmbda = boxcox(data) 
print("transformed data {}".format(transformed), lmbda) 
# invert transform 
inverted = [invert_boxcox(x, lmbda) for x in transformed] 
print("inverted data {}".format(inverted))