import numpy as np
import numpy.random as ran
from matplotlib import pyplot as plt
SAMPLES = 1000
MINS = [4.,9.]
MAXES = [5.,10.]
s = [
    ran.uniform(min,max,SAMPLES) for min,max in zip(MINS,MAXES)
]
MIXING_MATRIX = [
    [1.,1.],
    [1.,-1.]
]
x = [
    s[0] + s[1],
    s[0] - s[1]
]
BINS = 30
plt.hist(s[0],BINS,label='source 1')
plt.hist(s[1],BINS,label='source 2')
plt.hist(x[0],BINS,label='sum')
plt.hist(x[1],BINS,label='difference')
plt.legend()
plt.show()



