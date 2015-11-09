import numpy as np
import numpy.random as ran
from matplotlib import pyplot as plt
SAMPLES = 100
MAXES = [5.,10.]
s = [
    ran.uniform(0,max,SAMPLES) for max in MAXES
]
for signal in s:
    plt.plot(signal)
    plt.show()
plt.plot(*s)
plt.show()

MIXING_MATRIX = [
    [1.,1.],
    [1.,-1.]
]


