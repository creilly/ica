#Code to guess single source using ICA

import numpy as np
import numpy.random as ran
from matplotlib import pyplot as plt

ran.seed()
SAMPLES = 10000
MINS = [4.,9.]
MAXES = [5.,10.]

#s will be the original uniform distributions
s = [
    ran.uniform(min,max,SAMPLES) for min,max in zip(MINS,MAXES)
]

#x is the mixing matrix to combine the 2 s signals
x = [
    s[0] + s[1],
    s[0] - 2. * s[1]
]


BINS = 30
# plt.hist(s[0],BINS,label='source 1')
# plt.hist(s[1],BINS,label='source 2')
# plt.hist(x[0],BINS,label='sum')
# plt.hist(x[1],BINS,label='difference')
# plt.legend()
# plt.show()

#Normalize the two signal in order for preprocessing to make seperation easier
for i in range(len(x)):
    x[i] = x[i] - np.average(x[i])
    s[i] = s[i] - np.average(s[i])
s = np.vstack(s)
x = np.vstack(x)
# plt.plot(*x,ls='None',marker='o',label='raw')

def covariance(signals):
    x = signals.transpose()
    covariance = [
        [0.,0.],
        [0.,0.]
    ]
    for a,b in x:
        covariance[0][0] += a*a / len(x)
        covariance[0][1] += a*b / len(x)
        covariance[1][0] += b*a / len(x)
        covariance[1][1] += b*b / len(x)
    return covariance
eigen_values, eigen_vectors = np.linalg.eig(covariance(x))

D = np.diag(1. / np.sqrt(eigen_values))
E = eigen_vectors
xBar = np.dot(                                 #xbar is the whitened mixing matrix 
    E,
    np.dot(
        D,
        np.dot(
            E.transpose(),
            x
        )
    )
)
# plt.plot(*y,ls='None',marker='.',label='whitened')
# plt.legend()
# plt.show()

print '\n'.join(
    '\t'.join(
        str(d) for d in row 
    ) for row in covariance(xBar)
)

def F(x):                                   #F(x) is just the g2(u) on pdf eq 39
    return x * np.exp(-1. * np.square(x) / 2.)

#derivative of g(u)
def f(x):
    return (1. - np.square(x)) * np.exp(-1. * np.square(x) / 2.)


def compute_w_plus(signals,w):
    c, d = w
    w_plus = [0.,0.]
    x = signals.transpose()

    for a,b in x:
        w_plus[0] += 1. / len(x) * (
            a * F(c*a + d*b) - f(c*a + d*b) * c
        )
        w_plus[1] += 1. / len(x) * (
            b * F(c*a + d*b) - f(c*a + d*b) * d
        )
    w_plus = np.array(w_plus)
    w_plus = w_plus / np.linalg.norm(w_plus)
    return w_plus

theta = ran.uniform(0.,2. * np.pi)
gamma = ran.uniform(0.,2. * np.pi)
wOne = np.array([np.cos(theta),np.sin(theta)])
wTwo = np.array([np.cos(gamma),np.sin(gamma)])
guessesOne = []
guessesOne.append(list(wOne))
guessesTwo = []
guessesTwo.append(list(wTwo))

print "This is one"
while True:
    
    print wOne
    w_plusOne = compute_w_plus(xBar,wOne)


    guessesOne.append(list(w_plusOne))

    if np.abs(np.dot(wOne.transpose(),w_plusOne) - 1.) < 1.e-100:
        break
    wOne = w_plusOne

print "this is two"
while True:
    
    print wTwo
    w_plusTwo = compute_w_plus(xBar,wTwo)


    guessesTwo.append(list(w_plusTwo))

    if np.abs(np.dot(wTwo.transpose(),w_plusTwo) - 1.) < 1.e-100:
        break
    wTwo = w_plusTwo


plt.plot(*zip(*guessesOne),label='Guess 1')
plt.plot(*zip(*guessesTwo),label='Guess 2')
plt.show()

demixedOne = []
demixedTwo = []
for a, b in x.transpose():
    c, d = w_plusOne
    e, f = w_plusTwo

    demixedOne.append(a*c + b*d)
    demixedTwo.append(a*e + b*f)

plt.hist(demixedOne,BINS, label='Guess 1')
plt.hist(demixedTwo,BINS, label='Guess 2')
plt.legend()
plt.show()

plt.plot(s[0][:100], label="Signal 1")
plt.plot(demixedOne[:100],label='Guess 1')
plt.plot(demixedTwo[:100],label='Guess 2')
plt.legend()
plt.show()
plt.plot(s[1][:100],label='Signal 2')
plt.plot(demixedOne[:100],label='Guess 1')
plt.plot(demixedTwo[:100], label='Guess 2')
plt.legend()
plt.show()
