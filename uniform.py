import numpy as np
import numpy.random as ran
from matplotlib import pyplot as plt
ran.seed()
SAMPLES = 10000
MINS = [4.,9.]
MAXES = [5.,10.]
s = [
    ran.uniform(min,max,SAMPLES) for min,max in zip(MINS,MAXES)
]
x = [
    s[0] + s[1],
    s[0] - 2. * s[1]
]
BINS = 30
plt.hist(s[0],BINS,label='source 1')
plt.hist(s[1],BINS,label='source 2')
plt.hist(x[0],BINS,label='sum')
plt.hist(x[1],BINS,label='difference')
plt.legend()
plt.show()

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
y = np.dot(
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
    ) for row in covariance(y)
)
def F(x):
    return x * np.exp(-1. * np.square(x) / 2.)
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
w = np.array([np.cos(theta),np.sin(theta)])
guesses = []
guesses.append(list(w))
while True:
    print w
    w_plus = compute_w_plus(y,w)
    guesses.append(list(w_plus))
    if np.abs(np.dot(w.transpose(),w_plus) - 1.) < 1.e-100:
        break
    w = w_plus
plt.plot(*zip(*guesses))
plt.show()

demixed = []
for a, b in x.transpose():
    c, d = w_plus
    demixed.append(a*c + b*d)
plt.hist(demixed,BINS)
plt.show()
plt.plot(s[0][:100])
plt.plot(demixed[:100])
plt.show()
plt.plot(s[1][:100])
plt.plot(demixed[:100])
plt.show()
