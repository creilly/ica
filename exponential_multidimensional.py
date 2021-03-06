import numpy as np
from matplotlib import pyplot as plt
import sys

dimensions = 4
samples = 1000000
delta = 1.e-2

HIST = False
if '--hist' in sys.argv:
    HIST = True

PLOT = False
if '--plot' in sys.argv:
    PLOT = True

STRENGTH = False
if '--strength' in sys.argv:
    STRENGTH = True

np.random.seed()

lifetimes = np.random.uniform(.1,20.0,dimensions)
sources = [
    np.random.exponential(lifetime,samples) for lifetime in lifetimes
]
if HIST:
    for source in sources:
        plt.hist(source,50)
        plt.title('source hist')
        plt.show()

sources = np.vstack(sources)

mixing_matrix = np.random.uniform(1.,5.,(dimensions,dimensions))

signals = np.dot(mixing_matrix,sources)

if HIST:
    for signal in signals:
        plt.hist(signal,50)
        plt.title('mixed signal hist')
        plt.show()

centered = signals - np.average(signals,1).reshape(dimensions,1)

cov = np.cov(centered)

eigenvalues, eigenvectors = np.linalg.eig(cov)

E = eigenvectors
D = np.diag(1. / np.sqrt(eigenvalues))

whitened = E.dot(D).dot(E.transpose()).dot(centered)

components = []

def F(x):
    return x * np.exp(-1. * np.square(x) / 2.)

def f(x):
    return (1. - np.square(x)) * np.exp(-1. * np.square(x) / 2.)

for dimension in range(dimensions):
    old_guess = np.random.uniform(-1.,1.,dimensions)
    old_guess = old_guess - sum( 
        [
            component.transpose().dot(old_guess) * component for component in components
        ],
        np.zeros(dimensions)
    )
    old_guess = old_guess / np.linalg.norm(old_guess)
    iterations = 0
    while True:
        iterations += 1
        new_guess = np.average(
            whitened * F(
                old_guess.transpose().dot(whitened)
            ).reshape(1,samples),
            1
        ) - np.average(
            f(
                old_guess.transpose().dot(whitened)
            )            
        ) * old_guess
        new_guess = new_guess - sum( 
            [
                component.transpose().dot(new_guess) * component for component in components
            ],
            np.zeros(dimensions)
        )
        new_guess = new_guess / np.linalg.norm(new_guess)
        delta_pos = np.linalg.norm(new_guess - old_guess)
        delta_neg = np.linalg.norm(new_guess + old_guess)
        old_guess = new_guess
        if delta_pos < delta or delta_neg < delta:
            break        
    components.append(old_guess)
    print 'dimension %d found on %d iterations' % (dimension + 1,iterations)

extracted_signals = np.vstack(components).dot(whitened)

if HIST:
    for extracted_signal in extracted_signals:
        plt.hist(extracted_signal,50)
        plt.show()
image = []
for source_index, source in enumerate(sources):    
    image_row = []
    for extracted_signal_index, extracted_signal in enumerate(extracted_signals):
        cov = np.cov(
            np.vstack(
                [
                    source,
                    extracted_signal
                ]
            )
        )
        corr = np.abs(cov[1][0]/np.sqrt(cov[1][1]*cov[0][0]))
        print source_index, extracted_signal_index, corr
        image_row.append(corr)
    if PLOT:
        max_index = image_row.index(max(image_row))
        plt.plot(source[:50]/np.std(source))
        plt.plot(extracted_signals[max_index][:50])
        plt.show()
    image.append(image_row)
plt.pcolor(np.array(image).transpose())
if STRENGTH:
    print lifetimes
    signal_strength = np.square(mixing_matrix).sum(0)*lifetimes
    print signal_strength
    plt.plot(np.arange(dimensions)+.5,lifetimes/lifetimes.max()*dimensions,'wo',label='lifetimes')
    plt.plot(np.arange(dimensions)+.5,signal_strength/signal_strength.max()*dimensions,'ks',label='source strength')
    plt.legend()
plt.xlabel('source signal')
plt.ylabel('extracted signal')
plt.title('correlation coefficients')
plt.colorbar()
if STRENGTH:
    plt.ylim(0.,3./2.*dimension)
plt.show()
