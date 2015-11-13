import numpy as np
from matplotlib import pyplot as plt

dimensions = 3
samples = 50
delta = 1.e-12

np.random.seed()

endpoints = np.random.uniform(.1,1.0,dimensions)

sources = [
    np.random.uniform(-1. * endpoint,1. * endpoint,samples) for endpoint in endpoints
]

sources = np.vstack(sources)

mixing_matrix = np.random.uniform(1.,5.,(dimensions,dimensions))

signals = np.dot(mixing_matrix,sources)

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

for signal_index, signal in enumerate(signals):    
    for extracted_signal_index, extracted_signal in enumerate(extracted_signals):
        cov = np.cov(
            np.vstack(
                [
                    signal,
                    extracted_signal
                    ]
            )
        )
        print signal_index, extracted_signal_index, cov[1][0]/np.sqrt(cov[1][1]*cov[0][0])


