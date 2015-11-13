import numpy as np
from matplotlib import pyplot as plt
import sys
import wave
import struct
from scipy.io import wavfile
import os

samples = 190000
delta = 1.e-6

wave_files = [
    'africa.wav',
    'dont-speak.wav',
    'mambo-no-5.wav',
    'i-ran-so-far-away.wav'
]
sources = []
for index, wave_file in enumerate(wave_files):
    wave_read = wave.Wave_read(wave_file)
    data = np.array(
        [
            float(
                struct.unpack(
                    '<h',
                    wave_read.readframes(1)
                )[0]
            ) for frame in range(samples)
        ]
    )
    data = data * {
        0:3.,
    }.get(index,1.)
    sources.append(data) 

dimensions = len(wave_files)

HIST = False
if '--hist' in sys.argv:
    HIST = True

PLOT = False
if '--plot' in sys.argv:
    PLOT = True

STRENGTH = False
if '--strength' in sys.argv:
    STRENGTH = True

PLAY = False
if '--play' in sys.argv:
    PLAY = True

np.random.seed()

# lifetimes = np.random.uniform(.1,20.0,dimensions)
# sources = [
#     np.random.exponential(lifetime,samples) for lifetime in lifetimes
# ]

if HIST:
    for source in sources:
        plt.hist(source,50)
        plt.title('source hist')
        plt.show()

# sources = np.vstack(sources)
sources = np.array(sources)

mixing_matrix = np.random.uniform(1.,5.,(dimensions,dimensions))

signals = np.dot(mixing_matrix,sources)
for index, signal in enumerate(signals):
    fname = 'mixed_%d.wav' % (index+1)
    signal = signal - signal.min()
    signal = (signal / signal.max() * 2**16 - 2**15).astype('i2')
    wave_write = wave.Wave_write(fname)
    wave_write.setsampwidth(2)
    wave_write.setframerate(wave_read.getframerate())
    wave_write.setnchannels(1)
    for sample in signal:
        wave_write.writeframes(
            struct.pack('h',sample)
        )
    if PLAY:
        os.system('aplay %s' % fname)

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

for index, signal in enumerate(extracted_signals):
    fname = 'extracted_%d.wav' % (index+1)
    signal = signal - signal.min()
    signal = (signal / signal.max() * 2**16 - 2**15).astype('i2')
    wave_write = wave.Wave_write(fname)
    wave_write.setsampwidth(2)
    wave_write.setframerate(wave_read.getframerate())
    wave_write.setnchannels(1)
    for sample in signal:
        wave_write.writeframes(
            struct.pack('h',sample)
        )
    if PLAY:
        os.system('aplay %s' % fname)

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
