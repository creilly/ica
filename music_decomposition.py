import numpy as np
from matplotlib import pyplot as plt
import sys
import wave
import struct
import os

samples = 190000 # how many music samples per track to analyze (max 200000)
delta = 1.e-10 # criterion for convergence
bins = 500

# these files should be in the local folder
wave_files = [
    'africa.wav',
    'dont-speak.wav',
    'mambo-no-5.wav',
    'i-ran-so-far-away.wav'
]

# this list will hold the numeric waveform data
sources = []

for wave_file in wave_files:
    wave_read = wave.Wave_read(wave_file)

    # unpack the binary .wav data
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

    # add waveform data to list of sources
    sources.append(data) 

dimensions = len(wave_files)

# command line argument to show histograms of the source, signal, and extracted data (shows 3*dimensions number of plots)
HIST = False
if '--hist' in sys.argv:
    HIST = True

# command line argument to show how similar the extracted waveforms are to source waveforms (shows dimensions number of plots)
PLOT = False
if '--plot' in sys.argv:
    PLOT = True

# plays the files as they are generated (uses aplay command, maybe mac has this?)
PLAY = False
if '--play' in sys.argv:
    PLAY = True

np.random.seed()

if HIST:
    # plot source histograms
    for source in sources:
        plt.hist(source,bins)
        plt.title('source hist')
        plt.show()

# stack source data into rectangular (2d) numpy array
sources = np.array(sources)

# constuct random mixing matrix
mixing_matrix = np.random.uniform(1.,5.,(dimensions,dimensions))

# mix together sources to get signals
signals = np.dot(mixing_matrix,sources)

# write each mixed signal to .wav file
for index, signal in enumerate(signals):
    fname = 'mixed_%d.wav' % (index+1)

    # scale data to lie between -2^15 and +2^15
    signal = signal - signal.min()
    signal = (signal / signal.max() * 2**16 - 2**15).astype('i2')

    # open handle to wave file object
    wave_write = wave.Wave_write(fname)

    # specify bits per sample
    wave_write.setsampwidth(2)

    # make sure framerate is same as original files
    wave_write.setframerate(wave_read.getframerate())

    # working with mono data
    wave_write.setnchannels(1)

    # write data to binary .wav format
    for sample in signal:
        wave_write.writeframes(
            struct.pack('h',sample)
        )

    if PLAY:
        # only works if system has aplay program
        os.system('aplay %s' % fname)

if HIST:
    # plot mixed signal histograms
    for signal in signals:
        plt.hist(signal,bins)
        plt.title('mixed signal hist')
        plt.show()

# make each signal have zero mean
centered = signals - np.average(signals,1).reshape(dimensions,1)

# compute expectation value of covariance matrix
cov = np.cov(centered)

# compute eigenvalue / eigenvector decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov)

E = eigenvectors

# construct diagonal matrix of inverse square root of eigenvalues
D = np.diag(1. / np.sqrt(eigenvalues))

# apply linear transformation on centered signal data to have unit covariance
whitened = E.dot(D).dot(E.transpose()).dot(centered)

# will store extracted components here
components = []

# first derivative of nonlinear nongaussianity-maximizing function (gaussian)
def F(x):
    return x * np.exp(-1. * np.square(x) / 2.)

# second derivative
def f(x):
    return (1. - np.square(x)) * np.exp(-1. * np.square(x) / 2.)

# there are as many components as there are dimensions
for dimension in range(dimensions):
    # make an initial guess
    old_guess = np.random.uniform(-1.,1.,dimensions)
    
    # project out any component along previously extracted components
    old_guess = old_guess - sum( 
        [
            component.transpose().dot(old_guess) * component for component in components
        ],
        np.zeros(dimensions)
    )
    
    # normalize
    old_guess = old_guess / np.linalg.norm(old_guess)
    
    # keep track of number of iterations it takes to converge
    iterations = 0
    while True:
        iterations += 1

        # compute improved component from old one
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

        # perform same projection / normalization as we did with first guess
        new_guess = new_guess - sum( 
            [
                component.transpose().dot(new_guess) * component for component in components
            ],
            np.zeros(dimensions)
        )
        new_guess = new_guess / np.linalg.norm(new_guess)

        # compute difference between new and old guess
        delta_pos = np.linalg.norm(new_guess - old_guess)

        # compute difference between new and negative of old guess
        delta_neg = np.linalg.norm(new_guess + old_guess)

        # set new guess to be old guess of next loop
        old_guess = new_guess

        # if old guess is "same" as new guess (i.e. within arbitrary negative sign) then we're done
        if delta_pos < delta or delta_neg < delta:
            break        
    # add extracted component to list
    components.append(old_guess)
    print 'dimension %d found on %d iterations' % (dimension + 1,iterations)

# compute extracted signals by applying extracted weights on whitened signal data
extracted_signals = np.vstack(components).dot(whitened)

if HIST:
    # plot extracted signal histograms
    for extracted_signal in extracted_signals:
        plt.hist(extracted_signal,bins)
        plt.show()

# write extracted waveforms to .wav files
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

# construct matrix of source / extracted correlation coefficients
image = []
for source_index, source in enumerate(sources):
    
    # store computed corr coeffs between current source and extracted signals in this list
    image_row = []
    
    for extracted_signal_index, extracted_signal in enumerate(extracted_signals):
        # compute expectation covariance matrix
        cov = np.cov(
            np.vstack(
                [
                    source,
                    extracted_signal
                ]
            )
        )
        # from covariance matrix get absolute value of corr coeff
        corr = np.abs(cov[1][0]/np.sqrt(cov[1][1]*cov[0][0]))
        print source_index, extracted_signal_index, corr
        # add to coeffs for current source
        image_row.append(corr)
        
    if PLOT:
        # find extracted signal most correlated with current source
        max_index = image_row.index(max(image_row))

        # plot first 50 data points of each and see how well they match
        plt.plot(source[:50]/np.std(source))
        plt.plot(extracted_signals[max_index][:50])        
        plt.show()
        
    # add current source's corr coeffs to corr coeff matrix
    image.append(image_row)

# plot corr coeff matrix
plt.pcolor(np.array(image).transpose())
plt.xlabel('source signal')
plt.ylabel('extracted signal')
plt.title('correlation coefficients')
plt.colorbar()
plt.show()
