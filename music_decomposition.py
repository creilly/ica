import numpy as np
from matplotlib import pyplot as plt
import sys
import wave
import struct
import os
from ica import extract_sources

samples = 195000 # how many music samples per track to analyze (max 200000)
bins = 500 # number of bins in histgrams
sound_program = 'aplay' # program that plays wav files (change this for your os)

# command line argument to show histograms of the source, signal, and extracted data (shows 3*dimensions number of plots)
HIST = False
if '--hist' in sys.argv:
    HIST = True

# command line argument to show how similar the extracted waveforms are to source waveforms (shows dimensions number of plots)
PLOT = False
if '--plot' in sys.argv:
    PLOT = True

# plays the files as they are generated (uses command specified in sound_program variable)
PLAY = False
if '--play' in sys.argv:
    PLAY = True

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
    wave_write.close()
    if PLAY:
        # issue terminal command to play song
        os.system(
            '%s %s' % (
                sound_program,fname
            )
        )

if HIST:
    # plot mixed signal histograms
    for signal in signals:
        plt.hist(signal,bins)
        plt.title('mixed signal hist')
        plt.show()


extracted_signals = extract_sources(signals)

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
    wave_write.close()        
    if PLAY:
        os.system(
            '%s %s' % (
                sound_program,fname
            )
        )

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
