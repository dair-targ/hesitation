#!/usr/bin/python2.7

import logging
import optparse
import math
import numpy
from matplotlib import mlab
from librosa import core


REF_Hz = 440.0 * 2**(3.0 / 12.0 - 5.0)


def f_cent(f_hz):
    return 1200.0 * math.log(f_hz / REF_Hz, 2.0)


def G(x, m, sigma):
    sqr_sigma = 2.0 * sigma**2.0
    return math.exp(-((x - m)**2.0 / sqr_sigma)**2.0) * (math.pi * sqr_sigma)**-0.5


def p(x, F, N=8, H=0.97, W_1=20.0):
    return sum(H**h * G(x, F + 1200*math.log(h + 1.0, 2.0), W_1) for h in xrange(N))


def P_F0(F, Psi_p, stable_frequencies):
    return sum(p(x, F) * Psi_p[ix] for ix, x in stable_frequencies)


def main():
    parser = optparse.OptionParser()
    parser.add_option('--nfft', dest='nfft', action='store', type='int', help='NFFT', default=1024)
    parser.add_option('--step', dest='step', action='store', type='int', help='Number of measurements between frames', default=160)
    options, args = parser.parse_args()

    input_path = args[0]
    logging.info('Reading %s', input_path)

    data, rate = core.load(input_path)
    logging.info('Sample length: %.2fsec', float(len(data)) / float(rate))

    logging.info('Calculating specgram...')
    spectrum, frequencies, midpoints = mlab.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    spectrum = numpy.transpose(spectrum)
    cent_frequencies = numpy.array([0.0] + map(f_cent, frequencies[1:]))

    logging.info('Calculating instantaneous frequencies...')
    instantaneous_frequencies, short_time_fourier_transform = core.ifgram(data, rate, n_fft=options.nfft, hop_length=options.step)
    instantaneous_frequencies = numpy.transpose(instantaneous_frequencies)
    short_time_fourier_transform = numpy.transpose(short_time_fourier_transform)

    logging.info('Estimating Fundamental Frequency...')
    F0s = []
    delta = instantaneous_frequencies - frequencies
    for t, midpoint in enumerate(midpoints):
        logging.info('Calculating for %f', midpoint)
        stable_frequencies = []
        delta_t = delta[t]
        for k in xrange(len(delta_t) - 1):
            if delta_t[k] > 0 and delta_t[k + 1] < 0:
                stable_frequencies.append((k, delta_t[k]))
        best_F = frequencies[1]
        best_P0 = P_F0(best_F, short_time_fourier_transform[t], stable_frequencies)
        for iF, F in enumerate(frequencies[2:]):
            candidate_P_F0 = P_F0(F, short_time_fourier_transform[t], stable_frequencies)
            if candidate_P_F0 > best_P0:
                best_F = F
                best_P0 = candidate_P_F0
        F0s.append(best_F)
    logging.info('Done')



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
