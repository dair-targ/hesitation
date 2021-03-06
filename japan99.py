#!/usr/bin/python2.7

import logging
import optparse
import math
import numpy
from matplotlib import mlab, pyplot
from librosa import core
from scipy.io import wavfile

REF_Hz = 440.0 * 2**(3.0 / 12.0 - 5.0)


def f_cent(f_hz):
    return 1200.0 * math.log(f_hz / REF_Hz, 2.0)

def cent_to_hz(in_cent):
    return 2.0**(in_cent / 1200.0) * REF_Hz


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

    rate, data = wavfile.read(input_path)
    logging.info('Sample length: %.2fsec', float(len(data)) / float(rate))

    logging.info('Calculating specgram...')
    spectrum, frequencies, midpoints = mlab.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    spectrum = numpy.transpose(spectrum)
    cent_frequencies = numpy.array([0.0] + map(f_cent, frequencies[1:]))
    # pyplot.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    # pyplot.show()
    # exit()

    logging.info('Calculating instantaneous frequencies...')
    instantaneous_frequencies, short_time_fourier_transform = core.ifgram(data, rate, n_fft=options.nfft, hop_length=options.step)
    instantaneous_frequencies = numpy.transpose(instantaneous_frequencies)
    short_time_fourier_transform = numpy.transpose(short_time_fourier_transform)

    logging.info('Computing p(x, F)...')
    pXF = numpy.ndarray(shape=(len(frequencies), len(frequencies)), dtype='float64')
    for ix, x in enumerate(frequencies):
        for iF, F in enumerate(frequencies):
            pXF[ix][iF] = p(frequencies[ix], frequencies[iF])

    # t_x = int(8.1 * rate * 0.01)

    logging.info('Estimating Fundamental Frequency...')
    F_F0 = []
    Pow = []
    delta = instantaneous_frequencies - frequencies
    for t, midpoint in enumerate(midpoints[::100]):
        logging.info('Calculating for %f (#%d)', midpoint, t)
        stable_frequencies = []
        Psi_p = short_time_fourier_transform[t]
        delta_t = delta[t]
        for k in xrange(len(delta_t) - 1):
            if delta_t[k] > 0 > delta_t[k + 1]:
                stable_frequencies.append(k)
        P_F0 = lambda iF: sum(pXF[ix][iF] * Psi_p[ix] for ix in stable_frequencies)

        best_iF = 1
        best_P0 = P_F0(best_iF)
        for iF in xrange(len(frequencies[2:])):
            candidate_P_F0 = P_F0(iF)
            if candidate_P_F0 > best_P0:
                best_iF = iF
                best_P0 = candidate_P_F0
        F_F0.append(frequencies[best_iF])

        W2 = 35
        Pow.append([
            max(abs(G(frequencies[ix], F_F0[t] + 1200.0 * math.log(k, 2.0), W2) * Psi_p[ix]) for ix in xrange(1, len(frequencies)))
            for k in xrange(1, len(frequencies))
        ])

        # F_F0[-1]

        # pyplot.plot(frequencies[1:], Pow[-1])
    pyplot.imshow(Pow)
    pyplot.show()

    # pyplot.plot(F_F0)
    # pyplot.show()
    # Suppose F0s are calculated correctly...

    # logging.info('Estimating Spectral Envelope...')
    # W2 = 35
    # for t, midpoint in enumerate(midpoints[9:10]):
    #     logging.info('Calculating for %f', midpoint)
    #     Psi_p = short_time_fourier_transform[t]
    #     Pow_local = lambda
    #     Pow = lambda k: max(G(x, F_F0[t] + 1200.0 * math.log(k, 2.0), W2) * Psi_p[ix] for ix, x in stable_frequencies)
    # pyplot.axvline(f_cent(F0s[9]))
    # x = cent_frequencies
    # y = spectrum[10]
    # pyplot.plot(x, y)
    # pyplot.plot(x, mlab.normpdf(x, *stats.norm.fit(y)))
    # pyplot.show()
    # exit()

    # logging.info('Obtaining Two Features of Filled Pauses...')
    # pyplot.plot(frequencies, short_time_fourier_transform[10])
    # pyplot.show()
    # exit()
        R = 0.034
        W = 0.575
        P = math.exp(- (R * S_f + (1 - R) * S_s)**2.0 / W**2.0)




if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
