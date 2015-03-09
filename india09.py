import logging
import optparse
import math

from matplotlib import pyplot, mlab
import numpy
from scipy.io import wavfile
from scipy import signal
from scikits import talkbox

def main():
    parser = optparse.OptionParser()
    parser.add_option('--nfft', dest='nfft', action='store', type='int', help='NFFT', default=1024)
    parser.add_option('--step', dest='step', action='store', type='int', help='Number of measurements between frames', default=160)
    options, args = parser.parse_args()

    input_path = args[0]
    logging.info('Reading %s', input_path)

    rate, data = wavfile.read(input_path)
    logging.info('Sample length: %.2fsec', float(len(data)) / float(rate))
    spectrum, frequencies, midpoints = mlab.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    spectrum = numpy.transpose(spectrum)

    logging.info('VAD')
    power = numpy.array([
        sum(numpy.abs(spectrum[t]))
        for t, midpoint in enumerate(midpoints)
    ])
    limit = numpy.percentile(power, 5) * 3.0
    vad = numpy.array([p > limit for p in power])

    logging.info('Formant Computation')
    # t = int(6.0 * rate / float(options.step))
    # sample = data[t:t + options.step]
    # lfiltered_data = signal.lfilter([1., 0.63], 1, sample * numpy.hamming(len(sample)))
    # a, e, k = talkbox.lpc(lfiltered_data, order=2 + rate / 1000)
    # roots = [r for r in numpy.roots(a) if numpy.imag(r) >= 0.0]
    # angles = numpy.arctan2(numpy.imag(roots), numpy.real(roots))
    # formants = sorted(angles * (rate / (2 * math.pi)))[:2]
    # map(pyplot.axvline, formants)
    #
    # pyplot.plot(frequencies, numpy.log10(spectrum[t]))
    # pyplot.show()
    # exit()

    F1 = []
    F2 = []
    delta = options.step / 2
    order = int(2 + rate / 1000)
    for t, midpoint in enumerate(midpoints):
        middle = int(midpoint * rate)
        sample = data[middle - delta:middle + delta + 1]
        if vad[t] and len(sample) >= order:
            lfiltered_data = signal.lfilter([1.0], [1.0, -0.63], sample * numpy.hamming(len(sample)))
            a, e, k = talkbox.lpc(lfiltered_data, order=order)
            roots = [r for r in numpy.roots(a) if numpy.imag(r) >= 0.0]
            angles = numpy.arctan2(numpy.imag(roots), numpy.real(roots))
            formants = sorted(angles * (rate / (2 * math.pi)))
            F1.append(formants[0])
            F2.append(formants[1])
        else:
            F1.append(0.0)
            F2.append(0.0)

    pyplot.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    pyplot.plot(midpoints, F1)
    pyplot.plot(midpoints, F2)
    pyplot.show()

    logging.info('LLR Computation for F1SD and F2SD')
    W = 11
    F1SD = [0.0] * (W / 2)
    F2SD = [0.0] * (W / 2)
    for t, midpoint in enumerate(midpoints):
        if t < W / 2: continue
        if t > len(midpoints) - W / 2: continue
        F1SD.append(numpy.std(F1[t - W/2:t + W/2 + 1]))
        F2SD.append(numpy.std(F2[t - W/2:t + W/2 + 1]))
    F1SD += [0.0] * (W / 2 - 1)
    F2SD += [0.0] * (W / 2 - 1)

    # pyplot.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    # print len(midpoints)
    # print len(F1SD)
    # pyplot.plot(midpoints, F1SD)
    # pyplot.plot(midpoints, F2SD)
    # pyplot.show()

    logging.info('Decision Combination')
    logging.info('Duration Constraint')
    logging.info('Filled Pause Estimates')

    # pyplot.axvline(x=6.0)
    # pyplot.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    # pyplot.plot(midpoints, vad / max(vad) * 8000.0)
    # pyplot.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
