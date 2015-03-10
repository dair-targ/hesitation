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
    formants = []
    order = 2 + rate / 1000
    for t, midpoint in enumerate(midpoints):
        if not vad[t]:
            formants.append([])
            continue
        # Frame index
        fi = t * rate / options.step
        sample = data[fi - options.nfft / 2:fi + options.nfft / 2 + 1]
        x1 = sample * numpy.hamming(len(sample))
        lfiltered_sample = signal.lfilter([1.0], [1.0, 0.63], x1)
        if len(lfiltered_sample) < order:
            formants.append([])
            continue
        a, e, k = talkbox.lpc(lfiltered_sample, order=order)
        roots = filter(lambda v: numpy.imag(v) >= 0, numpy.roots(a))
        angles = numpy.arctan2(numpy.imag(roots), numpy.real(roots)) * rate / (2.0 * math.pi)
        formants.append(sorted(angles))
    F = lambda k: [formants[t][k] if len(formants[t]) > k else 0.0 for t, _ in enumerate(midpoints)]
    # pyplot.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    # for k in xrange(order):
    #     pyplot.plot(midpoints, F(k))
    # pyplot.show()
    # exit()
    F1 = F(0)
    F2 = F(1)

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
    # pyplot.plot(midpoints, F1SD)
    # pyplot.plot(midpoints, F2SD)
    # pyplot.show()
    # exit()

    logging.info('Decision Combination')
    threshold_1 = 40.0
    threshold_2 = threshold_1 * 2.0
    fp_candidate = [
        F1SD[t] < threshold_1 and
        F2SD[t] < threshold_2 and
        vad[t]
        for t, _ in enumerate(midpoints)
    ]

    # pyplot.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    # for t, v in enumerate(is_filled_pause):
    #     if v:
    #         pyplot.axvline(t)

    logging.info('Duration Constraint')
    logging.info('Filled Pause Estimates')

    # pyplot.axvline(x=6.0)
    # pyplot.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    # pyplot.plot(midpoints, vad / max(vad) * 8000.0)
    # pyplot.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
