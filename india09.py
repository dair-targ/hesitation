import logging
import optparse
import math

import os.path
from matplotlib import pyplot, mlab
import numpy
import numpy.fft
from scipy.io import wavfile
from scipy import signal
from scikits import talkbox
import librosa.feature

def main():
    parser = optparse.OptionParser()
    parser.add_option(
        '--nfft',
        dest='nfft',
        action='store',
        type='int',
        help='NFFT',
        default=1024,
    )
    parser.add_option(
        '--step',
        dest='step',
        action='store',
        type='int',
        help='Number of measurements between frames',
        default=160,
    )
    parser.add_option(
        '--min-duration',
        dest='minimum_duration',
        action='store',
        type='float',
        help='(sec)',
        default=0.150,
    )
    parser.add_option(
        '--max-gap',
        dest='maximum_gap',
        action='store',
        type='float',
        help='(sec)',
        default=0.02,
    )
    options, args = parser.parse_args()

    input_path = args[0]
    logging.info('Reading %s', input_path)

    rate, data = wavfile.read(input_path)
    frame_length = float(options.step) / float(rate)
    logging.info('Sample length: %.2fsec', float(len(data)) / float(rate))
    logging.info('Frame length: %.2fsec', frame_length)
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

    logging.info('MFCC')
    N = 10
    mfcc_sequencies = []
    for t, midpoint in enumerate(midpoints):
        if not vad[t]:
            formants.append([])
            continue
        # Frame index
        fi = t * rate / options.step
        sample = data[fi - options.nfft / 2:fi + options.nfft / 2 + 1]
        mfcc_sequencies.append(librosa.feature.mfcc(sample, rate, n_mfcc=13))
    exit()

    logging.info('Decision Combination')
    threshold_1 = 40.0
    threshold_2 = threshold_1 * 2.0
    frame_candidate = [
        F1SD[t] < threshold_1 and
        F2SD[t] < threshold_2 and
        vad[t]
        for t, _ in enumerate(midpoints)
    ]
    # midpoint_candidates = [midpoints[t] for t, v in enumerate(fp_candidate) if v]
    # pyplot.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    # for midpoint_candidate in midpoint_candidates:
    #     pyplot.axvline(midpoint_candidate, alpha=0.2)
    # pyplot.show()
    # exit()

    logging.info('Duration Constraint')
    sequences = []
    last_sequence = []
    for t, v in enumerate(frame_candidate):
        if not v:
            continue
        if last_sequence:
            previous_t = last_sequence[-1]
            if midpoints[t] - midpoints[previous_t] > options.maximum_gap:
                sequences.append(last_sequence)
                last_sequence = [t]
            else:
                last_sequence.append(t)
        else:
            last_sequence.append(t)
    sequences.append(last_sequence)
    sequences = filter(lambda s: len(s) > 15, sequences)
    filled_pauses = map(lambda sequence: (midpoints[sequence[0]], midpoints[sequence[-1]]), sequences)

    # pyplot.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    # for sequence in sequences:
    #     pyplot.axvspan(midpoints[sequence[0]], midpoints[sequence[-1]], alpha=0.5)
    # pyplot.show()
    # exit()

    logging.info('Filled Pause Estimates')
    fname, fext = os.path.splitext(input_path)
    with open(fname + '.seg') as f:
        for line in f:
            if line.strip() == '[LABELS]':
                break
        hesitation_start = None
        hesitations = []
        for line in f:
            position, level, label = line.split(',', 2)
            position = int(position) / 2
            if label == 'End File':
                break
            elif 'h.e' in label:
                hesitation_start = position
            elif hesitation_start is not None:
                hesitations.append((hesitation_start, position))
                hesitation_start = None

    pyplot.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    for hesitation in hesitations:
        pyplot.axvspan(hesitation[0] / float(rate), hesitation[1] / float(rate), alpha=0.5)
    for k in xrange(2):
        pyplot.plot(midpoints, F(k))
    pyplot.plot(midpoints, F1SD)
    pyplot.plot(midpoints, F2SD)
    pyplot.show()
    exit()


    # pyplot.axvline(x=6.0)
    # pyplot.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    # pyplot.plot(midpoints, vad / max(vad) * 8000.0)
    # pyplot.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
