import logging
import optparse
import math

from matplotlib import pyplot, mlab
import numpy
import numpy.fft
import numpy.linalg
from scipy.io import wavfile
from scipy import signal, optimize, stats
from scikits import talkbox
import librosa.feature

import seg_file


def plot_hesitations(hesitations, **kwargs):
    for hesitation in hesitations:
        pyplot.axvspan(hesitation[0], hesitation[1], **kwargs)


def normed(a, start, end):
    low = min(a)
    high = max(a)
    return (a - low) / (high - low) * (end - start) + start


def extract_sequencies(frame_candidate, midpoints, options):
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
    return filled_pauses


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
        default=0.100,
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

    pyplot.title(input_path)

    logging.info('Loading manual hesitations...')
    hesitations = seg_file.load_manual_hesitations(input_path, rate)

    logging.info('VAD')
    power = numpy.array([
        sum(numpy.abs(spectrum[t]))
        for t, midpoint in enumerate(midpoints)
    ])
    limit = numpy.percentile(power, 5) * 3.0
    vad = numpy.array([p > limit for p in power])

    logging.info('Formant Computation')
    n_formants = 2
    formants = []
    order = 2 + rate / 1000
    for t, midpoint in enumerate(midpoints):
        if not vad[t]:
            formants.append([0.0] * n_formants)
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
        angles = sorted(angles)
        if len(angles) < n_formants:
            angles += [0.0] * (n_formants - len(angles))
        formants.append(angles)
    F1 = [formants[t][0] for t, _ in enumerate(midpoints)]
    F2 = [formants[t][1] for t, _ in enumerate(midpoints)]

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


    logging.info('MFCC')
    mfcc_sequencies = numpy.transpose(librosa.feature.mfcc(data, rate, n_mfcc=20, hop_length=160))[:len(midpoints)]
    N = 10
    mu = numpy.roll(
        numpy.array([
            numpy.roll(mfcc_sequencies, k)
            for k in xrange(N)
        ]).mean(axis=0),
        shift=-N / 2,
    )

    assert mu.shape == mfcc_sequencies.shape
    normes_of_mfss_sequencies = numpy.linalg.norm(mfcc_sequencies, axis=1)
    assert normes_of_mfss_sequencies.shape == midpoints.shape
    normes_of_deltas = numpy.linalg.norm(mfcc_sequencies - mu, axis=1)
    assert normes_of_deltas.shape == midpoints.shape

    cepstral_instability = numpy.roll(
        numpy.sum((numpy.roll(normes_of_deltas, k, axis=0) for k in xrange(N)), axis=0),
        shift=-N / 2,
        axis=0,
    ) / normes_of_mfss_sequencies
    W = 11
    CISD = [0.0] * (W / 2)
    for t, midpoint in enumerate(midpoints):
        if t < W / 2: continue
        if t > len(midpoints) - W / 2: continue
        CISD.append(numpy.std(cepstral_instability[t - W/2:t + W/2 + 1]))
    CISD += [0.0] * (W / 2 - 1)
    CISD = numpy.array(CISD)

    logging.info('Duration Constraint')
    logging.info('Decision Combination')
    fsd_filled_pauses = extract_sequencies(
        numpy.array([
            F1SD[t] < 40.0 and
            F2SD[t] < 80.0
            for t, _ in enumerate(midpoints)
        ]) * vad,
        midpoints=midpoints,
        options=options)
    mfcc_filled_pauses = extract_sequencies(
        numpy.array([
            v < 0.05
            for v in CISD
        ]) * vad,
        midpoints=midpoints,
        options=options)

    logging.info('Writing data...')
    # seg_file.save_hesitations(input_path, 'india09', filled_pauses)

    pyplot.specgram(data, NFFT=options.nfft, Fs=rate, noverlap=options.nfft - options.step)
    plot_hesitations(fsd_filled_pauses, alpha=0.5, color='#A4A4A4')
    plot_hesitations(mfcc_filled_pauses, alpha=0.5, color='#2E9AFE')
    plot_hesitations(hesitations, alpha=0.5, color='#DF01D7')
    pyplot.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
