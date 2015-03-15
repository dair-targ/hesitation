import optparse
import os
from scipy.io import wavfile


def load_manual_hesitations(input_path, rate):
    fname, fext = os.path.splitext(input_path)
    with open(fname + '.seg') as f:
        for line in f:
            if line.strip() == '[LABELS]':
                break
        hesitation_start = None
        hesitations = []
        for line in f:
            position, rest = line.split(',', 1)
            position = int(position) / 2
            if 'End File' in rest:
                break
            elif 'h.e' in rest:
                hesitation_start = position
            elif hesitation_start is not None:
                hesitations.append((hesitation_start, position))
                hesitation_start = None
    return [(hesitation[0] / float(rate), hesitation[1] / float(rate)) for hesitation in hesitations]


def save_hesitations(input_path, infix, hesitations):
    rate, data = wavfile.read(input_path)
    fname, fext = os.path.splitext(input_path)
    with open(fname + '_' + infix + '.seg', 'w') as f:
        f.write('[PARAMETERS]\n')
        f.write('SAMPLING_FREQ=%d\n' % rate)
        f.write('BYTE_PER_SAMPLE=2\n')
        f.write('CODE=0\n')
        f.write('N_CHANNEL=1\n')
        f.write('N_LABEL=%d\n' % (2 * (len(hesitations) + 1)))
        f.write('[LABELS]\n')
        f.write('0,1,Begin File\n')
        for hesitation in hesitations:
            f.write('%d,2,Hesitation Start' % int(hesitation[0] * rate))
            f.write('%d,2,Hesitation End' % int(hesitation[1] * rate))
        f.write('%d,1,End File\n' % len(data))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    options, args = parser.parse_args()

    input_path = args[0]
    hesitations = load_manual_hesitations(input_path, 1.0)
    print len(hesitations)
