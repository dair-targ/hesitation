import sys
import optparse
import os.path
import librosa
import numpy as np


def build_parser():
  parser = optparse.OptionParser(
    '%prog [options] WAVFILE'
  )
  parser.add_option(
    '-v', '--verbose',
    dest='verbose',
    action='store_true',
    default=False,
    help='Write what`s going on',
  )
  parser.add_option(
    '-p',
    '--hop',
    dest='hop',
    action='store',
    type='float',
    default=0.01,
    help='Time interval between two adjusted instant specters',
  )
  parser.add_option(
    '-o',
    '--output',
    dest='output',
    action='store',
    help='Output .npy file',
  )
  return parser


def run(input_path, hop, output_path, log):
  log('reading from %s' % input_path)
  
  y, sr = librosa.load(input_path)
  log('sample rate: %d' % sr)
  log('samples: %d' % len(y))
  log('duration: %.1f sec' % (float(len(y)) / float(sr)))

  hop_length = int(sr * hop)
  log('hop length: %d samples' % hop_length)
  log('number of spectres: %d' % int(len(y) / float(hop_length) + 1))

  y_stft = librosa.stft(
    y,
    hop_length=hop_length,
  )
  re_y_stft = np.transpose(np.abs(y_stft) ** 2)
  
  log('writing to %s' % output_path)
  np.save(output_path, re_y_stft)


def main():
  parser = build_parser()
  options, args = parser.parse_args()
  
  if options.verbose:
    def log(s):
      sys.stderr.write(s + '\n')
  else:
    log = lambda s: None
  
  if not args:
    parser.error('You must specify a .wav file to saw')
  path = os.path.abspath(args[0])
  
  if not options.output:
    parser.exit('You must specify the output file')
  output_path = os.path.abspath(options.output)
  
  run(path, options.hop, output_path, log)


if __name__ == '__main__':
  main()
