"""
Plot given data or its slice.
"""

import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot
import optparse
import os

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
    '-o',
    '--output',
    dest='output',
    action='store',
    help='Output .npy file',
  )
  return parser


def run(input_path, output_path, log):
  log('Loading from %s' % input_path)
  if input_path.endswith('.npy'):
    log('Loading as npy')
    data = np.load(input_path)
  pyplot.imshow(np.transpose(np.log(data)))
  pyplot.show()


def main():
  parser = build_parser()
  options, args = parser.parse_args()
  
  if options.verbose:
    def log(s):
      sys.stderr.write(s + '\n')
  else:
    log = lambda s: None
  
  if not args:
    parser.error('You must specify a file to plot')
  path = os.path.abspath(args[0])
  
  if not options.output:
    parser.exit('You must specify the output file')
  output_path = os.path.abspath(options.output)
  
  run(path, output_path, log)


if __name__ == '__main__':
  main()
