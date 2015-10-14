#!/usr/bin/python2.7
"""
Performs Short-Time Fourier Transformation of the given file
"""

import os.path
import argparse

def _create_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'input',
    metavar='input.wav',
    help='Input .wav file',
  )
  parser.add_argument(
    'output',
    nargs='?',
    metavar='output.sftf.npy',
  )
  return parser


def main():
  parser = _create_parser()
  args = parser.parse_args()

  input_path = os.path.abspath(args.input)
  print 'Input path: %s' % input_path
  
  output_path = os.path.abspath(
    args.output or (os.path.splitext(args.input)[0] + '.stft.npy'))
  print 'Output path: %s' % output_path
  
  data, sr = librosa.load(args.input)
  stft_data = librosa.stft(data, hop_length=hop_length)
  prepared_data = librosa.logamplitude(np.abs(stft_data)**2, ref_power=np.max)


if __name__ == '__main__':
  main()
