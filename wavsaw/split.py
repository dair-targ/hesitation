#!/usr/bin/python2.7
"""
Splits the input .wav file into an even chunks and writes them to the output directory.
"""

import os
import argparse
import librosa
from units import unit

Sample = unit('sample')
Second = unit('s')
Frame = unit('f')

def _create_parser():
  unit_description = '\nAvailable units: frames (f), seconds (s), samples ().\nDefault %(default)s.'
  parser = argparse.ArgumentParser(
    description=__doc__
  )
  parser.add_argument(
    '--chunk',
    dest='chunk',
    default='10s',
    help='Chunk length.' + unit_description,
  )
  parser.add_argument(
    '--hop',
    dest='hop',
    default='100',
    help='Hop length.' + unit_description,
  )
  parser.add_argument(
    'input',
    metavar='input.wav',
    help='Input .wav file',
  )
  parser.add_argument(
    'output',
    nargs='?',
    help='Output directory (by default is equals to name of input file)',
  )
  return parser


def parse_value(value, s2samples, f2samples):
  if value.endswith('s'):
    value = s2samples(Second(float(value[:-1])))
  elif value.endswith('f'):
    value = f2samples(Frame(float(value[:-1])))
  else:
    value = Sample(int(float(value)))
  return value


def main():
  parser = _create_parser()
  args = parser.parse_args()
  n_fft = 2048
  hop_length = int
  
  input_path = os.path.abspath(args.input)
  print 'Reading from', input_path, '...'
  data, sr = librosa.load(args.input)
  
  total_duration_samples = Sample(float(len(data)))
  sr = Sample(sr) / Second(1.0)
  
  print 'Sample rate:', sr
  print 'Total duration: %s' % (total_duration_samples / sr)
  
  frame_size = Sample(1 + n_fft / 2) / Frame(1)
  print 'Frame size: %s' % frame_size
  
  hop_length = parse_value(
    args.hop,
    s2samples=lambda s: s * sr,
    f2samples=lambda f: f * frame_size,
  )
  print 'Hop length: %s' % hop_length
  
  chunk_s2f = lambda s: Frame(int((s * sr - frame_size * Frame(1)) / hop_length) + 1)
  chunk_f2samples = lambda f: frame_size * Frame(1) + (f - Frame(1)) / Frame(1) * hop_length
  chunk_size_samples = parse_value(
    args.chunk,
    s2samples=lambda s: chunk_f2samples(chunk_s2f(s)),
    f2samples=chunk_f2samples,
  )
  
  output_path = os.path.abspath(args.output or os.path.splitext(input_path)[0])
  print 'Writing to %s/' % output_path
  if not os.path.exists(output_path):
    os.mkdir(output_path)
  
  for chunk_index, start_sample in enumerate(xrange(0, len(data), int(chunk_size_samples - frame_size * Frame(1) + hop_length) / Sample(1))):
    start_sample = Sample(start_sample)
    chunk_data = data[start_sample / Sample(1):(start_sample + chunk_size_samples) / Sample(1)]
    output_filename = os.path.join(output_path, '%d.wav' % chunk_index)
    action = 'Replacing' if os.path.exists(output_filename) else 'Writing'
    print '%s chunk #%d[%d..%d] to %s' % (action, chunk_index, start_sample, start_sample + Sample(len(chunk_data)), output_filename)
    librosa.output.write_wav(output_filename, chunk_data, sr)


if __name__ == '__main__':
  main()
