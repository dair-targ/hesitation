#!/usr/bin/python2.7
"""
wavsaw-split --chunk=10s input.wav output
Creates an ${output} directory and produces chunks of wav format
with the specific length.
"""

import os
import argparse
import librosa
from units import unit

Sample = unit('sample')
Second = unit('s')
Frame = unit('f')

def _create_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--chunk',
    dest='chunk',
    default='1000f',
    help='Chunk length. Available units: frames (f).'
  )
  parser.add_argument(
    '--hop',
    dest='hop',
    default='100',
    help='Hop length (samples)',
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
  
  output_path = os.path.abspath(args.output)
  
  print 'Sample rate:', sr
  print 'Total duration: %s' % (total_duration_samples / sr)
  
  frame_size = Sample(1 + n_fft / 2) / Frame(1)
  print 'Frame size: %s' % frame_size
  hop_length = Sample(int(args.hop))
  print 'Hop length: %s' % hop_length
  
  chunk_size = args.chunk
  if chunk_size.endswith('f'):
    chunk_size_frames = Frame(int(chunk_size[:-1]))
  elif chunk_size.endswith('s'):
    chunk_size_seconds = Second(int(chunk_size[:-1]))
    chunk_size_frames = Frame(int((chunk_size_seconds * sr - frame_size * Frame(1)) / hop_length) + 1)
  chunk_size_samples = frame_size * Frame(1) + (chunk_size_frames - Frame(1)) / Frame(1) * hop_length
  print 'Chunk size: %d frames' % chunk_size_frames
  print 'Chunk size: %d samples' % chunk_size_samples
  
  for chunk_index, start_sample in enumerate(xrange(0, len(data), (chunk_size_samples - frame_size * Frame(1) + hop_length) / Sample(1))):
    start_sample = Sample(start_sample)
    chunk_data = data[start_sample / Sample(1):(start_sample + chunk_size_samples) / Sample(1)]
    output_filename = os.path.join(output_path, '%d.wav' % chunk_index)
    print 'Writing chunk #%d[%d..%d] to %s' % (chunk_index, start_sample, start_sample + Sample(len(chunk_data)), output_path)
    #librosa.output.write_wav('', chunk_data, sr)


if __name__ == '__main__':
  main()
