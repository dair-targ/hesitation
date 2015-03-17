#!/bin/bash

cd $(dirname $0)
find $1 -name '*.wav' | xargs -n 1 python2.7 ./india09.py
tar cvfz $2 -- $(find $2 -name '*.txt')
