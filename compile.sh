#!/bin/bash
gcc -o gc_dynamics.o gc_dynamics.c -march=native -Ofast -lfftw3f_threads -lfftw3f -lziggurat -lm
