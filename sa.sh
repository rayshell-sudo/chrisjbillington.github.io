#! /bin/bash
set -e
export MPLBACKEND=Agg
python sa.py
python sa.py vax
