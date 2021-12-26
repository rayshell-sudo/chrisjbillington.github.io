#! /bin/bash
set -e
export MPLBACKEND=Agg
python tas.py
python tas.py vax
