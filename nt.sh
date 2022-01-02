#! /bin/bash
set -e
export MPLBACKEND=Agg
python nt.py
python nt.py vax
