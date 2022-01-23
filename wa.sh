#! /bin/bash
set -e
export MPLBACKEND=Agg
python wa.py
python wa.py vax
