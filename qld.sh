#! /bin/bash
set -e
export MPLBACKEND=Agg
python qld.py
python qld.py vax
