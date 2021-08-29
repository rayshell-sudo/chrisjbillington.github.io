#! /bin/bash
set -e
export MPLBACKEND=Agg
python nz.py
python nz.py vax
python nz_vax.py
