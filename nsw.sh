#! /bin/bash
set -e
export MPLBACKEND=Agg
python nsw.py
python nsw.py noniso
python nsw.py vax
python nsw.py accel_vax
