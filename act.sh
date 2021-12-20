#! /bin/bash
set -e
export MPLBACKEND=Agg
python act.py
python act.py vax
# python act_vax.py
