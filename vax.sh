#! /bin/bash
set -e
export MPLBACKEND=Agg
python aus_vax.py
python aus_vax.py project
