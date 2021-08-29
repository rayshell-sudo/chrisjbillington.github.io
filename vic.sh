#! /bin/bash
set -e
export MPLBACKEND=Agg
python vic-2021.py
# python vic-2021.py noniso
python vic-2021.py vax
python vic_vax.py
