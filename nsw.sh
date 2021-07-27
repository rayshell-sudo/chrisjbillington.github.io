#! /bin/bash
set -e
export MPLBACKEND=Agg
python nsw.py
python nsw.py noniso
python nsw.py vax
python nsw.py accel_vax
python nsw.py Fairfield
python nsw.py Canterbury-Bankstown
python nsw.py Cumberland
python nsw.py Liverpool
python nsw.py Blacktown
python nsw.py Waverley
python nsw.py 'Georges River'
python nsw.py Randwick
python nsw.py Bayside
python nsw.py Parramatta
python nsw.py Sydney
