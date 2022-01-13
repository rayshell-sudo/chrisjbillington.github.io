#! /bin/bash
set -e
export MPLBACKEND=Agg
python vic-2021.py
python vic-2021.py vax
for i in {0..9}
do
   python vic-2021.py $i
done
# python vic_vax.py
