#! /bin/bash
set -e
export MPLBACKEND=Agg
python nsw.py
# python nsw.py noniso
python nsw.py vax
# python nsw.py accel_vax
for i in {0..11}
do
   python nsw.py $i
done
python nsw.py others
python nsw.py concern
