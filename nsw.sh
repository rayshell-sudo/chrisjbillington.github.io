#! /bin/bash
set -e
export MPLBACKEND=Agg
python nsw.py
python nsw.py vax
for i in {0..11}
do
   python nsw.py $i
done
python nsw.py others
python nsw.py concern
python nsw.py sydney
python nsw.py notsydney
python nsw.py hunter
python nsw.py illawarra
python nsw.py wnsw
# python nsw.py others vax
# python nsw.py concern vax
# python nsw.py hunter vax
# python nsw.py illawarra vax
# python nsw.py wnsw vax
# python nsw.py bipartite
python nsw_vax.py
