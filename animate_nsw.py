from subprocess import check_call
from pathlib import Path
from datetime import datetime


# The first date I made any vaccine projections
start_date = datetime.fromisoformat('2021-07-22')
today = datetime.today()
n_days = (today - start_date).days

Path('nsw_animated').mkdir(exist_ok=True)
Path('nsw_animated_linear').mkdir(exist_ok=True)

for i in range(n_days + 1):
    print(i)
    check_call(['python', 'nsw.py', 'old', str(i)])

DELAY = 3000
for name in ['nsw_animated', 'nsw_animated_linear']:
    check_call(
        ['convert', '-delay', '25']
        + [f'{name}/{j:04d}.png' for j in range(n_days)]
        + [
            '-delay',
            '500',
            f'{name}/{n_days:04d}.png',
            f'{name}.gif',
        ],
    )
