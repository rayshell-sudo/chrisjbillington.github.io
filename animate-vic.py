from subprocess import check_call
from pathlib import Path
from datetime import datetime


# The first date I made any vaccine projections
start_date = datetime.fromisoformat('2021-08-29')
today = datetime.today()
n_days = (today - start_date).days

Path('vic_animated').mkdir(exist_ok=True)
Path('vic_animated_linear').mkdir(exist_ok=True)

for i in range(n_days + 1):
    print(i)
    check_call(['python', 'vic-2021.py', 'old', str(i)])

for name in ['vic_animated', 'vic_animated_linear']:
    check_call(
        [
            'ffmpeg',
            '-y',
            '-r',
            '4',
            '-i',
            f'{name}/%04d.png',
            '-r',
            '4',
            '-vf',
            'tpad=stop_mode=clone:stop_duration=2',
            f'{name}.webm',
        ]
    )

    # check_call(
    #     ['convert', '-delay', '25']
    #     + [f'{name}/{j:04d}.png' for j in range(n_days)]
    #     + [
    #         '-delay',
    #         '500',
    #         f'{name}/{n_days:04d}.png',
    #         f'{name}.gif',
    #     ],
    # )
