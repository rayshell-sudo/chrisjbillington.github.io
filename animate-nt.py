from subprocess import check_call
from pathlib import Path
from datetime import datetime


start_date = datetime.fromisoformat('2022-01-01')
today = datetime.today()
n_days = (today - start_date).days

Path('nt_animated').mkdir(exist_ok=True)
Path('nt_animated_linear').mkdir(exist_ok=True)

for i in range(n_days + 1):
    print(i)
    check_call(['python', 'nt.py', 'old', str(i)])

for name in ['nt_animated', 'nt_animated_linear']:
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
