import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

converter = mdates.ConciseDateConverter()

munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime] = converter


def seven_day_average(data):
    n = 7
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def first_and_second_by_state(state):
    df = pd.read_html(
        f"https://covidlive.com.au/report/daily-vaccinations-first-doses/{state.lower()}"
    )[1]
    first = np.array(df['FIRST'][::-1])
    first_dates = np.array(
        [np.datetime64(datetime.strptime(d, '%d %b %y'), 'D') for d in df['DATE'][::-1]]
    )

    df = pd.read_html(
        f"https://covidlive.com.au/report/daily-vaccinations-people/{state.lower()}"
    )[1]
    second = np.array(df['SECOND'][::-1])
    second_dates = np.array(
        [np.datetime64(datetime.strptime(d, '%d %b %y'), 'D') for d in df['DATE'][::-1]]
    )

    first[np.isnan(first)] = 0
    second[np.isnan(second)] = 0
    maxlen = max(len(first), len(second))
    if len(first) < len(second):
        first = np.concatenate([np.zeros(maxlen - len(first)), first])
        dates = second_dates
    elif len(second) < len(first):
        second = np.concatenate([np.zeros(maxlen - len(second)), second])
        dates = second_dates
    else:
        dates = first_dates

    IX_CORRECTION = np.where(dates==np.datetime64('2021-07-29'))[0][0]

    if state.lower() in ['nt', 'act']:
        first[:IX_CORRECTION] += first[IX_CORRECTION] - first[IX_CORRECTION - 1]
        second[:IX_CORRECTION] += second[IX_CORRECTION] - second[IX_CORRECTION - 1]

    if first[-1] == first[-2]:
        first = first[:-1]
        dates = dates[:-1]
        second = second[:-1]

    return dates - 1, np.diff(first + second, prepend=0)

POPS = {
    'AUS': 25.36e6,
    'NSW': 8.166e6,
    'VIC': 6.681e6,
    'QLD': 5.185e6,
    'SA': 1.771e6,
    'WA': 2.667e6,
    'TAS': 541071,
    'NT': 246500,
}

STATE = 'QLD'

qld_dates, qld_doses = first_and_second_by_state(STATE)


days_projection = 200
t_projection = np.arange(qld_dates[-1], qld_dates[-1] + days_projection + 1)

SEP = np.datetime64('2021-09-01')
OCT = np.datetime64('2021-10-01')

qld_proj_rate = np.zeros(len(t_projection))

qld_proj_rate[:] =  0.3
# clip to 85% fully vaxed
initial_coverage =  100 * qld_doses.sum() / POPS[STATE]
qld_proj_rate[initial_coverage + qld_proj_rate.cumsum() > 2 * 80.0] = 0

plt.figure(figsize=(10, 5))
plt.subplot(121)
qld_rate = 100 * seven_day_average(qld_doses) / POPS[STATE]
plt.step(qld_dates, qld_rate, label="Actual")
plt.step(t_projection, qld_proj_rate, label="Assumed for projection")
plt.legend()
locator = mdates.DayLocator([1, 15])
formatter = mdates.ConciseDateFormatter(locator)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.grid(True, color='k', linestyle=":", alpha=0.5)
plt.axis(
    xmin=np.datetime64('2021-05-01'), xmax=np.datetime64('2022-02-28'), ymax=2.2, ymin=0
)
plt.ylabel('7d avg doses per hundred population per day')
plt.title(f"{STATE} daily vaccinations per capita")

plt.subplot(122)
qld_cumulative = 100 * qld_doses.cumsum() / POPS[STATE]
plt.step(qld_dates, qld_cumulative, label="Actual")
plt.step(
    t_projection,
    (qld_proj_rate.cumsum() + qld_cumulative[-1]).clip(0, 2 * 85),
    label="Assumed for projection",
)
plt.legend()

locator = mdates.DayLocator([1, 15])
formatter = mdates.ConciseDateFormatter(locator)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.grid(True, color='k', linestyle=":", alpha=0.5)
plt.axis(
    xmin=np.datetime64('2021-05-01'), xmax=np.datetime64('2022-02-28'), ymax=200, ymin=0
)
plt.ylabel('cumulative doses per hundred population')
plt.title(f"{STATE} cumulative vaccinations per capita")

plt.tight_layout()

plt.savefig("COVID_qld_projected_doses.svg")
plt.savefig("COVID_qld_projected_doses.png", dpi=133)
plt.show()
