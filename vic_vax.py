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


def get_data(state):
    url = f"https://covidlive.com.au/report/daily-vaccinations/{state.lower()}"

    df = pd.read_html(url)[1]

    doses = df['DOSES'][::-1]
    doses = np.diff(doses, prepend=0)
    dates = np.array(
        [np.datetime64(datetime.strptime(d, '%d %b %y'), 'D') for d in df['DATE'][::-1]]
    )
    dates = dates[:-1]
    doses = doses[:-1]
    return dates, doses

POPS = {
    'AUS': 25.36e6,
    'vic': 8.166e6,
    'VIC': 6.681e6,
    'QLD': 5.185e6,
    'SA': 1.771e6,
    'WA': 2.667e6,
    'TAS': 541071,
    'NT': 246500,
}

STATE = 'VIC'

vic_dates, vic_doses = get_data(STATE)

# Smooth out the data correction made on Aug 16th:
CORRECTION_DATE = np.datetime64('2021-08-16')
CORRECTION_DOSES = -75000
vic_doses = vic_doses.astype(float)
vic_doses[vic_dates == CORRECTION_DATE] -= CORRECTION_DOSES
sum_prior = vic_doses[vic_dates < CORRECTION_DATE].sum()
SCALE_FACTOR = 1 + CORRECTION_DOSES / sum_prior
vic_doses[vic_dates < CORRECTION_DATE] *= SCALE_FACTOR


days_projection = 200
t_projection = np.arange(vic_dates[-1], vic_dates[-1] + days_projection + 1)

SEP = np.datetime64('2021-09-01')
OCT = np.datetime64('2021-10-01')

vic_proj_rate = np.zeros(len(t_projection))

vic_proj_rate[:] =  1.4 # Oct onward
vic_proj_rate[t_projection < OCT] =  1.2 # Sep
vic_proj_rate[t_projection < SEP] =  1.0 # Aug
# clip to 85% fully vaxed
initial_coverage =  100 * vic_doses.sum() / POPS[STATE]
vic_proj_rate[initial_coverage + vic_proj_rate.cumsum() > 2 * 85] = 0

plt.figure(figsize=(10, 5))
plt.subplot(121)
vic_rate = 100 * seven_day_average(vic_doses) / POPS[STATE]
plt.step(vic_dates, vic_rate, label="Actual")
plt.step(t_projection, vic_proj_rate, label="Assumed for projection")
plt.legend()
locator = mdates.DayLocator([1, 15])
formatter = mdates.ConciseDateFormatter(locator)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.grid(True, color='k', linestyle=":", alpha=0.5)
plt.axis(
    xmin=np.datetime64('2021-05-01'), xmax=np.datetime64('2021-12-31'), ymax=2.2, ymin=0
)
plt.ylabel('7d avg doses per hundred population per day')
plt.title(f"{STATE} daily vaccinations per capita")

plt.subplot(122)
vic_cumulative = 100 * vic_doses.cumsum() / POPS[STATE]
plt.step(vic_dates, vic_cumulative, label="Actual")
plt.step(
    t_projection,
    (vic_proj_rate.cumsum() + vic_cumulative[-1]).clip(0, 2 * 85),
    label="Assumed for projection",
)
plt.legend()

locator = mdates.DayLocator([1, 15])
formatter = mdates.ConciseDateFormatter(locator)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.grid(True, color='k', linestyle=":", alpha=0.5)
plt.axis(
    xmin=np.datetime64('2021-05-01'), xmax=np.datetime64('2021-12-31'), ymax=200, ymin=0
)
plt.ylabel('cumulative doses per hundred population')
plt.title(f"{STATE} cumulative vaccinations per capita")

plt.tight_layout()

plt.savefig("COVID_VIC_projected_doses.svg")
plt.savefig("COVID_VIC_projected_doses.png", dpi=133)
plt.show()
