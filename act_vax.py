import json
import requests
import numpy as np
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
    """Return array of dates and total doses administered to residents of the given
    state/territory"""
    url = "https://vaccinedata.covid19nearme.com.au/data/air_residence.json"
    data = json.loads(requests.get(url).content)
    # Convert dates to np.datetime64
    for row in data:
        row['DATE_AS_AT'] = np.datetime64(row['DATE_AS_AT'])
    data.sort(key=lambda row: row['DATE_AS_AT'])

    dates = np.array(sorted(set([row['DATE_AS_AT'] for row in data])))

    total_doses = {d: 0 for d in dates}

    for row in data:
        if row['STATE'] != state:
            continue
        date = row['DATE_AS_AT']
        age_range = (row['AGE_LOWER'], row['AGE_UPPER'] )
        if age_range == (16, 999): 
            FIRST_KEY = 'AIR_RESIDENCE_FIRST_DOSE_COUNT'
            SECOND_KEY = 'AIR_RESIDENCE_SECOND_DOSE_COUNT'
        elif age_range == (12, 15):
            FIRST_KEY = 'AIR_RESIDENCE_FIRST_DOSE_APPROX_COUNT'
            SECOND_KEY = 'AIR_RESIDENCE_SECOND_DOSE_APPROX_COUNT'
        else:
            continue

        total_doses[date] += row[FIRST_KEY] + row[SECOND_KEY] 
    return dates, np.diff(np.array(list(total_doses.values())), prepend=0)

POPS = {
    'AUS': 25.36e6,
    'VIC': 6.681e6,
    'QLD': 5.185e6,
    'SA': 1.771e6,
    'WA': 2.667e6,
    'TAS': 541071,
    'NT': 246500,
    'ACT': 431215
}

STATE = 'ACT'

act_dates, act_doses = get_data(STATE)


days_projection = 200
t_projection = np.arange(act_dates[-1], act_dates[-1] + days_projection + 1)

SEP = np.datetime64('2021-09-01')
OCT = np.datetime64('2021-10-01')

act_proj_rate = np.zeros(len(t_projection))

act_proj_rate[:] =  0.1
# clip to 85% fully vaxed
initial_coverage =  100 * act_doses.sum() / POPS[STATE]
act_proj_rate[initial_coverage + act_proj_rate.cumsum() > 2 * 84] = 0

plt.figure(figsize=(10, 5))
plt.subplot(121)
act_rate = 100 * seven_day_average(act_doses) / POPS[STATE]
plt.step(act_dates[7:], act_rate[7:], label="Actual")
plt.step(t_projection, act_proj_rate, label="Assumed for projection")
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
act_cumulative = 100 * act_doses.cumsum() / POPS[STATE]
plt.step(act_dates, act_cumulative, label="Actual")
plt.step(
    t_projection,
    (act_proj_rate.cumsum() + act_cumulative[-1]).clip(0, 2 * 85),
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

plt.savefig("COVID_ACT_projected_doses.svg")
plt.savefig("COVID_ACT_projected_doses.png", dpi=133)
plt.show()
