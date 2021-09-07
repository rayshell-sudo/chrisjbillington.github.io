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


def get_data():
    REPO_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master"
    DATA_DIR = "public/data/vaccinations"
    df = pd.read_csv(f"{REPO_URL}/{DATA_DIR}/vaccinations.csv")
    df = df[df['location']=="New Zealand"]
    dates = np.array([np.datetime64(d) for d in df['date']])
    daily_doses_per_100 = np.diff(df['total_vaccinations_per_hundred'], prepend=0)
    # Remove NaNs from the dataset, duplicate prev. day instead
    for i, val in enumerate(daily_doses_per_100):
        if np.isnan(val):
            daily_doses_per_100[i] = daily_doses_per_100[i-1]
    return dates, daily_doses_per_100

nz_dates, nz_doses_per_100 = get_data()

days_projection = 200
t_projection = np.arange(nz_dates[-1], nz_dates[-1] + days_projection + 1)

SEP = np.datetime64('2021-09-01')
OCT = np.datetime64('2021-10-01')

nz_proj_rate = np.zeros(len(t_projection))

nz_proj_rate[:] =  1.8 # Oct onward
nz_proj_rate[t_projection < OCT] =  1.6 # Sep
nz_proj_rate[t_projection < SEP] =  1.0 # Aug
# clip to 85% fully vaxed
initial_coverage =  nz_doses_per_100.sum()
nz_proj_rate[initial_coverage + nz_proj_rate.cumsum() > 2 * 85] = 0

plt.figure(figsize=(10, 5))
plt.subplot(121)
nsw_rate = seven_day_average(nz_doses_per_100)
plt.step(nz_dates, nsw_rate, label="Actual")
plt.step(t_projection, nz_proj_rate, label="Assumed for projection")
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
plt.title(f"NZ daily vaccinations per capita")

plt.subplot(122)
nsw_cumulative = nz_doses_per_100.cumsum()
plt.step(nz_dates, nsw_cumulative, label="Actual")
plt.step(
    t_projection,
    (nz_proj_rate.cumsum() + nsw_cumulative[-1]).clip(0, 2 * 85),
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
plt.title(f"NZ cumulative vaccinations per capita")

plt.tight_layout()

plt.savefig("COVID_NZ_projected_doses.svg")
plt.savefig("COVID_NZ_projected_doses.png", dpi=133)
plt.show()
