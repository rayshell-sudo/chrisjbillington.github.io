import urllib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

converter = mdates.ConciseDateConverter()

munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime] = converter

POP_OF_NZ = 4.917e6

def seven_day_average(data):
    n = 7
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


# def linear_interpolate_nans(x):
#     """Replace NaNs with a linear interpolation of nearest non NaNs"""
#     for i, value in enumerate(x):
#         if np.isnan(value):
#             j = i + 1
#             while True:
#                 if j == len(x):
#                     x[i] = x[i - 1]
#                     break
#                 if not np.isnan(x[j]):
#                     x[i] = x[i - 1] + (x[j] - x[i - 1]) / (j - i)
#                     break
#                 j += 1


# def get_data():
#     REPO_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master"
#     DATA_DIR = "public/data/vaccinations"
#     df = pd.read_csv(f"{REPO_URL}/{DATA_DIR}/vaccinations.csv")
#     df = df[df['location'] == "New Zealand"]
#     dates = np.array([np.datetime64(d) for d in df['date']])
#     doses_per_100 = np.array(df['total_vaccinations_per_hundred'])
#     linear_interpolate_nans(doses_per_100)
#     daily_doses_per_100 = np.diff(doses_per_100, prepend=0)
#     return dates, daily_doses_per_100


def get_data():
    # HTTP headers to emulate curl
    curl_headers = {'user-agent': 'curl/7.64.1'}

    for i in range(1, 8):
        datestring = (datetime.now() - timedelta(days=i)).strftime("%d_%m_%Y")
        url = (
            "https://www.health.govt.nz/system/files/documents/pages/"
            f"covid_vaccinations_{datestring}.xlsx"
        )
        print(url)
        try:
            df = pd.read_excel(url, sheet_name="Date", storage_options=curl_headers)
            break
        except urllib.error.HTTPError:
            continue

    dates = np.array(df['Date'], dtype='datetime64[D]')
    doses = np.array(df['First dose administered'] + df['Second dose administered'])
    return dates, 100 * doses / POP_OF_NZ


nz_dates, nz_doses_per_100 = get_data()

days_projection = 200
t_projection = np.arange(nz_dates[-1], nz_dates[-1] + days_projection + 1)

SEP = np.datetime64('2021-09-01')
OCT = np.datetime64('2021-10-01')

nz_proj_rate = np.zeros(len(t_projection))

nz_proj_rate[:] = 0.75  # Oct onward
# clip to 85% fully vaxed
initial_coverage = nz_doses_per_100.sum()
nz_proj_rate[initial_coverage + nz_proj_rate.cumsum() > 2 * 85] = 0

plt.figure(figsize=(10, 5))
plt.subplot(121)
nz_rate = seven_day_average(nz_doses_per_100)
plt.step(nz_dates, nz_rate, label="Actual")
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
