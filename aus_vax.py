import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
from pathlib import Path
from pytz import timezone

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime] = converter


def n_day_average(data, n=14):
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


STATES = ['aus', 'nsw', 'vic', 'sa', 'wa', 'tas', 'qld', 'nt', 'act']


def gaussian_smoothing(data, pts):
    """gaussian smooth an array by given number of points"""
    from scipy.signal import convolve

    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return convolve(data , kernel, mode='same') / normalisation


START_DATE = np.datetime64('2021-02-22')
PHASE_1B = np.datetime64('2021-03-22')

doses_by_state = {}
for s in STATES:
    print(f"getting data for {s}")
    df = pd.read_html(f"https://covidlive.com.au/report/daily-vaccinations/{s}")[1]
    dates = np.array(df['DATE'][::-1])
    state_doses = np.array(df['DOSES'][::-1])
    dates = np.array(
        [np.datetime64(datetime.strptime(d, '%d %b %y'), 'D') for d in dates]
    )
    state_doses = state_doses[dates >= START_DATE]
    dates = dates[dates >= START_DATE]
    doses_by_state[s] = state_doses


doses_by_state['fed'] = doses_by_state['aus'] - sum(
    doses_by_state[s] for s in STATES if s != 'aus'
)

doses = doses_by_state['aus']


pfizer_supply_data = """
2021-02-21      142000
2021-02-28      308000
2021-03-07      443000
2021-03-14      592000
2021-03-28      751000
2021-04-11      870000      
""" 

AZ_OS_supply_data = """
2021-03-07      300000
2021-03-21      700000
"""

AZ_local_supply_data = """
2021-03-28      832000
2021-04-11      1300000
"""

def unpack_data(s):
    dates = []
    values = []
    for line in s.splitlines():
        if line.strip():
            date, value = line.split()
            dates.append(np.datetime64(date))
            values.append(float(value))
    return np.array(dates) - 4, np.array(values)

pfizer_supply_dates, pfizer_supply = unpack_data(pfizer_supply_data)
AZ_OS_supply_dates, AZ_OS_suppy = unpack_data(AZ_OS_supply_data)
AZ_local_supply_dates, AZ_local_supply = unpack_data(AZ_local_supply_data)

pfizer_shipments = np.diff(pfizer_supply, prepend=0)
AZ_shipments = np.diff(AZ_OS_suppy, prepend=0)
AZ_production = np.diff(AZ_local_supply, prepend=0)

# Calculate vaccine utilisation:
first_doses = doses.astype(float)
second_doses = np.zeros_like(first_doses)
reserved = np.zeros_like(first_doses)
pfizer_available = np.zeros_like(reserved)
AZ_available = np.zeros_like(reserved)

tau_AZ = 84
tau_pfizer = 21

pfizer_available += pfizer_shipments[pfizer_supply_dates < dates[0]].sum()
AZ_available += AZ_shipments[AZ_OS_supply_dates < dates[0]].sum()
AZ_available += AZ_production[AZ_local_supply_dates < dates[0]].sum()

for i, date in enumerate(dates):
    if date in pfizer_supply_dates:
        pfizer_available[i:] += pfizer_shipments[pfizer_supply_dates == date][0]
    if date in AZ_OS_supply_dates:
        AZ_available[i:] += AZ_shipments[AZ_OS_supply_dates == date][0]
    if date in AZ_local_supply_dates:
        AZ_available[i:] += AZ_production[AZ_local_supply_dates == date][0]
    if i:
        first_doses_today = first_doses[i] - first_doses[i-1]
    else:
        first_doses_today = first_doses[i]

    AZ_frac = AZ_available[i] / (AZ_available[i] + pfizer_available[i])
    pfizer_frac = pfizer_available[i] / (AZ_available[i] + pfizer_available[i])

    AZ_first_doses_today = AZ_frac * first_doses_today
    pfizer_first_doses_today = pfizer_frac * first_doses_today

    AZ_available[i:] -= 2 * AZ_first_doses_today
    pfizer_available[i:] -= 2 * pfizer_first_doses_today

    reserved[i : i + tau_AZ] += AZ_first_doses_today
    reserved[i : i + tau_pfizer] += pfizer_first_doses_today

    first_doses[i + tau_AZ:] -= AZ_first_doses_today
    first_doses[i + tau_pfizer:] -= pfizer_first_doses_today

    second_doses[i + tau_AZ:] += AZ_first_doses_today
    second_doses[i + tau_pfizer:] += pfizer_first_doses_today


# plt.plot(dates, AZ_available + pfizer_available)
# plt.show()

N_DAYS_PROJECT = 250

days = (dates - dates[0]).astype(float)
days_model = np.linspace(days[0], days[-1] + N_DAYS_PROJECT, 1000)

fig1 = plt.figure(figsize=(8, 6))

plt.fill_between(
    dates + 1, doses / 1e6, label='Cumulative doses', step='pre', color='C0', zorder=10
)

ax1 = plt.gca()
target = 160000 * days_model
plt.plot(
    days_model + dates[0].astype(int),
    target / 1e6,
    'k--',
    label='Target',
)


plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=dates[0].astype(int) + 250,
    ymin=0,
    ymax=40,
)

plt.title(f'AUS cumulative doses. Total to date: {doses[-1]/1e3:.1f}k')
plt.ylabel('Cumulative doses (millions)')


fig2 = plt.figure(figsize=(8, 6))

MOST_RECENT_FED_UPDATE = np.datetime64('2021-04-08')
FED_CLIP = len(dates) - 1 - np.argwhere(dates == MOST_RECENT_FED_UPDATE)[0, 0]

cumsum = np.zeros(len(dates))
colours = list(reversed([f'C{i}' for i in range(9)]))
for i, state in enumerate(['nt', 'act', 'tas', 'sa', 'wa', 'qld', 'vic', 'nsw', 'fed']):
    doses = doses_by_state[state]
    if state == 'fed' and FED_CLIP:
        smoothed_doses = gaussian_smoothing(
            np.diff(doses[:-FED_CLIP], prepend=0), 2
        ).cumsum()
        smoothed_doses = np.append(smoothed_doses, [smoothed_doses[-1]] * FED_CLIP)
    else:
        smoothed_doses = gaussian_smoothing(np.diff(doses, prepend=0), 2).cumsum()

    daily_doses = np.diff(smoothed_doses, prepend=0)
    if state == 'fed' and FED_CLIP:
        latest_daily_doses = daily_doses[-FED_CLIP - 1]
    else:
        latest_daily_doses = daily_doses[-1]

    plt.fill_between(
        dates + 1,
        cumsum / 1e3,
        (cumsum + daily_doses) / 1e3,
        label=f'{state.upper()} ({latest_daily_doses / 1000:.1f}k/day)',
        step='pre',
        color=colours[i],
        zorder=10,
        linewidth=0,
    )
    if state == 'fed' and FED_CLIP:
        plt.fill_between(
            dates[-FED_CLIP - 1 :] + 1,
            cumsum[-FED_CLIP - 1 :] / 1e3,
            (cumsum[-FED_CLIP - 1 :] + latest_daily_doses) / 1e3,
            label=f'{state.upper()} (projected)',
            step='pre',
            color=colours[i],
            hatch="//////",
            edgecolor='tab:cyan',
            zorder=10,
            linewidth=0,
        )
    cumsum += daily_doses

latest_daily_doses = cumsum[-1]
if FED_CLIP:
    latest_daily_doses += daily_doses[-FED_CLIP - 1]

asterisk = '*' if FED_CLIP else ''
plt.title(
    f'Smoothed daily doses by state/territory. Latest national rate{asterisk}: {latest_daily_doses / 1000:.1f}k/day'
)
if FED_CLIP:
    text = plt.figtext(
        0.575,
        0.85,
        "* Includes projected federally-administered doses",
        fontsize='x-small',
    )
    text.set_bbox(dict(facecolor='white', alpha=0.8, linewidth=0))

plt.ylabel('Daily doses (thousands)')
plt.axhline(160, color='k', linestyle='--', label="Target")

plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=dates[0].astype(int) + 250,
    ymin=0,
    ymax=200,
)
ax2 = plt.gca()


fig3 = plt.figure(figsize=(8, 6))
cumsum = np.zeros(len(dates))
for arr, label, colour in [
    (first_doses, 'First doses', 'C0'),
    (second_doses, 'Second doses', 'C1'),
    (reserved, 'Reserved', 'C3'),
    (pfizer_available + AZ_available, 'Available', 'C2'),
]:
    plt.fill_between(
        dates + 1,
        cumsum / 1e3,
        (cumsum + arr) / 1e3,
        label=f'{label} ({arr[-1] / 1000:.0f}k doses)',
        step='pre',
        color=colour,
        zorder=10,
        linewidth=0,
    )
    cumsum += arr

used = first_doses[-1] + second_doses[-1] + reserved[-1]
unused = AZ_available[-1] + pfizer_available[-1]
utilisation = 100 * used / (used + unused)
plt.ylabel('Cumulative doses (thousands)')
plt.title(f"Estimated vaccine utilisation: latest utilisation rate: {utilisation:.1f}%")
plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=dates[0].astype(int) + 125,
    ymin=0,
    ymax=5000,
)
ax3 = plt.gca()



for ax in [ax1, ax2, ax3]:
    ax.fill_betweenx(
        [0, ax.get_ylim()[1]],
        2 * [START_DATE.astype(int)],
        2 * [PHASE_1B.astype(int)],
        color='red',
        alpha=0.5,
        linewidth=0,
        label='Phase 1a',
    )

    ax.fill_betweenx(
        [0, ax.get_ylim()[1]],
        2 * [PHASE_1B.astype(int)],
        2 * [dates[-1].astype(int) + 30],
        color='orange',
        alpha=0.5,
        linewidth=0,
        label='Phase 1b',
    )

    for i in range(10):
        ax.fill_betweenx(
            [0, ax.get_ylim()[1]],
            2 * [dates[-1].astype(int) + 30 + i],
            2 * [dates[-1].astype(int) + 31 + i],
            color='orange',
            alpha=0.5 * (10 - i) / 10,
            linewidth=0,
        )


handles, labels = ax1.get_legend_handles_labels()
order = [1, 0, 2, 3]
ax1.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='lower right',
    # ncol=2,
)

handles, labels = ax2.get_legend_handles_labels()
order = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 10, 11]
ax2.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='lower right',
    # ncol=2,
)

handles, labels = ax3.get_legend_handles_labels()
order = [3, 2, 1, 0, 4, 5]
ax3.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='lower right',
    # ncol=2,
)

for ax in [ax1, ax2, ax3]:
    locator = mdates.DayLocator([1])
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.get_xaxis().get_major_formatter().show_offset = False
    ax.grid(True, linestyle=":", color='k')



# Update the date in the HTML
html_file = 'aus_vaccinations.html'
html_lines = Path(html_file).read_text().splitlines()
now = datetime.now(timezone('Australia/Melbourne')).strftime('%Y-%m-%d-%H:%M')
for i, line in enumerate(html_lines):
    if 'Last updated' in line:
        html_lines[i] = f'    Last updated: {now} Melbourne time'
Path(html_file).write_text('\n'.join(html_lines) + '\n')

fig1.savefig('cumulative_doses.svg')
fig2.savefig('daily_doses_by_state.svg')
fig3.savefig('utilisation.svg')

fig1.savefig('cumulative_doses.png')
fig2.savefig('daily_doses_by_state.png')
fig3.savefig('utilisation.png')

plt.show()
