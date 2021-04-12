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
    return convolve(data, kernel, mode='same') / normalisation


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

    # Only use data as of yesterday:
    # state_doses = state_doses[:-1]
    # dates = dates[:-1]

    state_doses = state_doses[dates >= START_DATE]
    dates = dates[dates >= START_DATE]
    doses_by_state[s] = state_doses


doses_by_state['fed'] = doses_by_state['aus'] - sum(
    doses_by_state[s] for s in STATES if s != 'aus'
)

doses = doses_by_state['aus']


pfizer_supply_data = """
2021-02-21      142_000
2021-02-28      308_000
2021-03-07      443_000
2021-03-14      592_000
2021-03-28      751_000
2021-04-11      870_000
2021-04-18      1_000_000
2021-04-25      1_130_000
2021-05-02      1_300_000  
"""

AZ_OS_supply_data = """
2021-03-07      300_000
2021-03-21      700_000
"""

AZ_local_supply_data = """
2021-03-28        832_000
2021-04-11      1_300_000
2021-04-18      1_770_000
2021-04-25      2_250_000
2021-05-02      2_920_000
"""

PROJECT = True


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

if PROJECT:
    projection_dates = np.arange(dates[-1] + 1, np.datetime64('2021-05-05'))
    all_dates = np.concatenate((dates, projection_dates))
else:
    all_dates = dates

# Calculate vaccine utilisation:
first_doses = np.zeros(len(all_dates), dtype=float)
first_doses[: len(doses)] = doses
first_doses[len(doses) :] = doses[-1]
AZ_first_doses = np.zeros_like(first_doses)
pfizer_first_doses = np.zeros_like(first_doses)
AZ_second_doses = np.zeros_like(first_doses)
pfizer_second_doses = np.zeros_like(first_doses)
AZ_reserved = np.zeros_like(first_doses)
pfizer_reserved = np.zeros_like(first_doses)
AZ_available = np.zeros_like(first_doses)
pfizer_available = np.zeros_like(first_doses)

tau_AZ = 84
tau_pfizer = 21

pfizer_available += pfizer_shipments[pfizer_supply_dates < dates[0]].sum()
AZ_available += AZ_shipments[AZ_OS_supply_dates < dates[0]].sum()
AZ_available += AZ_production[AZ_local_supply_dates < dates[0]].sum()


for i, date in enumerate(all_dates):
    if date in pfizer_supply_dates:
        pfizer_available[i:] += 0.5 * pfizer_shipments[pfizer_supply_dates == date][0]
        pfizer_reserved[i:] += 0.5 * pfizer_shipments[pfizer_supply_dates == date][0]
    if date in AZ_OS_supply_dates:
        AZ_available[i:] += 0.5 * AZ_shipments[AZ_OS_supply_dates == date][0]
        AZ_reserved[i:] += 0.5 * AZ_shipments[AZ_OS_supply_dates == date][0]
    if date in AZ_local_supply_dates:
        AZ_available[i:] += 0.5 * AZ_production[AZ_local_supply_dates == date][0]
        AZ_reserved[i:] += 0.5 * AZ_production[AZ_local_supply_dates == date][0]
    if i == 0:
        first_doses_today = first_doses[i]
    elif i < len(dates):
        first_doses_today = first_doses[i] - first_doses[i - 1]
    else:
        # This is the assumption for projecting based on expected supply. That we use 5%
        # of available doses each day on first doses. Since a dose will be reserved as
        # well, this means we're always 10 days away from running out of vaccine at the
        # current rate - which is approximately what we see in the data.
        first_doses_today = 0.1 * (pfizer_available[i] + AZ_available[i])
        first_doses[i:] += first_doses_today

    AZ_frac = AZ_available[i] / (AZ_available[i] + pfizer_available[i])
    pfizer_frac = pfizer_available[i] / (AZ_available[i] + pfizer_available[i])

    AZ_first_doses_today = AZ_frac * first_doses_today
    pfizer_first_doses_today = pfizer_frac * first_doses_today

    AZ_first_doses[i:] += AZ_first_doses_today
    pfizer_first_doses[i:] += pfizer_first_doses_today

    AZ_available[i:] -= AZ_first_doses_today
    pfizer_available[i:] -= pfizer_first_doses_today

    AZ_reserved[i + tau_AZ:] -= AZ_first_doses_today
    pfizer_reserved[i + tau_pfizer:] -= pfizer_first_doses_today

    first_doses[i + tau_AZ :] -= AZ_first_doses_today
    first_doses[i + tau_pfizer :] -= pfizer_first_doses_today

    AZ_second_doses[i + tau_AZ :] += AZ_first_doses_today
    pfizer_second_doses[i + tau_pfizer :] += pfizer_first_doses_today


proj_doses = AZ_first_doses + AZ_second_doses + pfizer_first_doses + pfizer_second_doses

N_DAYS_TARGET = 250

days = (dates - dates[0]).astype(float)
days_model = np.linspace(days[0], days[-1] + N_DAYS_TARGET, 1000)

fig1 = plt.figure(figsize=(8, 6))

plt.fill_between(
    dates + 1,
    doses / 1e6,
    label='Cumulative doses',
    step='pre',
    color='C0',
)

if PROJECT:
    plt.fill_between(
        all_dates[len(dates) - 1 :] + 1,
        proj_doses[len(dates) - 1 :] / 1e6,
        label='Projected',
        step='pre',
        color='cyan',
        alpha=0.5,
        linewidth=0,
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

cumsum = np.zeros(len(dates))
colours = list(reversed([f'C{i}' for i in range(9)]))
for i, state in enumerate(['nt', 'act', 'tas', 'sa', 'wa', 'qld', 'vic', 'nsw', 'fed']):
    doses = doses_by_state[state]
    smoothed_doses = gaussian_smoothing(np.diff(doses, prepend=0), 2).cumsum()
    daily_doses = np.diff(smoothed_doses, prepend=0)
    latest_daily_doses = daily_doses[-1]

    plt.fill_between(
        dates + 1,
        cumsum / 1e3,
        (cumsum + daily_doses) / 1e3,
        label=f'{state.upper()} ({daily_doses[-1] / 1000:.1f}k/day)',
        step='pre',
        color=colours[i],
        linewidth=0,
    )
    cumsum += daily_doses


if PROJECT:
    daily_proj_doses = np.diff(proj_doses, prepend=0)
    plt.fill_between(
        all_dates[len(dates) - 1 :] + 1,
        gaussian_smoothing(daily_proj_doses / 1e3, 2)[len(dates) - 1 :],
        label='Projected (national)',
        step='pre',
        color='cyan',
        alpha=0.5,
        linewidth=0,
    )

latest_daily_doses = cumsum[-1]

plt.title(
    f'Smoothed daily doses by state/territory. Latest national rate: {latest_daily_doses / 1000:.1f}k/day'
)

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
cumsum = np.zeros(len(all_dates))
for arr, label, colour in [
    (AZ_first_doses + pfizer_first_doses, 'Administered first doses', 'C0'),
    (AZ_available + pfizer_available, 'Available for first doses', 'C2'),
    (AZ_second_doses + pfizer_second_doses, 'Administered second doses', 'C1'),
    (AZ_reserved + pfizer_reserved, 'Reserved for second doses', 'C3'),
]:
    plt.fill_between(
        all_dates[: len(dates)] + 1,
        cumsum[: len(dates)] / 1e3,
        (cumsum + arr)[: len(dates)] / 1e3,
        label=f'{label} ({arr[len(dates)-1] / 1000:.0f}k)',
        step='pre',
        color=colour,
        linewidth=0,
    )
    cumsum += arr

used = AZ_first_doses[len(dates) - 1] + pfizer_first_doses[len(dates) - 1]
unused = AZ_available[len(dates) - 1] + pfizer_available[len(dates) - 1]
utilisation = 100 * used / (used + unused)
plt.ylabel('Cumulative doses (thousands)')
plt.title(f"Estimated vaccine utilisation: first dose utilisation rate: {utilisation:.1f}%")
plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=dates[0].astype(int) + 125,
    ymin=0,
    ymax=5000,
)
ax3 = plt.gca()


fig4 = plt.figure(figsize=(8, 6))
cumsum = np.zeros(len(all_dates))
for arr, label, colour in [
    (AZ_first_doses, 'AZ administered first doses', 'C0'),
    (AZ_available, 'AZ available for first doses', 'C2'),
    (AZ_second_doses, 'AZ administered second doses', 'C1'),
    (AZ_reserved, 'AZ reserved for second doses', 'C3'),
]:
    plt.fill_between(
        all_dates[: len(dates)] + 1,
        cumsum[: len(dates)] / 1e3,
        (cumsum + arr)[: len(dates)] / 1e3,
        label=f'{label} ({arr[len(dates)-1] / 1000:.0f}k)',
        step='pre',
        color=colour,
        linewidth=0,
    )
    cumsum += arr

used = AZ_first_doses[len(dates) - 1]
unused = AZ_available[len(dates) - 1]
utilisation = 100 * used / (used + unused)
plt.ylabel('Cumulative doses (thousands)')
plt.title(
    f"Estimated AZ vaccine utilisation: first dose utilisation rate: {utilisation:.1f}%"
)
plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=dates[0].astype(int) + 125,
    ymin=0,
    ymax=5000,
)
ax4 = plt.gca()


fig5 = plt.figure(figsize=(8, 6))
cumsum = np.zeros(len(all_dates))
for arr, label, colour in [
    (pfizer_first_doses, 'Pfizer administered first doses', 'C0'),
    (pfizer_available, 'Pfizer available for first doses', 'C2'),
    (pfizer_second_doses, 'Pfizer administered second doses', 'C1'),
    (pfizer_reserved, 'Pfizer reserved for second doses', 'C3'),
]:
    plt.fill_between(
        all_dates[: len(dates)] + 1,
        cumsum[: len(dates)] / 1e3,
        (cumsum + arr)[: len(dates)] / 1e3,
        label=f'{label} ({arr[len(dates)-1] / 1000:.0f}k)',
        step='pre',
        color=colour,
        linewidth=0,
    )
    cumsum += arr

used = pfizer_first_doses[len(dates) - 1]
unused = pfizer_available[len(dates) - 1]
utilisation = 100 * used / (used + unused)
plt.ylabel('Cumulative doses (thousands)')
plt.title(
    f"Estimated Pfizer vaccine utilisation: first dose utilisation rate: {utilisation:.1f}%"
)
plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=dates[0].astype(int) + 125,
    ymin=0,
    ymax=5000,
)
ax5 = plt.gca()


for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.fill_betweenx(
        [0, ax.get_ylim()[1]],
        2 * [START_DATE.astype(int)],
        2 * [PHASE_1B.astype(int)],
        color='red',
        alpha=0.5,
        linewidth=0,
        label='Phase 1a',
        zorder=-10,
    )

    ax.fill_betweenx(
        [0, ax.get_ylim()[1]],
        2 * [PHASE_1B.astype(int)],
        2 * [dates[-1].astype(int) + 30],
        color='orange',
        alpha=0.5,
        linewidth=0,
        label='Phase 1b',
        zorder=-10,
    )

    for i in range(10):
        ax.fill_betweenx(
            [0, ax.get_ylim()[1]],
            2 * [dates[-1].astype(int) + 30 + i],
            2 * [dates[-1].astype(int) + 31 + i],
            color='orange',
            alpha=0.5 * (10 - i) / 10,
            linewidth=0,
            zorder=-10,
        )


handles, labels = ax1.get_legend_handles_labels()
if PROJECT:
    order = [1, 2, 0, 3, 4]
else:
    order = [1, 0, 2, 3]
ax1.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='lower right',
    # ncol=2,
)

handles, labels = ax2.get_legend_handles_labels()
if PROJECT:
    order = [9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 0, 11, 12]
else:
    order = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 10, 11]
ax2.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='lower right',
    # ncol=2,
)

for ax in [ax3, ax4, ax5]:
    handles, labels = ax.get_legend_handles_labels()
    order = [3, 2, 1, 0, 4, 5]
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc='lower right',
        # ncol=2,
    )

for ax in [ax1, ax2, ax3, ax4, ax5]:
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

for extension in ['png', 'svg']:
    fig1.savefig(f'cumulative_doses.{extension}')
    fig2.savefig(f'daily_doses_by_state.{extension}')
    fig3.savefig(f'utilisation.{extension}')
    fig4.savefig(f'az_utilisation.{extension}')
    fig5.savefig(f'pfizer_utilisation.{extension}')

plt.show()
