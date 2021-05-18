import sys
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
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

def padded_gaussian_smoothing(data, pts, pad_avg=7):
    """gaussian smooth an array by given number of points, with padding at the edges
    equal to the pad_avg-point average at each edge"""
    from scipy.signal import convolve

    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    kernel /= kernel.sum()
    padded_data = np.concatenate(
        [
            np.full(4 * pts, data[:pad_avg].mean()),
            data,
            np.full(4 * pts, data[-pad_avg:].mean()),
        ]
    )
    return convolve(padded_data, kernel, mode='same')[4 * pts : -4 * pts]


def exponential_smoothing(data, pts):
    """exponentially smooth an array by given number of points"""
    from scipy.signal import convolve

    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-x / pts)
    kernel[x < 0] = 0
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return convolve(data, kernel, mode='same') / normalisation


START_DATE = np.datetime64('2021-02-22')
PHASE_1B = np.datetime64('2021-03-22')
PHASE_2A = np.datetime64('2021-05-03')

doses_by_state = {}
for s in STATES:
    print(f"getting data for {s}")
    df = pd.read_html(f"https://covidlive.com.au/report/daily-vaccinations/{s}")[1]
    dates = np.array(df['DATE'][::-1])
    state_doses = np.array(df['DOSES'][::-1], dtype=float)
    dates = np.array(
        [np.datetime64(datetime.strptime(d, '%d %b %y'), 'D') for d in dates]
    )

    # Only use data as of yesterday:
    # state_doses = state_doses[:-1]
    # dates = dates[:-1]

    # Extrapolate one day
    # for _ in range(2):
    #     dates = np.append(dates, [dates[-1] + 1])
    #     state_doses = np.append(state_doses, [2 * state_doses[-1] - state_doses[-2]])

    state_doses = state_doses[dates >= START_DATE]
    dates = dates[dates >= START_DATE]
    doses_by_state[s] = state_doses

# Data not yet on covidlive
# doses_by_state['aus'][-1] = 3_100_137
# doses_by_state['nsw'][-1] = 280_135
# doses_by_state['vic'][-1] = 313_539
# doses_by_state['qld'][-1] = 170_330
# doses_by_state['wa'][-1] = 130_649
# doses_by_state['tas'][-1] = 49_739
# doses_by_state['sa'][-1] = 80_017
# doses_by_state['act'][-1] = 38_696
# doses_by_state['nt'][-1] = 22_953


doses_by_state['fed'] = doses_by_state['aus'] - sum(
    doses_by_state[s] for s in STATES if s != 'aus'
)


# 80560 doses were reported on April 19th that actually were administered "prior to
# April 17". We don't know how much prior, so we'll spread these doses out proportional
# to each day's doses from the start of phase 1b until April 16.
LATE_REPORTED_GP_DOSES = 80560
daily_fed_doses = np.diff(doses_by_state['fed'], prepend=0)
reportedix = np.where(dates == np.datetime64('2021-04-19'))[0][0]
backdates = (PHASE_1B <= dates) & (dates <= np.datetime64('2021-04-17'))
daily_fed_doses[reportedix] -= LATE_REPORTED_GP_DOSES
total_in_backdate_period = daily_fed_doses[backdates].sum()
daily_fed_doses[backdates] *= 1 + LATE_REPORTED_GP_DOSES / total_in_backdate_period
doses_by_state['fed'] = daily_fed_doses.cumsum()



doses = doses_by_state['aus']


pfizer_supply_data = """
2021-02-21        142_000
2021-02-28        308_000
2021-03-07        443_000
2021-03-14        592_000
2021-03-21        592_000
2021-03-28        751_000
2021-04-04        751_000
2021-04-11        870_000
2021-04-18      1_172_000
2021-04-25      1_345_000
2021-05-02      1_518_000
2021-05-09      1_869_000
2021-05-16      2_220_000
2021-05-23      2_572_000
"""

LONGPROJECT = False or 'project' in sys.argv

AZ_OS_supply_data = """
2021-03-07      300_000
2021-03-21      714_000
"""

AZ_local_supply_data = """
2021-03-28        832_000
# 2021-04-04        832_000
2021-04-11      1_300_000
2021-04-18      1_770_000
2021-04-25      2_238_000
2021-05-02      2_920_000
2021-05-09      3_681_900
2021-05-16      4_712_500
2021-05-23      5_712_500
"""

# Doses distributed by the feds (scroll to weekly updates):
# https://www.health.gov.au/resources/collections/covid-19-vaccine-rollout-updates
#distributed_doses_data = """
#2021-04-04  1_905_294
#2021-04-11  2_447_865
#2021-04-18  3_025_852 
#2021-04-25  3_601_029
#2021-05-02  4_086_946
#2021-05-09  4_620_108
#"""

PLOT_END_DATE = (
    np.datetime64('2021-12-31') if LONGPROJECT else dates[-1] + 50 #np.datetime64('2021-05-31')
)
CUMULATIVE_YMAX = 7  # million

PROJECT = True


def unpack_data(s):
    dates = []
    values = []
    for line in s.splitlines():
        if line.strip() and not line.strip().startswith('#'):
            date, value = line.split(maxsplit=1)
            dates.append(np.datetime64(date))
            values.append(eval(value))
    return np.array(dates), np.array(values)


pfizer_supply_dates, pfizer_supply = unpack_data(pfizer_supply_data)
AZ_OS_supply_dates, AZ_OS_suppy = unpack_data(AZ_OS_supply_data)
AZ_local_supply_dates, AZ_local_supply = unpack_data(AZ_local_supply_data)
#distributed_doses_dates, distributed_doses = unpack_data(distributed_doses_data)

WASTAGE = 0.1

# Estimated AZ supply. Assume 670k per week locally-produced AZ up to 16M (plus
# wastage):
AZ_MAX_DOSES = 16e6 / (1 - WASTAGE)
n_weeks = int((AZ_MAX_DOSES - AZ_local_supply[-1]) // 1000000) + 1
AZ_local_supply_dates = np.append(
    AZ_local_supply_dates,
    [AZ_local_supply_dates[-1] + 7 * (i + 1) for i in range(n_weeks)],
)
AZ_local_supply = np.append(
    AZ_local_supply, [AZ_local_supply[-1] + 1000000 * (i + 1) for i in range(n_weeks)]
)
AZ_local_supply[-1] = AZ_MAX_DOSES


# Estimated Pfizer supply. 300k per week until July. Then 600k per week until Oct, then
# whatever weekly rate is required to get to 40M by EOY.
MID_MAY = np.datetime64('2021-05-15')
JULY = np.datetime64('2021-07-01')
OCTOBER = np.datetime64('2021-10-01')
while pfizer_supply_dates[-1] <= JULY:
    pfizer_supply_dates = np.append(pfizer_supply_dates, [pfizer_supply_dates[-1] + 7])
    pfizer_supply = np.append(pfizer_supply, [pfizer_supply[-1] + 300000])
while pfizer_supply_dates[-1] <= OCTOBER:
    pfizer_supply_dates = np.append(pfizer_supply_dates, [pfizer_supply_dates[-1] + 7])
    pfizer_supply = np.append(pfizer_supply, [pfizer_supply[-1] + 600000])
remaining_per_week = (40e6 - pfizer_supply[-1]) / 12
for i in range(12):
    pfizer_supply_dates = np.append(pfizer_supply_dates, [pfizer_supply_dates[-1] + 7])
    pfizer_supply = np.append(pfizer_supply, [pfizer_supply[-1] + remaining_per_week])

pfizer_shipments = np.diff(pfizer_supply, prepend=0)
AZ_shipments = np.diff(AZ_OS_suppy, prepend=0)
AZ_production = np.diff(AZ_local_supply, prepend=0)

if PROJECT:
    if LONGPROJECT:
        projection_end = np.datetime64('2021-12-31')
    else:
        projection_end = np.datetime64('2021-12-31')
    projection_dates = np.arange(dates[-1] + 1, projection_end)
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
wasted = np.zeros_like(first_doses)

tau_AZ = 84
tau_pfizer = 21

pfizer_available += pfizer_shipments[pfizer_supply_dates < dates[0]].sum()
AZ_available += AZ_shipments[AZ_OS_supply_dates < dates[0]].sum()
AZ_available += AZ_production[AZ_local_supply_dates < dates[0]].sum()


for i, date in enumerate(all_dates):
    if date in pfizer_supply_dates:
        pfizer_lot = pfizer_shipments[pfizer_supply_dates == date][0]
        pfizer_available[i:] +=  0.5 * (1 - WASTAGE) * pfizer_lot
        pfizer_reserved[i:] += 0.5 * (1 - WASTAGE) * pfizer_lot
        wasted[i:] += WASTAGE * pfizer_lot
    if date in AZ_OS_supply_dates:
        AZ_lot = AZ_shipments[AZ_OS_supply_dates == date][0]
        AZ_available[i:] += 0.5 * (1 - WASTAGE) * AZ_lot
        AZ_reserved[i:] += 0.5 * (1 - WASTAGE) * AZ_lot
        wasted[i:] += WASTAGE * AZ_lot
    if date in AZ_local_supply_dates:
        if date < np.datetime64('2021-04-11'):
            AZ_lot = AZ_production[AZ_local_supply_dates == date][0]
            AZ_available[i:] += 0.5 * (1 - WASTAGE) * AZ_lot
            AZ_reserved[i:] += 0.5 * (1 - WASTAGE) * AZ_lot
        else:
            outstanding_AZ_second_doses = AZ_first_doses[i] - AZ_second_doses[i]
            reserve_allocation = 1.0 * outstanding_AZ_second_doses - AZ_reserved[i]
            AZ_lot = AZ_production[AZ_local_supply_dates == date][0]
            AZ_available[i:] += (1 - WASTAGE) * AZ_lot - reserve_allocation
            AZ_reserved[i:] += reserve_allocation
            wasted[i:] += WASTAGE * AZ_lot
            # Once we're finished our 8M local (plus 350k imported) AZ first doses, all
            # remaining supply is reserved for 2nd doses:
            if AZ_available[i] + AZ_first_doses[i] > 8.35e6:
                excess = AZ_first_doses[i] + AZ_available[i] - 8.35e6
                AZ_available[i:] -= excess
                AZ_reserved[i:] += excess
    if i == 0:
        first_doses_today = first_doses[i]
    elif i < len(dates):
        first_doses_today = first_doses[i] - first_doses[i - 1]
    else:
        # This is the assumption for projecting based on expected supply. That we use 5%
        # of available doses each day on first doses. Since a dose will be reserved as
        # well, this means we're always 10 days away from running out of vaccine at the
        # current rate - which is approximately what we see in the data.
        first_doses_today = 1 / 14 * (pfizer_available[i] + AZ_available[i])
        total_first_doses = AZ_first_doses[i] + pfizer_first_doses[i]
        first_doses_today = max(0, min(20000000 - total_first_doses, first_doses_today))
        first_doses[i:] += first_doses_today

    AZ_frac = AZ_available[i] / (AZ_available[i] + pfizer_available[i])
    pfizer_frac = pfizer_available[i] / (AZ_available[i] + pfizer_available[i])

    AZ_first_doses_today = AZ_frac * first_doses_today
    pfizer_first_doses_today = pfizer_frac * first_doses_today

    # Once we're finished our 8M local (plus 350k imported) AZ first doses, all
    # remaining supply is reserved for 2nd doses:
    # if AZ_first_doses[i] + AZ_first_doses_today > 8.35e6:
    #     excess = AZ_first_doses[i] + AZ_first_doses_today - 8.35e6
    #     AZ_first_doses_today -= excess
    #     pfizer_first_doses_today += excess
    #     AZ_reserved[i:] += AZ_available[i:]
    #     AZ_available[i:] = 0

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


days = (dates - dates[0]).astype(float)

daily_doses = np.diff(doses_by_state['aus'], prepend=0)
smoothed_daily_doses = n_day_average(daily_doses, 7)


# Model of projected dosage rate:
all_nonreserved = (
    AZ_first_doses
    + AZ_second_doses
    + AZ_available
    # + AZ_reserved
    + pfizer_first_doses
    + pfizer_second_doses
    + pfizer_available
    # + pfizer_reserved
)

nonreserved_rate = np.diff(all_nonreserved, prepend=0)
# # Smooth over the next 3 weeks:
# nonreserved_rate_smoothed = exponential_smoothing(nonreserved_rate, 21)
# # Gaussian smooth an extra week:
# nonreserved_rate_smoothed = gaussian_smoothing(nonreserved_rate, 7)
# # Delay by a week:
# tmp = np.zeros_like(nonreserved_rate_smoothed)
# tmp[7:] = nonreserved_rate_smoothed[:-7]
# nonreserved_rate_smoothed = tmp

# Delay by 3 weeks:
tmp = np.zeros_like(nonreserved_rate)
tmp[21:] = nonreserved_rate[:-21]
nonreserved_rate = tmp

# Include current doses in smoothing so as to ramp from the current rate to the
# projected rate over the short term
nonreserved_rate[: len(dates)] = daily_doses
# 7-day average:
nonreserved_rate = n_day_average(nonreserved_rate, 7)
# Gaussian smooth 1 week:
nonreserved_rate = gaussian_smoothing(nonreserved_rate, 7)

# Smooth out any remaining discontinuity:
err = nonreserved_rate[len(dates) - 2] - smoothed_daily_doses[-1]
offset = err * np.exp(-(np.arange(50)) / 14)
nonreserved_rate[len(dates) - 1 : len(dates) + 49] -= offset

def state_label(state):
    if state == 'fed':
        return 'GPs/fed. care'
    else:
        return f"{state.upper()} clinics"


fig1 = plt.figure(figsize=(8, 6))

cumsum = np.zeros(len(dates))
colours = list(reversed([f'C{i}' for i in range(9)]))
for i, state in enumerate(['nt', 'act', 'tas', 'sa', 'wa', 'qld', 'vic', 'nsw', 'fed']):
    doses = doses_by_state[state]
    latest_doses = doses[-1]

    plt.fill_between(
        dates + 1,
        cumsum / 1e6,
        (cumsum + doses) / 1e6,
        label=f'{state_label(state)} ({doses[-1] / 1000:.1f}k)',
        step='pre',
        color=colours[i],
        linewidth=0,
    )
    cumsum += doses

if PROJECT:
    plt.fill_between(
        all_dates[len(dates) - 1 :] + 1,
        (
            doses_by_state["aus"][-1]
            + nonreserved_rate[len(dates) - 1 :].cumsum()
        )
        / 1e6,
        # proj_doses[len(dates) - 1 :] / 1e6,
        label='Projection',
        step='post',
        color='cyan',
        alpha=0.5, linewidth=0,
    )

ax1 = plt.gca()

plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=PLOT_END_DATE,
    ymin=0,
    ymax=40 if LONGPROJECT else CUMULATIVE_YMAX,
)

if LONGPROJECT:
    plt.title("Projected cumulative doses")
else:
    plt.title(f'AUS cumulative doses. Total to date: {doses_by_state["aus"][-1]/1e6:.2f}M')
plt.ylabel('Cumulative doses (millions)')


fig2 = plt.figure(figsize=(8, 6))

cumsum = np.zeros(len(dates))
colours = list(reversed([f'C{i}' for i in range(9)]))
for i, state in enumerate(['nt', 'act', 'tas', 'sa', 'wa', 'qld', 'vic', 'nsw', 'fed']):
    doses = doses_by_state[state]
    # smoothed_doses = gaussian_smoothing(np.diff(doses, prepend=0), 2).cumsum()
    # smoothed_doses = padded_gaussian_smoothing(np.diff(doses, prepend=0), 2).cumsum()
    smoothed_doses = n_day_average(np.diff(doses, prepend=0), 7).cumsum()
    smoothed_doses = gaussian_smoothing(np.diff(smoothed_doses, prepend=0), 1).cumsum()
    daily_doses = np.diff(smoothed_doses, prepend=0)
    latest_daily_doses = daily_doses[-1]

    plt.fill_between(
        dates + 1,
        cumsum / 1e3,
        (cumsum + daily_doses) / 1e3,
        label=f'{state_label(state)} ({daily_doses[-1] / 1000:.1f}k/day)',
        step='pre',
        color=colours[i],
        linewidth=0,
    )
    cumsum += daily_doses


if PROJECT:
    daily_proj_doses = np.diff(proj_doses, prepend=0)
    plt.fill_between(
        all_dates[len(dates) - 1 :] + 1,
        nonreserved_rate[len(dates) - 1 :] / 1e3,
        # gaussian_smoothing(daily_proj_doses / 1e3, 4)[len(dates) - 1 :],
        # padded_gaussian_smoothing(daily_proj_doses / 1e3, 4)[len(dates) - 1 :],
        label='Projection',
        step='post',
        color='cyan',
        alpha=0.5,
        linewidth=0,
    )

latest_daily_doses = cumsum[-1]

if LONGPROJECT:
    plt.title("Projected daily doses")
else:
    plt.title(
        '7-day average daily doses by administration channel\n'
        + f'Latest national rate: {latest_daily_doses / 1000:.1f}k/day'
    )

plt.ylabel('Daily doses (thousands)')

plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=PLOT_END_DATE,
    ymin=0,
    ymax=300 if LONGPROJECT else 120,
)
ax2 = plt.gca()


if LONGPROJECT:
    endindex = len(all_dates)
else:
    endindex = len(dates)
    
fig3 = plt.figure(figsize=(8, 6))
cumsum = np.zeros(len(all_dates))
for arr, label, colour in [
    (AZ_first_doses + pfizer_first_doses, 'Administered first doses', 'C0'),
    (AZ_available + pfizer_available, 'Available for first doses', 'C2'),
    (AZ_second_doses + pfizer_second_doses, 'Administered second doses', 'C1'),
    (AZ_reserved + pfizer_reserved, 'Reserved for second doses', 'C3'),
    # (wasted, 'Wasted', 'C4'),
]:
    plt.fill_between(
        all_dates[: endindex] + 1,
        cumsum[: endindex] / 1e6,
        (cumsum + arr)[: endindex] / 1e6,
        label=f'{label} ({arr[len(dates)-1] / 1000:.0f}k)',
        step='pre',
        color=colour,
        linewidth=0,
    )
    cumsum += arr

used = AZ_first_doses[len(dates) - 1] + pfizer_first_doses[len(dates) - 1]
unused = AZ_available[len(dates) - 1] + pfizer_available[len(dates) - 1]
utilisation = 100 * used / (used + unused)
plt.ylabel('Cumulative doses (millions)')
plt.title(f"Estimated vaccine utilisation: first dose utilisation rate: {utilisation:.1f}%")
plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=PLOT_END_DATE,
    ymin=0,
    ymax=40 if LONGPROJECT else CUMULATIVE_YMAX,
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
        all_dates[: endindex] + 1,
        cumsum[: endindex] / 1e6,
        (cumsum + arr)[: endindex] / 1e6,
        label=f'{label} ({arr[len(dates)-1] / 1000:.0f}k)',
        step='pre',
        color=colour,
        linewidth=0,
    )
    cumsum += arr

used = AZ_first_doses[len(dates) - 1]
unused = AZ_available[len(dates) - 1]
utilisation = 100 * used / (used + unused)
plt.ylabel('Cumulative doses (millions)')
plt.title(
    f"Estimated AZ vaccine utilisation: first dose utilisation rate: {utilisation:.1f}%"
)
plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=PLOT_END_DATE,
    ymin=0,
    ymax=40 if LONGPROJECT else CUMULATIVE_YMAX,
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
        all_dates[: endindex] + 1,
        cumsum[: endindex] / 1e6,
        (cumsum + arr)[: endindex] / 1e6,
        label=f'{label} ({arr[len(dates)-1] / 1000:.0f}k)',
        step='pre',
        color=colour,
        linewidth=0,
    )
    cumsum += arr

used = pfizer_first_doses[len(dates) - 1]
unused = pfizer_available[len(dates) - 1]
utilisation = 100 * used / (used + unused)
plt.ylabel('Cumulative doses (millions)')
plt.title(
    f"Estimated Pfizer vaccine utilisation: first dose utilisation rate: {utilisation:.1f}%"
)
plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=PLOT_END_DATE,
    ymin=0,
    ymax=40 if LONGPROJECT else CUMULATIVE_YMAX,
)
ax5 = plt.gca()


for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.fill_betweenx(
        [0, ax.get_ylim()[1]],
        2 * [START_DATE.astype(int)],
        2 * [PHASE_1B.astype(int)],
        color='red',
        alpha=0.35,
        linewidth=0,
        label='Phase 1a',
        zorder=-10,
    )

    ax.fill_betweenx(
        [0, ax.get_ylim()[1]],
        2 * [PHASE_1B.astype(int)],
        2 * [PHASE_2A.astype(int)],
        color='orange',
        alpha=0.35,
        linewidth=0,
        label='Phase 1b',
        zorder=-10,
    )

    ax.fill_betweenx(
        [0, ax.get_ylim()[1]],
        2 * [PHASE_2A.astype(int)],
        2 * [max(dates[-1], PHASE_2A).astype(int) + 20],
        color='yellow',
        alpha=0.35,
        linewidth=0,
        label='Phase 2a',
        zorder=-10,
    )

    for i in range(10):
        ax.fill_betweenx(
            [0, ax.get_ylim()[1]],
            2 * [max(dates[-1], PHASE_2A).astype(int) + 20 + i],
            2 * [max(dates[-1], PHASE_2A).astype(int) + 21 + i],
            color='yellow',
            alpha=0.3 * (10 - i) / 10,
            linewidth=0,
            zorder=-10,
        )


handles, labels = ax1.get_legend_handles_labels()
if PROJECT:
    order = [8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 10, 11, 12]
else:
    order = [8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 10, 11]
ax1.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    ncol=2,
    fontsize="small"
)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(5 if LONGPROJECT else 1))


handles, labels = ax2.get_legend_handles_labels()
if PROJECT:
    order = [8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 10, 11, 12]
else:
    order = [8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 10, 11]
ax2.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    ncol=2,
    fontsize="small"
)


for ax in [ax3, ax4, ax5]:
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5 if LONGPROJECT else 1))
    handles, labels = ax.get_legend_handles_labels()
    order = [3, 2, 1, 0, 4, 5, 6]
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc='upper left',
        # ncol=2,
        fontsize="small"
    )

for ax in [ax1, ax2, ax3, ax4, ax5]:
    locator = mdates.DayLocator([1] if LONGPROJECT else [1, 15])
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.get_xaxis().get_major_formatter().show_offset = False
    ax.grid(True, linestyle=":", color='k')


# Plot of doses by weekday
fig6 = plt.figure(figsize=(8, 6))

doses_by_day = np.diff(doses_by_state['aus'])
doses_by_day = np.append(doses_by_day, [np.nan] * (7 - len(doses_by_day) % 7))
N_WEEKS = 4
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in reversed(range(N_WEEKS)):
    start = len(doses_by_day) + (-N_WEEKS + i) * 7
    block = doses_by_day[start : start + 7]
    date = dates[start].astype(datetime).strftime('%B %d')
    plt.plot(days, block / 1e3, 'o-', label=f"Week beginning {date}")
plt.grid(True, linestyle=':', color='k', alpha=0.5)
# plt.gca().set_xticklabels()
plt.legend()
plt.ylabel('Daily doses (thousands)')
plt.axis(ymin=0)
plt.title('National daily doses by weekday')


# Update the date in the HTML
html_file = 'aus_vaccinations.html'
html_lines = Path(html_file).read_text().splitlines()
now = datetime.now(timezone('Australia/Melbourne')).strftime('%Y-%m-%d-%H:%M')
for i, line in enumerate(html_lines):
    if 'Last updated' in line:
        html_lines[i] = f'    Last updated: {now} Melbourne time'
Path(html_file).write_text('\n'.join(html_lines) + '\n')

for extension in ['png', 'svg']:
    if LONGPROJECT:
        fig1.savefig(f'cumulative_doses_longproject.{extension}')
        fig2.savefig(f'daily_doses_by_state_longproject.{extension}')
    else:
        fig1.savefig(f'cumulative_doses.{extension}')
        fig2.savefig(f'daily_doses_by_state.{extension}')
        fig3.savefig(f'utilisation.{extension}')
        fig4.savefig(f'az_utilisation.{extension}')
        fig5.savefig(f'pfizer_utilisation.{extension}')
        fig6.savefig(f'doses_by_weekday.{extension}')

plt.show()
