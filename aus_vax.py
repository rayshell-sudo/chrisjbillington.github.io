import sys
from datetime import datetime
import json
import requests
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


def get_data():
    COVIDLIVE = 'https://covidlive.com.au/covid-live.json'
    # PDFPARSER = "https://vaccinedata.covid19nearme.com.au/data/all.json"
    covidlivedata = json.loads(requests.get(COVIDLIVE).content)
    # pdfdata = json.loads(requests.get(PDFPARSER).content)[-1]

    START_DATE = np.datetime64('2021-02-21')

    doses_by_state = {
        'AUS': [],
        'NSW': [],
        'VIC': [],
        'SA': [],
        'WA': [],
        'TAS': [],
        'QLD': [],
        'NT': [],
        'ACT': [],
    }

    # Get data before today from covidlive:
    # YESTERDAY = np.datetime64(datetime.now().strftime('%Y-%m-%d')) - 1
    for report in covidlivedata:
        date = np.datetime64(report['REPORT_DATE']) - 1
        # if date == YESTERDAY:
        #     continue
        if report['VACC_DOSE_CNT'] is None:
            continue
        state = report['CODE']
        doses = int(report['VACC_DOSE_CNT'])
        if state != 'AUS' and report['VACC_GP_CNT'] is not None:
            doses -= int(report['VACC_AGED_CARE_CNT']) + int(report['VACC_GP_CNT'])
        doses_by_state[state].append((date, doses))

    for state, data in doses_by_state.items():
        data.sort()
        dates, doses = [np.array(a) for a in zip(*data)]
        doses_by_state[state] = dates[dates >= START_DATE], doses[dates >= START_DATE]

    # Truncate all to most recent date all jurisdictions have data for:
    valid_n_dates = min(len(d) for d, _ in doses_by_state.values())
    doses_by_state = {
        k: (dates[:valid_n_dates], doses[:valid_n_dates])
        for k, (dates, doses) in doses_by_state.items()
    }

    for dates, _ in doses_by_state.values():
        assert np.array_equal(dates, doses_by_state['AUS'][0])

    dates, *_ = doses_by_state['AUS']
    for state, (_, doses) in doses_by_state.items():
        doses_by_state[state] = doses.astype(float)

    # Get data for today, if it exists, from jxeeno/aust-govt-covid19-vaccine-pdf:
    # if np.datetime64(pdfdata['DATE_AS_AT']) == YESTERDAY:
    #     dates = np.append(dates, [YESTERDAY])
    #     for state in doses_by_state:
    #         if state == 'AUS':
    #             doses = int(pdfdata['TOTALS_NATIONAL_TOTAL'])
    #         else:
    #             doses = int(pdfdata[f'STATE_CLINICS_{state}_TOTAL'])
    #         doses_by_state[state] = np.append(doses_by_state[state], [doses])

    return dates, doses_by_state

dates, doses_by_state = get_data()

PHASE_1B = np.datetime64('2021-03-22')
PHASE_2A = np.datetime64('2021-05-03')
PHASE_2B = np.datetime64('2021-08-30')

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

# A correction of -9260 was applied to VIC's numbers on May 25th. We spread these out
# proportional to each day's doses from the start of phase 1b until May 23th.
VIC_DOSES_CORRECTION = -9260
daily_vic_doses = np.diff(doses_by_state['VIC'], prepend=0)
reportedix = np.where(dates == np.datetime64('2021-05-24'))[0][0]
backdates = (PHASE_1B <= dates) & (dates <= np.datetime64('2021-05-23'))
daily_vic_doses[reportedix] -= VIC_DOSES_CORRECTION
total_in_backdate_period = daily_vic_doses[backdates].sum()
daily_vic_doses[backdates] *= 1 + VIC_DOSES_CORRECTION / total_in_backdate_period
doses_by_state['AUS'] -= (daily_vic_doses.cumsum() - doses_by_state['VIC'])
doses_by_state['VIC'] = daily_vic_doses.cumsum()


doses_by_state['FED'] = doses_by_state['AUS'] - sum(
    doses_by_state[s] for s in doses_by_state if s != 'AUS'
)

# 80560 doses were reported on April 19th that actually were administered "prior to
# April 17". We don't know how much prior, so we'll spread these doses out proportional
# to each day's doses from the start of phase 1b until April 16.
LATE_REPORTED_GP_DOSES = 80560
daily_fed_doses = np.diff(doses_by_state['FED'], prepend=0)
reportedix = np.where(dates == np.datetime64('2021-04-18'))[0][0]
backdates = (PHASE_1B <= dates) & (dates <= np.datetime64('2021-04-17'))
daily_fed_doses[reportedix] -= LATE_REPORTED_GP_DOSES
total_in_backdate_period = daily_fed_doses[backdates].sum()
daily_fed_doses[backdates] *= 1 + LATE_REPORTED_GP_DOSES / total_in_backdate_period
doses_by_state['FED'] = daily_fed_doses.cumsum()



doses = doses_by_state['AUS']


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
2021-05-30      2_924_340
2021-06-06      3_222_690
2021-06-13      3_534_670 + 500_000 # Direct plus COVAX
2021-06-20      3_833_020 + 500_000
2021-06-27      4_131_370 + 500_000
2021-07-04      4_436_740 + 500_000
2021-07-11      4_729_000 + 500_000
2021-07-18      5_238_190 + 500_000
2021-07-25      6_238_190 + 500_000
2021-08-01      7_238_190 + 500_000
# Below not yet reflected on covid19data.com.au, but pretty sure I heard numbers like
# this in the federal pressers, and it is what was expected:
2021-08-08      7_238_190 + 500_000 +     1_125_000
2021-08-15      7_238_190 + 500_000 + 2 * 1_125_000 + 1_000_000 # Thanks Poland
2021-08-22      7_238_190 + 500_000 + 3 * 1_125_000 + 1_000_000
2021-08-29      7_238_190 + 500_000 + 4 * 1_125_000 + 1_000_000
2021-09-05      7_238_190 + 500_000 + 4 * 1_125_000 + 1_000_000 + 1_000_000 + 200_000 + 500_000 # Regular, Moderna, and Singapore Pfizer
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
2021-05-30      6_739_200
2021-06-06      7_688_500
2021-06-13      7_921_300
2021-06-20      8_642_400
2021-06-27      9_319_000
2021-07-04     10_023_000
2021-07-11     10_238_100
2021-07-18     10_564_600
2021-07-25     11_500_000
2021-08-01     12_300_000
2021-08-08     13_200_000
2021-08-15     14_000_000
2021-08-22     14_900_000
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

PFIZER_PROJECTED_SHIPMENTS= """ # In thousands per week
2021-09-12 1000 + 200  + 4000 # Pfizer + Moderna + UK Pfizer doses
2021-09-19 1000 + 200
2021-09-26 1000 + 200
2021-10-03 2000 + 200 
2021-10-10 2000 + 750 
2021-10-17 2000 + 750 
2021-10-24 2000 + 750 
2021-10-31 2000 + 750 
2021-11-07 2000 + 750 
2021-11-14 2000 + 750 
2021-11-21 2000 + 750 
2021-11-28 2000 + 750 
2021-12-05 2000 + 600 
2021-12-12 2000 + 600 
2021-12-19 2000 + 600 
2021-12-26 2000 + 600 
2022-01-02 2000 + 600
"""

PLOT_END_DATE = (
    np.datetime64('2022-01-31') if LONGPROJECT else dates[-1] + 50 #np.datetime64('2021-05-31')
)
CUMULATIVE_YMAX = 35  # million
DAILY_YMAX = 380

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

AZ_WASTAGE = 0.125
PFIZER_WASTAGE = 0.05

# Number of AZ first doses

# 90% of ~5M over sixties, 40% of ~3M people in their fifties, 50% of 1M NSW residents
# in their forties, 50% of ~2M NSW residents 18-39, 25% of ~2M VIC residents 20-39 plus
# 350k from early in the rollout
MAX_AZ_ADMINISTERED = .9 * 5e6 + .4 * 3e6 + .5 * 1e6 + 0.5 * 2e6 + 0.25 * 2e6 + 350e3

# Number of people 12 years old and older, from ABS ERP June 2020
MAX_ELIGIBLE = 21_852_349

# Number of people 16 years old and older, from ABS ERP June 2020
POP_16_PLUS = 20_607_204

# Estimated AZ supply. Assume 1M per week locally-produced AZ up to ~10.8M (plus
# wastage):
AZ_MAX_DOSES = 2 * MAX_AZ_ADMINISTERED / (1 - AZ_WASTAGE)
n_weeks = int((AZ_MAX_DOSES - AZ_local_supply[-1]) // 1000000) + 1
AZ_local_supply_dates = np.append(
    AZ_local_supply_dates,
    [AZ_local_supply_dates[-1] + 7 * (i + 1) for i in range(n_weeks)],
)
AZ_local_supply = np.append(
    AZ_local_supply, [AZ_local_supply[-1] + 1000000 * (i + 1) for i in range(n_weeks)]
)
AZ_local_supply[-1] = AZ_MAX_DOSES


# Estimated Pfizer supply.

for d, p in zip(*unpack_data(PFIZER_PROJECTED_SHIPMENTS)):
    pfizer_supply_dates = np.append(pfizer_supply_dates, [d])
    pfizer_supply = np.append(pfizer_supply, [pfizer_supply[-1] + p * 1000])


pfizer_shipments = np.diff(pfizer_supply, prepend=0)
AZ_shipments = np.diff(AZ_OS_suppy, prepend=0)
AZ_production = np.diff(AZ_local_supply, prepend=0)

if PROJECT:
    if LONGPROJECT:
        projection_end = np.datetime64('2022-01-31')
    else:
        projection_end = np.datetime64('2022-01-31')
    projection_dates = np.arange(dates[-1] + 1, projection_end)
    all_dates = np.concatenate((dates, projection_dates))
else:
    all_dates = dates

for d, p in zip(pfizer_supply_dates, pfizer_shipments):
    print(str(d), p / 1000)
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

tau_AZ = 77
tau_pfizer = 25

pfizer_available += pfizer_shipments[pfizer_supply_dates < dates[0]].sum()
AZ_available += AZ_shipments[AZ_OS_supply_dates < dates[0]].sum()
AZ_available += AZ_production[AZ_local_supply_dates < dates[0]].sum()


for i, date in enumerate(all_dates):
    if date in pfizer_supply_dates:
        pfizer_lot = pfizer_shipments[pfizer_supply_dates == date][0]
        if date < np.datetime64('2021-06-27'):
            pfizer_available[i:] +=  0.5 * (1 - PFIZER_WASTAGE) * pfizer_lot
            pfizer_reserved[i:] += 0.5 * (1 - PFIZER_WASTAGE) * pfizer_lot
            wasted[i:] += PFIZER_WASTAGE * pfizer_lot
        else:
            outstanding_pfizer_second_doses = pfizer_first_doses[i] - pfizer_second_doses[i]
            reserve_allocation = 0.4 * outstanding_pfizer_second_doses - pfizer_reserved[i]
            pfizer_available[i:] += (1 - PFIZER_WASTAGE) * pfizer_lot - reserve_allocation
            pfizer_reserved[i:] += reserve_allocation
            wasted[i:] += PFIZER_WASTAGE * pfizer_lot

    if date in AZ_OS_supply_dates:
        AZ_lot = AZ_shipments[AZ_OS_supply_dates == date][0]
        AZ_available[i:] += 0.5 * (1 - AZ_WASTAGE) * AZ_lot
        AZ_reserved[i:] += 0.5 * (1 - AZ_WASTAGE) * AZ_lot
        wasted[i:] += AZ_WASTAGE * AZ_lot
    if date in AZ_local_supply_dates:
        AZ_lot = AZ_production[AZ_local_supply_dates == date][0]
        if date < np.datetime64('2021-04-11'):
            AZ_available[i:] += 0.5 * (1 - AZ_WASTAGE) * AZ_lot
            AZ_reserved[i:] += 0.5 * (1 - AZ_WASTAGE) * AZ_lot
        else:
            outstanding_AZ_second_doses = AZ_first_doses[i] - AZ_second_doses[i]
            reserve_allocation = 0.66 * outstanding_AZ_second_doses - AZ_reserved[i]
            AZ_available[i:] += (1 - AZ_WASTAGE) * AZ_lot - reserve_allocation
            AZ_reserved[i:] += reserve_allocation
            wasted[i:] += AZ_WASTAGE * AZ_lot
            # Once we're finished our 5M local (plus 350k imported) AZ first doses, all
            # remaining supply is reserved for 2nd doses:
            if AZ_available[i] + AZ_first_doses[i] > MAX_AZ_ADMINISTERED:
                excess = AZ_first_doses[i] + AZ_available[i] - MAX_AZ_ADMINISTERED
                AZ_available[i:] -= excess
                AZ_reserved[i:] += excess
    if i == 0:
        first_doses_today = first_doses[i]
    elif i < len(dates):
        first_doses_today = first_doses[i] - first_doses[i - 1]
    if i < len(dates):
        P = 2 # Pfizer preference factor
        AZ_frac = AZ_available[i] / (AZ_available[i] + P * pfizer_available[i])
        pfizer_frac = P * pfizer_available[i] / (AZ_available[i] + P * pfizer_available[i])

        AZ_first_doses_today = AZ_frac * first_doses_today
        pfizer_first_doses_today = pfizer_frac * first_doses_today
    else:
        # This is the assumption for projecting based on expected supply. That we use 5%
        # of available doses each day on first doses. Since a dose will be reserved as
        # well, this means we're always 10 days away from running out of vaccine at the
        # current rate - which is approximately what we see in the data.
        AZ_first_doses_today = 1 / 21 * AZ_available[i]
        pfizer_first_doses_today = 1 / 21 * pfizer_available[i]

        first_doses_today = AZ_first_doses_today + pfizer_first_doses_today
        total_first_doses = AZ_first_doses[i] + pfizer_first_doses[i]
        first_doses_today = max(
            0, min(MAX_ELIGIBLE - total_first_doses, first_doses_today)
        )
        pfizer_first_doses_today = first_doses_today - AZ_first_doses_today
        first_doses[i:] += first_doses_today

    # AZ_frac = AZ_available[i] / (AZ_available[i] + pfizer_available[i])
    # pfizer_frac = pfizer_available[i] / (AZ_available[i] + pfizer_available[i])

    # AZ_first_doses_today = AZ_frac * first_doses_today
    # pfizer_first_doses_today = pfizer_frac * first_doses_today

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

daily_doses = np.diff(doses_by_state['AUS'], prepend=0)
smoothed_daily_doses = n_day_average(daily_doses, 7)


# Model of projected dosage rate:
# all_nonreserved = (
#     AZ_first_doses
#     + AZ_second_doses
#     # + AZ_available
#     # + AZ_reserved
#     + pfizer_first_doses
#     + pfizer_second_doses
#     # + pfizer_available
#     # + pfizer_reserved
# )

def diff_and_smooth(dat):
    dat = np.diff(dat, prepend=0)
    dat = n_day_average(dat, 7)
    dat = gaussian_smoothing(dat, 7)
    return dat

# nonreserved_rate = np.diff(all_nonreserved, prepend=0)
# # Smooth over the next 3 weeks:
# nonreserved_rate_smoothed = exponential_smoothing(nonreserved_rate, 21)
# # Gaussian smooth an extra week:
# nonreserved_rate_smoothed = gaussian_smoothing(nonreserved_rate, 7)
# # Delay by a week:
# tmp = np.zeros_like(nonreserved_rate_smoothed)
# tmp[7:] = nonreserved_rate_smoothed[:-7]
# nonreserved_rate_smoothed = tmp

# Delay by 1 weeks:
# tmp = np.zeros_like(nonreserved_rate)
# tmp[7:] = nonreserved_rate[:-7]
# nonreserved_rate = tmp

# Include current doses in smoothing so as to ramp from the current rate to the
# projected rate over the short term
# nonreserved_rate[: len(dates)] = daily_doses
# # 7-day average:
# nonreserved_rate = n_day_average(nonreserved_rate, 7)
# # Gaussian smooth 1 week:
# nonreserved_rate = gaussian_smoothing(nonreserved_rate, 7)

nonreserved_rate = diff_and_smooth(
    AZ_first_doses + AZ_second_doses + pfizer_first_doses + pfizer_second_doses
)
# Smooth out any remaining discontinuity:
err = nonreserved_rate[len(dates) - 1] - smoothed_daily_doses[-1]
offset = err * np.exp(-(np.arange(50)) / 14)
nonreserved_rate[len(dates) - 1 : len(dates) + 49] -= offset

def state_label(state):
    if state == 'FED':
        return 'GPs/fed. care'
    else:
        return f"{state.upper()} clinics"


fig1 = plt.figure(figsize=(8, 6))

cumsum = np.zeros(len(dates))
colours = list(reversed([f'C{i}' for i in range(9)]))
for i, state in enumerate(['NT', 'ACT', 'TAS', 'SA', 'WA', 'QLD', 'VIC', 'NSW', 'FED']):
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
            doses_by_state['AUS'][-1]
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
    ymax=2*MAX_ELIGIBLE/1e6 if LONGPROJECT else CUMULATIVE_YMAX,
)

latest_cumulative_doses = doses_by_state["AUS"][-1]

if LONGPROJECT:
    plt.title("Projected cumulative doses")
else:
    plt.title(f'AUS cumulative doses. Total to date: {latest_cumulative_doses/1e6:.2f}M')
plt.ylabel('Cumulative doses (millions)')


fig2 = plt.figure(figsize=(8, 6))

cumsum = np.zeros(len(dates))
colours = list(reversed([f'C{i}' for i in range(9)]))
for i, state in enumerate(['NT', 'ACT', 'TAS', 'SA', 'WA', 'QLD', 'VIC', 'NSW', 'FED']):
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
    ymax=DAILY_YMAX,
)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(20))
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
    ymax=2*MAX_ELIGIBLE/1e6 if LONGPROJECT else CUMULATIVE_YMAX,
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
    ymax=2*MAX_ELIGIBLE/1e6 if LONGPROJECT else CUMULATIVE_YMAX,
)
ax4 = plt.gca()


fig5 = plt.figure(figsize=(8, 6))
cumsum = np.zeros(len(all_dates))
for arr, label, colour in [
    (pfizer_first_doses, 'Pfizer/Moderna administered first doses', 'C0'),
    (pfizer_available, 'Pfizer/Moderna available for first doses', 'C2'),
    (pfizer_second_doses, 'Pfizer/Moderna administered second doses', 'C1'),
    (pfizer_reserved, 'Pfizer/Moderna reserved for second doses', 'C3'),
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
    f"Estimated Pfizer/Moderna vaccine utilisation: first dose utilisation rate: {utilisation:.1f}%"
)
plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=PLOT_END_DATE,
    ymin=0,
    ymax=2*MAX_ELIGIBLE/1e6 if LONGPROJECT else CUMULATIVE_YMAX,
)
ax5 = plt.gca()


# Plot of projection by dose type
fig6 = plt.figure(figsize=(8, 6))
cumsum = np.zeros(len(all_dates))
for doses, label in [
    (pfizer_first_doses, "Pfizer/Moderna first doses"),
    (pfizer_second_doses, "Pfizer/Moderna second doses"),
    (AZ_first_doses, "AZ first doses"),
    (AZ_second_doses, "AZ second doses"),
]:
    rate = diff_and_smooth(doses)
    plt.fill_between(
        all_dates + 1,
        cumsum / 1e3,
        (cumsum + rate) / 1e3,
        label=label,
        step='pre',
        # color=colours[i],
        linewidth=0,
    )
    cumsum += rate
plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=PLOT_END_DATE,
    ymin=0,
    ymax=DAILY_YMAX,
)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(20))
plt.title('Projected daily doses by type')
plt.ylabel('Daily doses (thousands)')
today = np.datetime64(datetime.now(), 'D')
plt.axvline(today, linestyle=":", color='k', label=f"Today ({today})")
ax6 = plt.gca()


# Plot of projection 1st vs 2nd doses
fig7 = plt.figure(figsize=(8, 6))

proj_first_doses = diff_and_smooth(AZ_first_doses + pfizer_first_doses).cumsum()
proj_second_doses = diff_and_smooth(AZ_second_doses + pfizer_second_doses).cumsum()

plt.step(
    all_dates + 1,
    proj_first_doses / 1e6,
    where='pre',
    label="First doses",
)
plt.step(
    all_dates + 1,
    proj_second_doses / 1e6,
    where='pre',
    label="Second doses",
)
    
# all_first_doses = diff_and_smooth(AZ_first_doses + pfizer_first_doses).cumsum()
# all_second_doses = diff_and_smooth(AZ_second_doses + pfizer_second_doses).cumsum()
# adult_first_dose_percent = 100 * all_first_doses / POP_16_PLUS
# adult_second_dose_percent = 100 * all_second_doses / POP_16_PLUS
# for i, date in enumerate(all_dates):
#     print(
#         date,
#         f"first dose: {adult_first_dose_percent[i]:.02f}",
#         f"second dose: {adult_second_dose_percent[i]:.02f}",
#     )

plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=PLOT_END_DATE,
    ymin=0,
    ymax=MAX_ELIGIBLE/1e6 if LONGPROJECT else CUMULATIVE_YMAX,

)
plt.title('Projected cumulative 1st and 2nd doses')
plt.ylabel('Cumulative doses (millions)')
today = np.datetime64(datetime.now(), 'D')
plt.axvline(today, linestyle=":", color='k', label=f"Today ({today})")
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2.0))
ax7 = plt.gca()

PHASE_B_DATE = all_dates[proj_second_doses.searchsorted(0.7 * POP_16_PLUS)] + 1
PHASE_C_DATE = all_dates[proj_second_doses.searchsorted(0.8 * POP_16_PLUS)] + 1

plt.axhline(
    0.7 * POP_16_PLUS / 1e6,
    linestyle="--",
    color='C4',
    label=f"Phase B 70% target ({PHASE_B_DATE})",
)
plt.axhline(
    0.8 * POP_16_PLUS / 1e6,
    linestyle="--",
    color='C2',
    label=f"Phase C 80% target ({PHASE_C_DATE})",
)

twinax = plt.twinx()
twinax.axis(ymin=0, ymax=100 * MAX_ELIGIBLE / POP_16_PLUS)
twinax.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
plt.ylabel("Percentage of population aged 16+")


for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
    ax.fill_betweenx(
        [0, ax.get_ylim()[1]],
        2 * [dates[0].astype(int)],
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
        2 * [PHASE_2B.astype(int)],
        color='yellow',
        alpha=0.35,
        linewidth=0,
        label='Phase 2a',
        zorder=-10,
    )

    ax.fill_betweenx(
        [0, ax.get_ylim()[1]],
        2 * [PHASE_2B.astype(int)],
        2 * [max(dates[-1], PHASE_2B).astype(int) + 20],
        color='green',
        alpha=0.25,
        linewidth=0,
        label='Phase 2b',
        zorder=-10,
    )

    for i in range(10):
        ax.fill_betweenx(
            [0, ax.get_ylim()[1]],
            2 * [max(dates[-1], PHASE_2B).astype(int) + 20 + i],
            2 * [max(dates[-1], PHASE_2B).astype(int) + 21 + i],
            color='green',
            alpha=0.25 * (10 - i) / 10,
            linewidth=0,
            zorder=-10,
        )


handles, labels = ax1.get_legend_handles_labels()
if PROJECT:
    order = [8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 10, 11, 12, 13]
else:
    order = [8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 10, 11, 12]
ax1.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    # ncol=2,
    fontsize="small"
)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(5 if LONGPROJECT else 2))


handles, labels = ax2.get_legend_handles_labels()
if PROJECT:
    order = [8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 10, 11, 12, 13]
else:
    order = [8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 10, 11, 12]
ax2.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    # ncol=2,
    fontsize="small"
)


for ax in [ax3, ax4, ax5]:
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5 if LONGPROJECT else 2))
    handles, labels = ax.get_legend_handles_labels()
    order = [3, 2, 1, 0, 4, 5, 6, 7]
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc='upper left',
        # ncol=2,
        fontsize="small"
    )

handles, labels = ax6.get_legend_handles_labels()
order = [1, 2, 3, 4, 5, 6, 7, 8, 0]
ax6.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    # ncol=2,
    fontsize="small"
)


handles, labels = ax7.get_legend_handles_labels()
order = [0, 1, 5, 6, 7, 8, 3, 4, 2]
ax7.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='lower right',
    # ncol=2,
    fontsize="small"
)

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
    locator = mdates.DayLocator([1] if LONGPROJECT else [1, 15])
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.get_xaxis().get_major_formatter().show_offset = False
    ax.grid(True, linestyle=":", color='k')


# Plot of doses by weekday
fig8 = plt.figure(figsize=(8, 6))

doses_by_day = np.diff(doses_by_state['AUS'])
if len(doses_by_day) % 7:
    doses_by_day = np.append(doses_by_day, [np.nan] * (7 - len(doses_by_day) % 7))
N_WEEKS = 5
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in reversed(range(N_WEEKS)):
    start = len(doses_by_day) + (-N_WEEKS + i) * 7
    block = doses_by_day[start : start + 7]
    date = (dates[start] + 1).astype(datetime).strftime('%B %d')
    plt.plot(days, block / 1e3, 'o-', label=f"Week beginning {date}",  zorder=i)
plt.grid(True, linestyle=':', color='k', alpha=0.5)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(20))
# plt.gca().set_xticklabels()
plt.legend()
plt.ylabel('Daily doses (thousands)')
plt.axis(ymin=0)
plt.title('National daily doses by weekday')


# Plots of percent coverage by age group

# ABS Estimated Resident Population, June 2020
# https://www.abs.gov.au/statistics/people/population/national-state-and-territory-population/jun-2020
POP_DATA = {
    '16_19': {'MALE': 614_430, 'FEMALE': 581_206},
    '20_24': {'MALE': 880_327, 'FEMALE': 832_409},
    '25_29': {'MALE': 960_798, 'FEMALE': 945_753},
    '30_34': {'MALE': 948_799, 'FEMALE': 975_556},
    '35_39': {'MALE': 909_066, 'FEMALE': 926_513},
    '40_44': {'MALE': 805_245, 'FEMALE': 815_516},
    '45_49': {'MALE': 825_971, 'FEMALE': 850_749},
    '50_54': {'MALE': 763_177, 'FEMALE': 800_990},
    '55_59': {'MALE': 759_231, 'FEMALE': 793_509},
    '60_64': {'MALE': 695_820, 'FEMALE': 736_391},
    '65_69': {'MALE': 607_161, 'FEMALE': 647_157},
    '70_74': {'MALE': 539_496, 'FEMALE': 563_487},
    '75_79': {'MALE': 370_469, 'FEMALE': 402_454},
    '80_84': {'MALE': 239_703, 'FEMALE': 288_405},
    '85_89': {'MALE': 130_999, 'FEMALE': 185_099},
    '90_94': {'MALE': 57_457, 'FEMALE':  100_955},
    '95_PLUS': {'MALE': 15_720, 'FEMALE': 37_186},
}

for pops in POP_DATA.values():
    pops['TOTAL'] = pops['MALE'] + pops['FEMALE'] 

AIR_JSON = "https://vaccinedata.covid19nearme.com.au/data/air.json"
AIR_data = json.loads(requests.get(AIR_JSON).content)
with open('covidbase_data.json') as f:
    covidbase_data = json.load(f)

# Convert dates in both datasets to np.datetime64:
for row in AIR_data + covidbase_data:
    row['DATE_AS_AT'] = np.datetime64(row['DATE_AS_AT'])

# Merge the datasets, using covidbase prior to June 30 and Ken Tsang's data thereafter:
AIR_START_DATE = np.datetime64('2021-06-30')

doses_by_age = AIR_data + [
    row for row in covidbase_data if row['DATE_AS_AT'] < AIR_START_DATE
]
doses_by_age.sort(key=lambda x: x['DATE_AS_AT'])

first_dose_coverage_dates = np.array([row['DATE_AS_AT'] for row in doses_by_age])
second_dose_coverage_dates = first_dose_coverage_dates[
    first_dose_coverage_dates >= AIR_START_DATE
]

labels_by_age = []
first_dose_coverage_by_age = []
second_dose_coverage_by_age = []

for group_start in [16, 20, 30, 40, 50, 60, 70, 80, 90][::-1]:
    if group_start == 16:
        ranges = ['16_19']
    elif group_start == 90:
        ranges = ['90_94', '95_PLUS']
    else:
        ranges = [f'{group_start}_{group_start+4}', f'{group_start+5}_{group_start+9}']
    first_dose_coverage = []
    second_dose_coverage = []
    for row in doses_by_age:
        first_doses = 0
        second_doses = 0
        pop = 0
        for age_range in ranges:
            first_doses += row[f'AIR_{age_range}_FIRST_DOSE_COUNT']
            if row['DATE_AS_AT'] >= AIR_START_DATE:
                second_doses += row[f'AIR_{age_range}_SECOND_DOSE_COUNT']
            pop += POP_DATA[age_range]['TOTAL']
        first_dose_coverage.append(100 * first_doses / pop)
        if row['DATE_AS_AT'] >= AIR_START_DATE:
            second_dose_coverage.append(100 * second_doses / pop)
    first_dose_coverage_by_age.append(np.array(first_dose_coverage))
    second_dose_coverage_by_age.append(np.array(second_dose_coverage))
    if group_start == 90:
        label = 'Ages 90+'
    else:
        label = f'Ages {ranges[0].split("_")[0]}–{ranges[-1].split("_")[-1]}'
    labels_by_age.append(label)

fig9 = plt.figure(figsize=(8, 6))
for coverage, label in zip(first_dose_coverage_by_age, labels_by_age):
    plt.plot(
        first_dose_coverage_dates, coverage, label=f"{label} ({coverage[-1]:.1f} %)"
    )

plt.legend(loc='upper left', prop={'size': 9})
plt.grid(True, linestyle=':', color='k', alpha=0.5)
locator = mdates.DayLocator([1, 15])
formatter = mdates.ConciseDateFormatter(locator)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
plt.axis(
    xmin=np.datetime64('2021-05-09'), xmax=np.datetime64('2021-10-01'), ymin=0, ymax=100
)
plt.title("First dose coverage by age group")
plt.ylabel("Vaccine coverage (%)")


# import pickle

# with open('vax_by_age.pickle', 'wb') as f:
#     pickle.dump(
#         (first_dose_coverage_dates, first_dose_coverage_by_age, labels_by_age), f
#     )


fig10 = plt.figure(figsize=(8, 6))
for coverage, label in zip(first_dose_coverage_by_age, labels_by_age):
    smoothed_coverage = 7 * n_day_average(np.diff(coverage), 7)[7:]
    plt.plot(
        first_dose_coverage_dates[8:],
        smoothed_coverage,
        label=f"{label} ({smoothed_coverage[-1]:.1f} %/week)",
    )
plt.legend(loc='upper center', prop={'size': 9})
plt.grid(True, linestyle=':', color='k', alpha=0.5)
locator = mdates.DayLocator([1, 15])
formatter = mdates.ConciseDateFormatter(locator)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1.0))
plt.axis(
    xmin=np.datetime64('2021-05-09'), xmax=np.datetime64('2021-10-01'), ymin=0, ymax=10
)
plt.title("First dose weekly increase by age group")
plt.ylabel("Vaccination rate (% of age group / week)")

fig11 = plt.figure(figsize=(8, 6))
for coverage, label in zip(second_dose_coverage_by_age, labels_by_age):
    plt.plot(
        second_dose_coverage_dates, coverage, label=f"{label} ({coverage[-1]:.1f} %)"
    )

plt.legend(loc='upper left', prop={'size': 9})
plt.grid(True, linestyle=':', color='k', alpha=0.5)
locator = mdates.DayLocator([1, 15])
formatter = mdates.ConciseDateFormatter(locator)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
plt.axis(
    xmin=np.datetime64('2021-05-09'), xmax=np.datetime64('2021-10-01'), ymin=0, ymax=100
)
plt.title("Second dose coverage by age group")
plt.ylabel("Vaccine coverage (%)")


fig12 = plt.figure(figsize=(8, 6))
for coverage, label in zip(second_dose_coverage_by_age, labels_by_age):
    smoothed_coverage = 7 * n_day_average(np.diff(coverage), 7)[7:]
    plt.plot(
        second_dose_coverage_dates[8:],
        smoothed_coverage,
        label=f"{label} ({smoothed_coverage[-1]:.1f} %/week)",
    )
plt.legend(loc='upper left', prop={'size': 9})
plt.grid(True, linestyle=':', color='k', alpha=0.5)
locator = mdates.DayLocator([1, 15])
formatter = mdates.ConciseDateFormatter(locator)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1.0))
plt.axis(
    xmin=np.datetime64('2021-05-09'), xmax=np.datetime64('2021-10-01'), ymin=0, ymax=10
)
plt.title("Second dose weekly increase by age group")
plt.ylabel("Vaccination rate (% of age group / week)")

# Update the date in the HTML
html_file = 'aus_vaccinations.html'
html_lines = Path(html_file).read_text().splitlines()
now = datetime.now(timezone('Australia/Melbourne')).strftime('%Y-%m-%d %H:%M')
for i, line in enumerate(html_lines):
    if 'Last updated' in line:
        html_lines[i] = f'    Last updated: {now} Melbourne time'
Path(html_file).write_text('\n'.join(html_lines) + '\n')

# Save some deets to a file for the auto reddit posting to use:
with open("latest_vax_stats.json", 'w') as f:
    json.dump(
        {
            'latest_cumulative_doses': latest_cumulative_doses,
            'latest_daily_doses': latest_daily_doses,
            'phase_C_date': str(PHASE_C_DATE),
            'today': str(today),
        },
        f,
        indent=4,
    )

for extension in ['png', 'svg']:
    if LONGPROJECT:
        fig1.savefig(f'cumulative_doses_longproject.{extension}')
        fig2.savefig(f'daily_doses_by_state_longproject.{extension}')
        fig6.savefig(f'projection_by_type.{extension}')
        fig7.savefig(f'projection_cumulative_by_type.{extension}')
    else:
        fig1.savefig(f'cumulative_doses.{extension}')
        fig2.savefig(f'daily_doses_by_state.{extension}')
        fig3.savefig(f'utilisation.{extension}')
        fig4.savefig(f'az_utilisation.{extension}')
        fig5.savefig(f'pfizer-moderna_utilisation.{extension}')
        fig8.savefig(f'doses_by_weekday.{extension}')
        fig9.savefig(f'coverage_by_agegroup.{extension}')
        fig10.savefig(f'coverage_rate_by_agegroup.{extension}')
        fig11.savefig(f'coverage_2nd_by_agegroup.{extension}')
        fig12.savefig(f'coverage_2nd_rate_by_agegroup.{extension}')

plt.show()
