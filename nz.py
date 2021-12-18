import sys
from datetime import datetime, timedelta
from pytz import timezone
from pathlib import Path
import json
import time
import io

import requests
from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import pandas as pd
import urllib

from reff_plots_common import (
    exponential,
    determine_smoothed_cases_and_Reff,
    exponential_with_vax,
    th,
    get_SIR_projection,
    get_exp_projection,
)

# Our uncertainty calculations are stochastic. Make them reproducible, at least:
np.random.seed(0)

# HTTP headers to emulate curl
curl_headers = {'user-agent': 'curl/7.64.1'}

converter = mdates.ConciseDateConverter()

munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime] = converter

POP_OF_NZ = 4.917e6

VAX = 'vax' in sys.argv
OLD = 'old' in sys.argv
AUCKLAND = 'auckland' in sys.argv
NOTAUCKLAND = 'notauckland' in sys.argv

if not VAX and not (AUCKLAND or NOTAUCKLAND) and sys.argv[1:]:
    if OLD and len(sys.argv) == 3:
        OLD_END_IX = int(sys.argv[2])
    else:
        raise ValueError(sys.argv[1:])

if OLD:
    VAX = True


# Data from MoH
def get_data():

    today = datetime.now().strftime('%Y-%m-%d')

    # My saved data of cumulative counts by date
    data = json.loads(Path('nz_cases.json').read_text())
    
    if today not in [item['date'] for item in data]:
        # Get today's data and add it to the file
        URL = f"https://www.health.govt.nz/system/files/documents/pages/covid_cases_{today}.csv"
        
        df = pd.read_csv(URL, storage_options=curl_headers)
        
        df = df[
            (df["DHB"] != "Managed Isolation & Quarantine")
            & (df["Historical"] != "Yes")
        ]

        counts = df['Report Date'].value_counts()
        dates = np.array([np.datetime64(d) for d in counts.index])
        order = dates.argsort()
        dates = dates[order]
        counts = np.array(counts)[order]
        cumulative = counts[dates > np.datetime64('2021-08-16')].sum()
        data.append({'date': today, 'cumulative_cases': int(cumulative)})
        Path('nz_cases.json').write_text(json.dumps(data, indent=4))

    df = pd.DataFrame(data)
    dates = np.array(df['date'], 'datetime64[D]')
    new = np.diff(df['cumulative_cases'], prepend=0)

    return dates, new


def get_todays_cases():
    url = (
        "https://www.health.govt.nz/our-work/diseases-and-conditions/"
        "covid-19-novel-coronavirus/covid-19-data-and-statistics/covid-19-current-cases"
    )

    today = datetime.now().strftime('%d %B %Y')
    updated_today_string = f'Page last updated: <span class="date">{today}</span>'
    for i in range(10):
        page = requests.get(url, headers=curl_headers).content.decode('utf8')
        if updated_today_string in page:
            break
        print(f"Got old covid-19-current-cases page, retrying ({i+1}/10)...")
        time.sleep(5)
    else:
        raise ValueError("Didn't get an up-to-date MoH page")

    df = pd.read_html(page)[6]
    MIQ = "Managed Isolation & Quarantine"
    NET = "Change in last 24 hours"
    miq_net = df[df["Location"] == MIQ][NET].sum()
    all_net = df[df["Location"] == "Total"][NET].sum()

    # This isn't global because Waitematā is spelled with an "ā" on this webpage, but
    # with an "a" in the csv data:
    AUCKLAND_DHBs = ["Auckland", "Counties Manukau", "Waitematā"]

    def clean(n):
        return int(str(n).strip("*"))

    auckland_dhbs_net = df[df["Location"].isin(AUCKLAND_DHBs)][NET]

    auckland_net = sum(clean(n) for n in auckland_dhbs_net)
    national_net = clean(all_net) - clean(miq_net)
    if AUCKLAND:
        return auckland_net
    elif NOTAUCKLAND:
        return national_net - auckland_net
    return national_net


def midnight_to_midnight_data():

    today = datetime.now().strftime('%Y-%m-%d')
    URL = f"https://www.health.govt.nz/system/files/documents/pages/covid_cases_{today}.csv"

    df = None
    for suffix in ['', '_0', '_1', '_2', '_3']:
        try:
            df = pd.read_csv(
                URL.replace('.csv', f'{suffix}.csv'), storage_options=curl_headers
            )
            break
        except urllib.error.HTTPError:
            # Try again with _<n> appended to the filename
            continue
    assert df is not None, "No csv from MoH found"

    df = df[
        (df["DHB"] != "Managed Isolation & Quarantine") & (df["Historical"] != "Yes")
    ]

    AUCKLAND_DHBs = ["Auckland", "Counties Manukau", "Waitemata"]

    in_auckland = df["DHB"].isin(AUCKLAND_DHBs)
    if AUCKLAND:
        df = df[in_auckland]
    elif NOTAUCKLAND:
        df = df[~in_auckland]

    # Deliberately excluding today, as it is incomplete data
    dates = np.arange(np.datetime64('2021-08-10'), np.datetime64(today))
    counts = df['Report Date'].value_counts()
    counts = np.array([counts[str(d)] if str(d) in counts.index else 0 for d in dates])
    return dates, counts


def moh_latest_cumulative_doses():
    """Return the most recent cumulative first and second dose numbers"""
    url = (
        "https://www.health.govt.nz/our-work/diseases-and-conditions/"
        "covid-19-novel-coronavirus/covid-19-data-and-statistics/covid-19-vaccine-data"
    )

    today = datetime.now().strftime('%d %B %Y')
    updated_today_string = f'Page last updated: <span class="date">{today}</span>'
    for i in range(10):
        page = requests.get(url, headers=curl_headers).content.decode('utf8')
        if updated_today_string in page:
            break
        print(f"Got old covid-19-vaccine-data page, retrying ({i+1}/10)...")
        time.sleep(5)
    else:
        raise ValueError("Didn't get an up-to-date covid-19-vaccine-data page")

    df = pd.read_html(page)[0]
    first, second, _, _, _ = df['Cumulative total']
    return first, second


def moh_doses_per_100(n):
    """Cumulative doses per 100 population for the past n days"""
    for i in range(1, 8):
        datestring = (datetime.now() - timedelta(days=i)).strftime("%d_%m_%Y")
        url = (
            "https://www.health.govt.nz/system/files/documents/pages/"
            f"covid_vaccinations_{datestring}_update.xlsx"
        )
        try:
            print(f"trying to get vax data for {datestring}")
            response = requests.get(url, headers=curl_headers)
            if response.ok:
                break
        except urllib.error.HTTPError:
            continue
    else:
        raise RuntimeError("No vax excel spreadsheet found")

    df = pd.read_excel(io.BytesIO(response.content), sheet_name="Date")

    dates = np.array(df['Date'], dtype='datetime64[D]')
    daily_doses = np.array(df['First doses'] + df['Second doses'])

    # Interpolate up to yesterday based on the latest cumulative number
    latest_cumulative = sum(moh_latest_cumulative_doses())
    yesterday = np.datetime64(datetime.now(), 'D') - 1
    n_days_interp = (yesterday - dates[-1]).astype(int)
    if n_days_interp > 0:
        print(f"interpolating {n_days_interp} days")
        daily_doses_interp = (latest_cumulative - daily_doses.sum()) / n_days_interp
        dates = np.append(dates, np.arange(dates[-1] + 1, yesterday + 1))
        daily_doses = np.append(
            daily_doses, [int(round(daily_doses_interp))] * n_days_interp
        )
    return 100 * daily_doses.cumsum()[-n:] / POP_OF_NZ


def projected_vaccine_immune_population(t, historical_doses_per_100):
    """compute projected future susceptible population, given an array
    historical_doses_per_100 for cumulative doses doses per 100 population prior to and
    including today (length doesn't matter, so long as it goes back longer than
    VAX_ONSET_MU plus 3 * VAX_ONSET_SIGMA), and assuming a certain vaccine efficacy and
    rollout schedule"""

    # We assume vaccine effectiveness after each dose ramps up the integral of a Gaussian
    # with the following mean and stddev in days:
    VAX_ONSET_MU = 10.5 
    VAX_ONSET_SIGMA = 3.5

    SEP = np.datetime64('2021-09-01').astype(int) - dates[-1].astype(int)
    OCT = np.datetime64('2021-10-01').astype(int) - dates[-1].astype(int)

    doses_per_100 = np.zeros_like(t)
    doses_per_100[0] = historical_doses_per_100[-1]

    # History of previously projected rates, so I can remake old projections:
    if dates[-1] >= np.datetime64('2021-11-21'):
        AUG_RATE = None
        SEP_RATE = None
        OCT_RATE = 0.25
    elif dates[-1] >= np.datetime64('2021-11-09'):
        AUG_RATE = None
        SEP_RATE = None
        OCT_RATE = 0.5
    elif dates[-1] >= np.datetime64('2021-10-26'):
        AUG_RATE = None
        SEP_RATE = None
        OCT_RATE = 0.75
    elif dates[-1] >= np.datetime64('2021-10-06'):
        AUG_RATE = None
        SEP_RATE = None
        OCT_RATE = 1.5
    else:
        AUG_RATE = 1.0
        SEP_RATE = 1.6
        OCT_RATE = 1.8

    for i in range(1, len(doses_per_100)):
        if i < SEP:
            doses_per_100[i] = doses_per_100[i - 1] + AUG_RATE
        elif i < OCT:
            doses_per_100[i] = doses_per_100[i - 1] + SEP_RATE
        else:
            doses_per_100[i] = doses_per_100[i - 1] + OCT_RATE

    if dates[-1] >= np.datetime64('2021-11-21'):
        MAX_DOSES_PER_100 = 2 * 80.0
    else:
        MAX_DOSES_PER_100 = 2 * 85.0
    doses_per_100 = np.clip(doses_per_100, 0, MAX_DOSES_PER_100)

    all_doses_per_100 = np.concatenate([historical_doses_per_100, doses_per_100])
    # The "prepend=0" makes it as if all the doses in the initial day were just
    # administered all at once, but as long as historical_doses_per_100 is long enough
    # for it to have taken full effect, it doesn't matter.
    daily = np.diff(all_doses_per_100, prepend=0)

    # convolve daily doses with a transfer function for delayed effectiveness of vaccnes
    pts = int(VAX_ONSET_MU + 3 * VAX_ONSET_SIGMA)
    x = np.arange(-pts, pts + 1, 1)
    kernel = np.exp(-((x - VAX_ONSET_MU) ** 2) / (2 * VAX_ONSET_SIGMA ** 2))
    kernel /= kernel.sum()
    convolved = convolve(daily, kernel, mode='same')

    effective_doses_per_100 = convolved.cumsum()

    immune = 0.4 * effective_doses_per_100[len(historical_doses_per_100):] / 100

    return immune


dates, new = midnight_to_midnight_data()
# Last day is out of date, we replace it with the net number of cases in the last 24
# hours as of the latest update. It might be 9am-9am instead of midnight-midnight, but
# doesn't overlap with any other 24 hour period we're using and is a representative 24
# hour period so shouldn't bias anything.
new[-1] = get_todays_cases()

START_VAX_PROJECTIONS = 23  # Sep 2nd
all_dates = dates
all_new = new

# Current vaccination level:
doses_per_100 = moh_doses_per_100(n=len(dates))

if OLD:
    dates = dates[:START_VAX_PROJECTIONS + OLD_END_IX]
    new = new[:START_VAX_PROJECTIONS + OLD_END_IX]
    doses_per_100 = doses_per_100[:START_VAX_PROJECTIONS + OLD_END_IX]

START_PLOT = np.datetime64('2021-08-16')
END_PLOT = np.datetime64('2022-03-01') if VAX else dates[-1] + 28

tau = 5  # reproductive time of the virus in days
R_clip = 50

immune = projected_vaccine_immune_population(np.arange(100), doses_per_100)
s = 1 - immune
dk_dt = 1 / tau * (s[1] / s[0] - 1)

# Keep the old methodology for old plots:
if dates[-1] >= np.datetime64('2021-10-27'):
    padding_model = lambda x, A, k: exponential_with_vax(x, A, k, dk_dt)
else:
    padding_model = exponential


# Whether or not to do a 5dma of data prior to the fit. Change of methodology as of
# 2021-11-14, so keep old methodology for remaking plots prior to then. Changed
# methodology back on 2021-12-09.
if dates[-1] > np.datetime64('2021-12-09'):
    PRE_FIT_SMOOTHING = None
elif dates[-1] > np.datetime64('2021-11-14'):
    PRE_FIT_SMOOTHING = 5
else:
    PRE_FIT_SMOOTHING = None

# Where the magic happens, estimate everything:
(
    new_smoothed,
    u_new_smoothed,
    R,
    u_R,
    R_exp,
    cov,
    cov_exp,
    shot_noise_factor,
) = determine_smoothed_cases_and_Reff(
    new,
    fit_pts=min(20, len(dates[dates >= START_PLOT])),
    pre_fit_smoothing=PRE_FIT_SMOOTHING,
    padding_model=padding_model,
    R_clip=R_clip,
    tau=tau,
)

# Fudge what would happen with a different R_eff:
# cov_R_new_smoothed[-1] *= 0.05 / np.sqrt(variance_R[-1])
# R[-1] = 0.75
# variance_R[-1] = 0.05**2

R = R.clip(0, None)
R_upper = (R + u_R).clip(0, R_clip)
R_lower = (R - u_R).clip(0, R_clip)

new_smoothed = new_smoothed.clip(0, None)
new_smoothed_upper = (new_smoothed + u_new_smoothed).clip(0, None)
new_smoothed_lower = (new_smoothed - u_new_smoothed).clip(0, None)


# Projection of daily case numbers:
days_projection = (END_PLOT - dates[-1]).astype(int)
t_projection = np.linspace(0, days_projection, days_projection + 1)

if VAX:
    # Fancy stochastic SIR model
    (
        new_projection,
        new_projection_lower,
        new_projection_upper,
        R_eff_projection,
        R_eff_projection_lower,
        R_eff_projection_upper,
        total_cases,
        total_cases_lower,
        total_cases_upper,
    ) = get_SIR_projection(
        current_caseload=new_smoothed[-1],
        cumulative_cases=new.sum(),
        R_exp=R_exp[-1],
        tau=tau,
        population=POP_OF_NZ,
        test_detection_rate=0.2,
        vaccine_immunity=projected_vaccine_immune_population(
            t_projection, doses_per_100
        ),
        n_days=days_projection + 1,
        n_trials=1000 if OLD else 10000,  # just save some time if we're animating
        cov_exp=cov_exp,
    )

else:
    # Simple model, no vaccines or community immunity
    new_projection, new_projection_lower, new_projection_upper = get_exp_projection(
        t_projection=t_projection,
        current_caseload=new_smoothed[-1],
        R_eff=R[-1],
        cov=cov,
        tau=tau,
    )

ALERT_LEVEL_1 = np.datetime64('2021-06-29')
ALERT_LEVEL_4 = np.datetime64('2021-08-17')
ALERT_LEVEL_3 = np.datetime64('2021-09-22')
STEP_1 = np.datetime64('2021-10-06')
STEP_2 = np.datetime64('2021-11-10')
TRAFFIC_LIGHT_SYSTEM = np.datetime64('2021-12-03')
# END_LOCKDOWN = all_dates[-1] + 28

def whiten(color, f):
    """Mix a color with white where f is how much of the original colour to keep"""
    white = np.array(mcolors.to_rgb("white"))
    return (1 - f) * white + f * np.array(mcolors.to_rgb(color))

fig1 = plt.figure(figsize=(10, 6))
ax1 = plt.axes()

ax1.fill_betweenx(
    [-10, 10],
    [ALERT_LEVEL_4, ALERT_LEVEL_4],
    [ALERT_LEVEL_3, ALERT_LEVEL_3],
    color=whiten("#F27824", 0.5),
    # alpha=0.45,
    linewidth=0,
    label="Alert Level 4",
)

ax1.fill_betweenx(
    [-10, 10],
    [ALERT_LEVEL_3, ALERT_LEVEL_3],
    [STEP_1, STEP_1],
    color=whiten("#F6AE2F", 0.5),
    # alpha=0.45,
    linewidth=0,
    label="Alert Level 3",
)

ax1.fill_betweenx(
    [-10, 10],
    [STEP_1, STEP_1],
    [STEP_2, STEP_2],
    color=whiten("yellow", 0.5),
    # alpha=0.45,
    linewidth=0,
    label="Level 3, Step 1",
)

ax1.fill_betweenx(
    [-10, 10],
    [STEP_2, STEP_2],
    [TRAFFIC_LIGHT_SYSTEM, TRAFFIC_LIGHT_SYSTEM],
    color=whiten("green", 0.5),
    # alpha=0.45,
    linewidth=0,
    label="Level 3, Step 2",
)
ax1.fill_betweenx(
    [-10, 10],
    [TRAFFIC_LIGHT_SYSTEM, TRAFFIC_LIGHT_SYSTEM],
    [END_PLOT, END_PLOT],
    color=whiten("green", 0.25),
    # alpha=0.45,
    linewidth=0,
    label="Traffic light system",
)

ax1.fill_between(
    dates[1:] + 1,
    R,
    label=R"$R_\mathrm{eff}$",
    step='pre',
    color='C0',
)

if VAX:
    ax1.fill_between(
        np.concatenate([dates[1:].astype(int), dates[-1].astype(int) + t_projection]) + 1,
        np.concatenate([R_lower, R_eff_projection_lower]),
        np.concatenate([R_upper, R_eff_projection_upper]),
        label=R"$R_\mathrm{eff}$/projection uncertainty",
        color='cyan',
        edgecolor='blue',
        alpha=0.2,
        step='pre',
        zorder=2,
        hatch="////",
    )
    ax1.fill_between(
        dates[-1].astype(int) + t_projection + 1,
        R_eff_projection,
        label=R"$R_\mathrm{eff}$ (projection)",
        step='pre',
        color='C0',
        linewidth=0,
        alpha=0.75
    )
else:
    ax1.fill_between(
        dates[1:] + 1,
        R_lower,
        R_upper,
        label=R"$R_\mathrm{eff}$ uncertainty",
        color='cyan',
        edgecolor='blue',
        alpha=0.2,
        step='pre',
        zorder=2,
        hatch="////",
    )


ax1.axhline(1.0, color='k', linewidth=1)
ax1.axis(xmin=START_PLOT, xmax=END_PLOT, ymin=0, ymax=4)
ax1.grid(True, linestyle=":", color='k', alpha=0.5)

ax1.set_ylabel(R"$R_\mathrm{eff}$")

u_R_latest = (R_upper[-1] - R_lower[-1]) / 2

R_eff_string = fR"$R_\mathrm{{eff}}={R[-1]:.02f} \pm {u_R_latest:.02f}$"

latest_update_day = datetime.fromisoformat(str(dates[-1] + 1))
latest_update_day = f'{latest_update_day.strftime("%B")} {th(latest_update_day.day)}'

if VAX:
    title_lines = [
        f"Projected effect of New Zealand vaccination rollout as of {latest_update_day}",
        f"Starting from currently estimated {R_eff_string}",
    ]
else:
    if AUCKLAND:
        region = "Auckland"
    elif NOTAUCKLAND:
        region = "New Zealand excluding Auckland"
    else:
        region = "New Zealand"
    title_lines = [
        f"$R_\\mathrm{{eff}}$ in {region} as of {latest_update_day}, with Auckland alert level and daily cases",
        f"Latest estimate: {R_eff_string}",
    ]
    
ax1.set_title('\n'.join(title_lines))

ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
ax2 = ax1.twinx()
if OLD:
    ax2.step(all_dates + 1, all_new + 0.02, color='purple', alpha=0.5)
ax2.step(dates + 1, new + 0.02, color='purple', label='Daily cases')
ax2.plot(
    dates.astype(int) + 0.5,
    new_smoothed,
    color='magenta',
    label='Daily cases (smoothed)',
)

ax2.fill_between(
    dates.astype(int) + 0.5,
    new_smoothed_lower,
    new_smoothed_upper,
    color='magenta',
    alpha=0.3,
    linewidth=0,
    zorder=10,
    label=f'Smoothing/{"projection" if VAX else "trend"} uncertainty',
)
ax2.plot(
    dates[-1].astype(int) + 0.5 + t_projection,
    new_projection.clip(0, 1e6),  # seen SVG rendering issues when this is big
    color='magenta',
    linestyle='--',
    label=f'Daily cases ({"projection" if VAX else "trend"})',
)
ax2.fill_between(
    dates[-1].astype(int) + 0.5 + t_projection,
    new_projection_lower.clip(0, 1e6),  # seen SVG rendering issues when this is big
    new_projection_upper.clip(0, 1e6),
    color='magenta',
    alpha=0.3,
    linewidth=0,
)

ax2.set_ylabel(f"Daily cases (log scale)")

ax2.set_yscale('log')
ax2.axis(ymin=1, ymax=10_000)
fig1.tight_layout(pad=1.8)

handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles += handles2
labels += labels2

if VAX:
    order = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]
else:
    order = [5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]
ax2.legend(
    # handles,
    # labels,
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    ncol=1 if VAX else 2,
    prop={'size': 8},
)


ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax2.yaxis.set_minor_formatter(mticker.ScalarFormatter())
ax2.tick_params(axis='y', which='minor', labelsize='x-small')
plt.setp(ax2.get_yminorticklabels()[1::2], visible=False)
locator = mdates.DayLocator([1, 15] if VAX else [1, 5, 10, 15, 20, 25])
ax1.xaxis.set_major_locator(locator)
formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
ax1.xaxis.set_major_formatter(formatter)

axpos = ax1.get_position()

text = fig1.text(
    0.99,
    0.02,
    "@chrisbilbo | chrisbillington.net/COVID_NZ",
    size=8,
    alpha=0.5,
    color=(0, 0, 0.25),
    fontfamily="monospace",
    horizontalalignment="right"
)
text.set_bbox(dict(facecolor='white', alpha=0.8, linewidth=0))

if VAX:
    total_cases_range = f"{total_cases_lower:.0f}—{total_cases_upper:.0f}"
    text = fig1.text(
        0.63,
        0.83,
        "\n".join(
            [
                f"Projected total cases in outbreak:  {total_cases:.0f}",
                f"                                  68% range:  {total_cases_range}",
            ]
        ),
        fontsize='small',
    )
    text.set_bbox(dict(facecolor='white', alpha=0.8, linewidth=0))

    suffix = '_vax'
elif AUCKLAND:
    suffix = '_auckland'
elif NOTAUCKLAND:
    suffix = '_notauckland'
else:
    suffix = ''

if OLD:
    fig1.savefig(f'nz_animated/{OLD_END_IX:04d}.png', dpi=133)
else:
    fig1.savefig(f'COVID_NZ{suffix}.svg')
    fig1.savefig(f'COVID_NZ{suffix}.png', dpi=133)
if not (AUCKLAND or NOTAUCKLAND):
    ax2.set_yscale('linear')
    maxproj = new_projection[t_projection < (END_PLOT - dates[-1]).astype(int)].max()
    ymax = 400
    # if OLD:
    #     ymax = 400
    # elif maxproj < 60:
    #     ymax = 80
    # elif maxproj < 120:
    #     ymax = 160
    # elif maxproj < 150:
    #     ymax = 200
    # elif maxproj < 300:
    #     ymax = 400
    # elif maxproj < 600:
    #     ymax = 800
    # elif maxproj < 1200:
    #     ymax = 1600
    # elif maxproj < 1800:
    #     ymax = 2400
    # elif maxproj < 2400:
    #     ymax = 3200
    # else:
    #     ymax = 4000
    ax2.axis(ymin=0, ymax=ymax)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(ymax / 8))
    ax2.set_ylabel("Daily confirmed cases (linear scale)")
    if OLD:
        fig1.savefig(f'nz_animated_linear/{OLD_END_IX:04d}.png', dpi=133)
    else:
        fig1.savefig(f'COVID_NZ{suffix}_linear.svg')
        fig1.savefig(f'COVID_NZ{suffix}_linear.png', dpi=133)

# Save some deets to a file for the auto reddit posting to use:
try:
    # Add to existing file if already present
    stats = json.loads(Path("latest_nz_stats.json").read_text())
except FileNotFoundError:
    stats = {}

if AUCKLAND:
    stats['R_eff_auckland'] = R[-1] 
    stats['u_R_eff_auckland'] = u_R_latest
elif NOTAUCKLAND:
    stats['R_eff_notauckland'] = R[-1] 
    stats['u_R_eff_notauckland'] = u_R_latest
else:
    stats['R_eff'] = R[-1] 
    stats['u_R_eff'] = u_R_latest
    stats['today'] = str(np.datetime64(datetime.now(), 'D'))

if VAX:
    # Case number predictions
    stats['projection'] = []
    # in case I ever want to get the orig projection range not expanded - like to
    # compare past projections:
    stats['SHOT_NOISE_FACTOR'] = shot_noise_factor 
    for i, cases in enumerate(new_projection):
        date = dates[-1] + i
        lower = new_projection_lower[i]
        upper = new_projection_upper[i]
        lower = lower - shot_noise_factor * np.sqrt(lower)
        upper = upper + shot_noise_factor * np.sqrt(upper)
        lower = max(lower, 0)
        stats['projection'].append(
            {'date': str(date), 'cases': cases, 'upper': upper, 'lower': lower}
        )
        if i < 8:
            print(f"{cases:.0f} {lower:.0f}—{upper:.0f}")

if not OLD:
    # Only save data if this isn't a re-run on old data
    Path("latest_nz_stats.json").write_text(json.dumps(stats, indent=4))

    # Update the date in the HTML
    html_file = 'COVID_NZ.html'
    html_lines = Path(html_file).read_text().splitlines()
    now = datetime.now(timezone('NZ')).strftime('%Y-%m-%d %H:%M')
    for i, line in enumerate(html_lines):
        if 'Last updated' in line:
            html_lines[i] = f'    Last updated: {now} NZDT'
    Path(html_file).write_text('\n'.join(html_lines) + '\n')
    plt.show()
