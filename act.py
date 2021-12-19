import sys
from datetime import datetime
from pytz import timezone
from pathlib import Path
import json

import requests
from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from reff_plots_common import (
    covidlive_case_data,
    exponential,
    determine_smoothed_cases_and_Reff,
    exponential_with_vax,
    get_SIR_projection,
    get_exp_projection,
    whiten,
    th,
)

# Our uncertainty calculations are stochastic. Make them reproducible, at least:
np.random.seed(0)

converter = mdates.ConciseDateConverter()

munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime] = converter


POP_OF_ACT = 431215 

VAX = 'vax' in sys.argv
OLD = 'old' in sys.argv


if not VAX and sys.argv[1:]:
    if len(sys.argv) == 2:
        LGA_IX = int(sys.argv[1])
    elif OLD and len(sys.argv) == 3:
        OLD_END_IX = int(sys.argv[2])
    else:
        raise ValueError(sys.argv[1:])

if OLD:
    VAX = True

def air_doses_per_100(n):
    """return ACT cumulative doses per 100 population for the last n days"""
    url = "https://vaccinedata.covid19nearme.com.au/data/air_residence.json"
    data = json.loads(requests.get(url).content)
    # Convert dates to np.datetime64
    for row in data:
        row['DATE_AS_AT'] = np.datetime64(row['DATE_AS_AT'])
    data.sort(key=lambda row: row['DATE_AS_AT'])

    dates = np.array(sorted(set([row['DATE_AS_AT'] for row in data])))

    total_doses = {d: 0 for d in dates}

    for row in data:
        if row['STATE'] != 'ACT':
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

    doses = np.array(list(total_doses.values()))

    return 100 * doses[-n:] / POP_OF_ACT


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
    if dates[-1] >= np.datetime64('2021-10-21'):
        AUG_RATE = None
        SEP_RATE = None
        OCT_RATE = 0.1
    elif dates[-1] >= np.datetime64('2021-10-30'):
        AUG_RATE = None
        SEP_RATE = None
        OCT_RATE = 0.5
    elif dates[-1] >= np.datetime64('2021-10-10'):
        AUG_RATE = None
        SEP_RATE = None
        OCT_RATE = 1.3
    else:
        AUG_RATE = 1.4
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
        MAX_DOSES_PER_100 = 2 * 84.0
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


dates, new = covidlive_case_data('ACT', start_date=np.datetime64('2021-05-10'))

START_VAX_PROJECTIONS = 111  # Aug 29
all_dates = dates
all_new = new

# Current vaccination level:
doses_per_100 = air_doses_per_100(n=len(dates))

if OLD:
    dates = dates[:START_VAX_PROJECTIONS + OLD_END_IX]
    new = new[:START_VAX_PROJECTIONS + OLD_END_IX]
    doses_per_100 = doses_per_100[:START_VAX_PROJECTIONS + OLD_END_IX]

START_PLOT = np.datetime64('2021-08-10')
END_PLOT = np.datetime64('2022-05-01') if VAX else dates[-1] + 28

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
# 2021-11-19, so keep old methodology for remaking plots prior to then. Changed
# methodology back on 2021-12-11.
if dates[-1] > np.datetime64('2021-12-10'):
    PRE_FIT_SMOOTHING = None
elif dates[-1] > np.datetime64('2021-11-18'):
    PRE_FIT_SMOOTHING = 5
else:
    PRE_FIT_SMOOTHING = None

    
# Where the magic happens, estimate everything:
(
    new_smoothed,
    u_new_smoothed,
    R,
    u_R,
    cov,
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
        R_eff=R[-1],
        tau=tau,
        population=POP_OF_ACT,
        test_detection_rate=0.2,
        vaccine_immunity=projected_vaccine_immune_population(
            t_projection, doses_per_100
        ),
        n_days=days_projection + 1,
        n_trials=1000 if OLD else 10000,  # just save some time if we're animating
        cov=cov,
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


LOCKDOWN = np.datetime64('2021-08-13')
END_LOCKDOWN = np.datetime64('2021-10-15')
FURTHER_EASING = np.datetime64('2021-10-22')
EASING_95  = np.datetime64('2021-11-12')

fig1 = plt.figure(figsize=(10, 6))
ax1 = plt.axes()


ax1.fill_betweenx(
    [-10, 10],
    [LOCKDOWN, LOCKDOWN],
    [END_LOCKDOWN, END_LOCKDOWN],
    color=whiten("red", 0.45),
    linewidth=0,
    label="Lockdown",
)

ax1.fill_betweenx(
    [-10, 10],
    [END_LOCKDOWN, END_LOCKDOWN],
    [FURTHER_EASING, FURTHER_EASING],
    color=whiten("yellow", 0.5),
    linewidth=0,
    label="Easing",
)

ax1.fill_betweenx(
    [-10, 10],
    [FURTHER_EASING, FURTHER_EASING],
    [EASING_95, EASING_95],
    color=whiten("green", 0.5),
    linewidth=0,
    label="Further easing",
)

ax1.fill_betweenx(
    [-10, 10],
    [EASING_95, EASING_95],
    [END_PLOT, END_PLOT],
    color=whiten("green", 0.25),
    linewidth=0,
    label="95% vaccinated easing",
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
        f"SIR model of the Australian Capital Territory as of {latest_update_day}",
        f"Starting from currently estimated {R_eff_string}",
    ]
else:
    region = "the Australian Capital Territory"
    title_lines = [
        f"$R_\\mathrm{{eff}}$ in {region} as of {latest_update_day}, with restriction levels and daily cases",
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

ax2.set_ylabel("Daily cases (log scale)")

ax2.set_yscale('log')
ax2.axis(ymin=1, ymax=10_000)
fig1.tight_layout(pad=1.8)

handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles += handles2
labels += labels2

if VAX:
    order = [4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3]
else:
    order = [4, 5, 6, 7, 8, 9, 0, 1, 2, 3]
ax2.legend(
    # handles,
    # labels,
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    ncol=1,
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
    "@chrisbilbo | chrisbillington.net/COVID_ACT",
    size=8,
    alpha=0.5,
    color=(0, 0, 0.25),
    fontfamily="monospace",
    horizontalalignment="right"
)
text.set_bbox(dict(facecolor='white', alpha=0.8, linewidth=0))

if VAX:
    total_cases_range = f"{total_cases_lower/1000:.1f}k—{total_cases_upper/1000:.1f}k"
    text = fig1.text(
        0.63,
        0.83,
        "\n".join(
            [
                f"Projected total cases in outbreak:  {total_cases/1000:.1f}k",
                f"                                  68% range:  {total_cases_range}",
            ]
        ),
        fontsize='small',
    )
    text.set_bbox(dict(facecolor='white', alpha=0.8, linewidth=0))

    suffix = '_vax'
else:
    suffix = ''

if OLD:
    fig1.savefig(f'act_animated/{OLD_END_IX:04d}.png', dpi=133)
else:
    fig1.savefig(f'COVID_ACT{suffix}.svg')
    fig1.savefig(f'COVID_ACT{suffix}.png', dpi=133)
if True: # Just to keep the diff with nsw.py sensible here
    ax2.set_yscale('linear')
    maxproj = new_projection[t_projection < (END_PLOT - dates[-1]).astype(int)].max()
    # if maxproj < 30:
    #     ymax=40
    if OLD:
        ymax = 80
    elif maxproj < 60:
        ymax=80
    elif maxproj < 120:
        ymax=160
    elif maxproj < 150:
        ymax=200
    elif maxproj < 300:
        ymax=400
    elif maxproj < 600:
        ymax=800
    elif maxproj < 1200:
        ymax=1600
    elif maxproj < 1800:
        ymax=2400
    elif maxproj < 2400:
        ymax=3200
    else:
        ymax=4000
    # if VAX:
    #     ymax = 40
    # else:
    #     ymax = 40
    ax2.axis(ymin=0, ymax=ymax)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(ymax / 8))
    ax2.set_ylabel("Daily confirmed cases (linear scale)")
    if OLD:
        fig1.savefig(f'act_animated_linear/{OLD_END_IX:04d}.png', dpi=133)
    else:
        fig1.savefig(f'COVID_ACT{suffix}_linear.svg')
        fig1.savefig(f'COVID_ACT{suffix}_linear.png', dpi=133)

# Save some deets to a file for the auto reddit posting to use:
try:
    # Add to existing file if already present
    stats = json.loads(Path("latest_act_stats.json").read_text())
except FileNotFoundError:
    stats = {}

if True: # keep the diff simple
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
    Path("latest_act_stats.json").write_text(json.dumps(stats, indent=4))

    # Update the date in the HTML
    html_file = 'COVID_ACT.html'
    html_lines = Path(html_file).read_text().splitlines()
    now = datetime.now(timezone('Australia/Melbourne')).strftime('%Y-%m-%d %H:%M')
    for i, line in enumerate(html_lines):
        if 'Last updated' in line:
            html_lines[i] = f'    Last updated: {now} Melbourne time'
    Path(html_file).write_text('\n'.join(html_lines) + '\n')
    plt.show()
