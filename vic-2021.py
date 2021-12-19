import sys
from datetime import datetime
from pytz import timezone
from pathlib import Path
import json

from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd

from reff_plots_common import (
    covidlive_case_data,
    covidlive_doses_per_100,
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


POP_OF_VIC = 6.681e6


VAX = 'vax' in sys.argv
LGA_IX = None
LGA = None
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


# Data from DHHS by diagnosis date
def lga_data():
    url = "https://www.dhhs.vic.gov.au/ncov-covid-cases-by-lga-source-csv"

    df = pd.read_csv(url)

    LGAs = set(df['Localgovernmentarea'])
    cases_by_lga = {}
    for lga in LGAs:
        cases_by_date = {
            d: 0
            for d in np.arange(
                np.datetime64(df['diagnosis_date'].min()),
                np.datetime64(df['diagnosis_date'].max()) + 1,
            )
        }

        for _, row in df[df['Localgovernmentarea'] == lga].iterrows():
            cases_by_date[np.datetime64(row['diagnosis_date'])] += 1

        dates = np.array(list(cases_by_date.keys()))
        new = np.array(list(cases_by_date.values()))

        cases_by_lga[lga.split(' (')[0]] = new

    # Last day is incomplete data, ignore it:
    # dates = dates[:-1]
    # cases_by_lga = {lga: cases[:-1] for lga, cases in cases_by_lga.items()}
    return dates, cases_by_lga 


def statewide_data(start_date=np.datetime64('2021-05-10')):
    
    url = "https://www.dhhs.vic.gov.au/ncov-covid-cases-by-source-csv"

    df = pd.read_csv(url)

    cases_by_date = {
        d: 0
        for d in np.arange(
            np.datetime64(df['diagnosis_date'].min()),
            np.datetime64(df['diagnosis_date'].max()) + 1,
        )
    }

    for _, row in df[df['acquired'] != "Travel overseas"].iterrows():
        cases_by_date[np.datetime64(row['diagnosis_date'])] += 1

    dates = np.array(list(cases_by_date.keys()))
    cases = np.array(list(cases_by_date.values()))

    covidlive_dates, covidlive_cases = covidlive_case_data('VIC', start_date=start_date)

    # Fill in missing data from covidlive. In the mornings this will usually only be one
    # number, but sometimes when DH fail to update the official dataset for a whole day
    # it might be two
    covidlive_more_recent = covidlive_dates > dates[-1]
    if covidlive_more_recent.any():
        dates = np.append(dates, covidlive_dates[covidlive_more_recent])
        cases = np.append(cases, covidlive_cases[covidlive_more_recent])

    cases = cases[dates >= start_date]
    dates = dates[dates >= start_date]
    return dates, cases

def projected_vaccine_immune_population(t, historical_doses_per_100):
    """compute projected future susceptible population, given an array
    historical_doses_per_100 for cumulative doses doses per 100 population prior to and
    including today (length doesn't matter, so long as it goes back longer than
    VAX_ONSET_MU plus 3 * VAX_ONSET_SIGMA), and assuming a certain vaccine efficacy and
    rollout schedule"""

    # We assume vaccine effectiveness after each dose ramps up the integral of a
    # Gaussian with the following mean and stddev in days:
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
        OCT_RATE = 0.2
    elif dates[-1] >= np.datetime64('2021-11-09'):
        AUG_RATE = None
        SEP_RATE = None
        OCT_RATE = 0.5
    elif dates[-1] >= np.datetime64('2021-10-30'):
        AUG_RATE = None
        SEP_RATE = None
        OCT_RATE = 0.75
    elif dates[-1] >= np.datetime64('2021-09-12'):
        AUG_RATE = None
        SEP_RATE = 1.4
        OCT_RATE = 1.8
    else:
        AUG_RATE = 1.0
        SEP_RATE = 1.2
        OCT_RATE = 1.4


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


if LGA_IX is not None:
    dates, cases_by_lga = lga_data()
    # Sort LGAs in reverse order by last 14d cases
    sorted_lgas = sorted(
        cases_by_lga.keys(), key=lambda k: -cases_by_lga[k][-14:].sum()
    )
    # print(sorted_lgas)
    # for lga in sorted_lgas:
    #     print(lga, cases_by_lga[lga][-14:].sum())
if LGA_IX is not None:
    LGA = sorted_lgas[LGA_IX]
    new = cases_by_lga[LGA]
else:
    dates, new = statewide_data()

START_VAX_PROJECTIONS = 111  # August 29, when I started making vaccine projections
all_dates = dates
all_new = new

doses_per_100 = covidlive_doses_per_100(
    n=len(dates),
    state='VIC',
    population=POP_OF_VIC,
)

if OLD:
    dates = dates[:START_VAX_PROJECTIONS + OLD_END_IX]
    new = new[:START_VAX_PROJECTIONS + OLD_END_IX]
    doses_per_100 = doses_per_100[:START_VAX_PROJECTIONS + OLD_END_IX]


START_PLOT = np.datetime64('2021-05-20')
END_PLOT = np.datetime64('2022-05-01') if VAX else dates[-1] + 28

tau = 5  # reproductive time of the virus in days
R_clip = 50

immune = projected_vaccine_immune_population(np.arange(100), doses_per_100)
s = 1 - immune
dk_dt = 1 / tau * (s[1] / s[0] - 1)

# Keep the old methodology for old plots:
if dates[-1] >= np.datetime64('2021-10-07'):
    padding_model = lambda x, A, k: exponential_with_vax(x, A, k, dk_dt)
else:
    padding_model = exponential

# Usually fit 14, but switched to 10 temporarily, to decrease influence of the
# likely-temporary grand-final-weekend surge
if dates[-1] >= np.datetime64('2021-10-11'):
    x0 = -14
elif dates[-1] >= np.datetime64('2021-10-07'):
    x0 = -10
else:
    x0 = -14

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
    R_exp,
    cov,
    cov_exp,
    shot_noise_factor,
) = determine_smoothed_cases_and_Reff(
    new,
    fit_pts=min(20, len(dates[dates >= START_PLOT])),
    x0=x0,
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
        population=POP_OF_VIC,
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


PREV_LOCKDOWN = np.datetime64('2021-05-28')
PREV_EASING_1 = PREV_LOCKDOWN + 21
PREV_EASING_2 = np.datetime64('2021-07-09')


LOCKDOWN = np.datetime64('2021-07-16')
EASING_1 = np.datetime64('2021-07-28')

LOCKDOWN_AGAIN = np.datetime64('2021-08-06')
CURFEW = np.datetime64('2021-08-16')
CONSTRUCTION_SHUTDOWN = np.datetime64('2021-09-21')
END_CONSTRUCTION_SHUTDOWN = CONSTRUCTION_SHUTDOWN + 14
PHASE_B = np.datetime64('2021-10-22')
PHASE_C = np.datetime64('2021-10-30')
PHASE_D = np.datetime64('2021-11-19')

fig1 = plt.figure(figsize=(10, 6))
ax1 = plt.axes()

ax1.fill_betweenx(
    [-10, 10],
    [PREV_LOCKDOWN, PREV_LOCKDOWN],
    [PREV_EASING_1, PREV_EASING_1],
    color=whiten("red", 0.35),
    linewidth=0,
    label="Lockdown",
)

ax1.fill_betweenx(
    [-10, 10],
    [PREV_EASING_1, PREV_EASING_1],
    [PREV_EASING_2, PREV_EASING_2],
    color=whiten("orange", 0.5),
    linewidth=0,
    label="Eased stay-at-home orders",
)

ax1.fill_betweenx(
    [-10, 10],
    [PREV_EASING_2, PREV_EASING_2],
    [LOCKDOWN, LOCKDOWN],
    color=whiten("yellow", 0.5),
    linewidth=0,
    label="Eased gathering restrictions/Phase B",
)

ax1.fill_betweenx(
    [-10, 10],
    [LOCKDOWN, LOCKDOWN],
    [EASING_1, EASING_1],
    color=whiten("red", 0.35),
    linewidth=0,
    # label="Lockdown",
)

ax1.fill_betweenx(
    [-10, 10],
    [EASING_1, EASING_1],
    [LOCKDOWN_AGAIN, LOCKDOWN_AGAIN],
    color=whiten("orange", 0.5),
    linewidth=0,
    # label="Eased stay-at-home orders",
)

ax1.fill_betweenx(
    [-10, 10],
    [LOCKDOWN_AGAIN, LOCKDOWN_AGAIN],
    [CURFEW, CURFEW],
    color=whiten("red", 0.35),
    linewidth=0,
    # label="Eased stay-at-home orders",
)

ax1.fill_betweenx(
    [-10, 10],
    [CURFEW, CURFEW],
    [CONSTRUCTION_SHUTDOWN, CONSTRUCTION_SHUTDOWN],
    color=whiten("red", 0.35),
    edgecolor=whiten("red", 0.45),
    hatch="//////",
    linewidth=0,
    label="Curfew",
)

ax1.fill_betweenx(
    [-10, 10],
    [CONSTRUCTION_SHUTDOWN, CONSTRUCTION_SHUTDOWN],
    [END_CONSTRUCTION_SHUTDOWN, END_CONSTRUCTION_SHUTDOWN],
    color="red",
    alpha=0.45,
    linewidth=0,
    label="Construction shutdown",
)

ax1.fill_betweenx(
    [-10, 10],
    [END_CONSTRUCTION_SHUTDOWN, END_CONSTRUCTION_SHUTDOWN],
    [PHASE_B, PHASE_B],
    color=whiten("red", 0.35),
    edgecolor=whiten("red", 0.45),
    hatch="//////",
    linewidth=0,
    # label="Curfew",
)

ax1.fill_betweenx(
    [-10, 10],
    [PHASE_B, PHASE_B],
    [PHASE_C, PHASE_C],
    color=whiten("yellow", 0.5),
    linewidth=0,
)

ax1.fill_betweenx(
    [-10, 10],
    [PHASE_C, PHASE_C],
    [PHASE_D, PHASE_D],
    color=whiten("green", 0.5),
    label="Phase C",
    linewidth=0,
)

ax1.fill_betweenx(
    [-10, 10],
    [PHASE_D, PHASE_D],
    [END_PLOT, END_PLOT],
    color=whiten("green", 0.25),
    linewidth=0,
    label="Phase D",
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
        f"SIR model of Victoria as of {latest_update_day}",
        f"Starting from currently estimated {R_eff_string}",
    ]
else:
    if LGA:
        region = LGA
    else:
        region = "Victoria"
    title_lines = [
        f"$R_\\mathrm{{eff}}$ in {region} as of {latest_update_day}, with Melbourne restriction levels and daily cases",
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
    order = [7, 9, 8, 10, 11, 12, 13, 6, 5, 2, 1, 0, 3, 4]
else:
    order = [7, 8, 9, 10, 11, 12, 6, 5, 2, 1, 0, 3, 4]
ax2.legend(
    # handles,
    # labels,
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    ncol=2,
    prop={'size': 8},
)


ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax2.yaxis.set_minor_formatter(mticker.ScalarFormatter())
ax2.tick_params(axis='y', which='minor', labelsize='x-small')
plt.setp(ax2.get_yminorticklabels()[1::2], visible=False)
locator = mdates.DayLocator([1, 15])
ax1.xaxis.set_major_locator(locator)
formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
ax1.xaxis.set_major_formatter(formatter)

axpos = ax1.get_position()

text = fig1.text(
    0.99,
    0.02,
    "@chrisbilbo | chrisbillington.net/COVID_VIC_2021",
    size=8,
    alpha=0.5,
    color=(0, 0, 0.25),
    fontfamily="monospace",
    horizontalalignment="right"
)
text.set_bbox(dict(facecolor='white', alpha=0.8, linewidth=0))

if VAX:
    total_cases_range = f"{total_cases_lower/1000:.0f}k—{total_cases_upper/1000:.0f}k"
    text = fig1.text(
        0.65,
        0.83,
        "\n".join(
            [
                f"Projected total cases in outbreak:  {total_cases/1000:.0f}k",
                f"                                  68% range:  {total_cases_range}",
            ]
        ),
        fontsize='small',
    )
    text.set_bbox(dict(facecolor='white', alpha=0.8, linewidth=0))

    suffix = '_vax'
elif LGA:
    suffix=f'_LGA_{LGA_IX}'
else:
    suffix = ''

if OLD:
    fig1.savefig(f'vic_animated/{OLD_END_IX:04d}.png', dpi=133)
else:
    fig1.savefig(f'COVID_VIC_2021{suffix}.svg')
    fig1.savefig(f'COVID_VIC_2021{suffix}.png', dpi=133)
if not LGA:
    ax2.set_yscale('linear')
    maxproj = new_projection[t_projection < (END_PLOT - dates[-1]).astype(int)].max()
    if OLD:
        ymax = 4000
    elif maxproj < 1800:
        ymax = 2400
    elif maxproj < 2400:
        ymax = 3200
    elif maxproj < 3000:
        ymax = 4000
    elif maxproj < 6000:
        ymax = 8000
    elif maxproj < 12000:
        ymax = 16000
    elif maxproj < 18000:
        ymax = 24000
    elif maxproj < 24000:
        ymax = 32000
    else:
        ymax = 80000

    ax2.axis(ymin=0, ymax=ymax)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(ymax / 8))
    ax2.set_ylabel("Daily confirmed cases (linear scale)")
    if OLD:
        fig1.savefig(f'vic_animated_linear/{OLD_END_IX:04d}.png', dpi=133)
    else:
        fig1.savefig(f'COVID_VIC_2021{suffix}_linear.svg')
        fig1.savefig(f'COVID_VIC_2021{suffix}_linear.png', dpi=133)

# Save some deets to a file for the auto reddit posting to use:
try:
    # Add to existing file if already present
    stats = json.loads(Path("latest_vic_stats.json").read_text())
except FileNotFoundError:
    stats = {}

if not LGA:
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
    Path("latest_vic_stats.json").write_text(json.dumps(stats, indent=4))

    # Update the date in the HTML
    html_file = 'COVID_VIC_2021.html'
    html_lines = Path(html_file).read_text().splitlines()
    now = datetime.now(timezone('Australia/Melbourne')).strftime('%Y-%m-%d %H:%M')
    for i, line in enumerate(html_lines):
        if 'Last updated' in line:
            html_lines[i] = f'    Last updated: {now} AEST'
    Path(html_file).write_text('\n'.join(html_lines) + '\n')
    plt.show()
