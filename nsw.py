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
    exponential_with_infection_immunity,
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


POP_OF_NSW = 8.166e6

VAX = 'vax' in sys.argv
OTHERS = 'others' in sys.argv
CONCERN = 'concern' in sys.argv
SYDNEY = 'sydney' in sys.argv
NOT_SYDNEY = 'notsydney' in sys.argv
HUNTER = 'hunter' in sys.argv
ILLAWARRA = 'illawarra' in sys.argv
WESTERN_NSW = 'wnsw' in sys.argv
LGA_IX = None
LGA = None
OLD = 'old' in sys.argv

if (
    not (VAX or OTHERS or CONCERN or SYDNEY or NOT_SYDNEY or HUNTER or ILLAWARRA or WESTERN_NSW)
    and sys.argv[1:]
):
    if (VAX and OTHERS) or (VAX and CONCERN):
        pass  # That's fine and allowed
    if len(sys.argv) == 2:
        LGA_IX = int(sys.argv[1])
    elif OLD and len(sys.argv) == 3:
        OLD_END_IX = int(sys.argv[2])
    else:
        raise ValueError(sys.argv[1:])

if OLD:
    VAX = True


# Data from NSW Health by LGA and test notification date
def lga_data(start_date=np.datetime64('2021-06-10')):
    url = (
        "https://data.nsw.gov.au/data/dataset/"
        "aefcde60-3b0c-4bc0-9af1-6fe652944ec2/"
        "resource/21304414-1ff1-4243-a5d2-f52778048b29/"
        "download/confirmed_cases_table1_location.csv"
    )
    df = pd.read_csv(url)

    LGAs = set(df['lga_name19'])
    cases_by_lga = {}
    for lga in LGAs:
        if not isinstance(lga, str):
            continue
        cases_by_date = {
            d: 0
            for d in np.arange(
                np.datetime64(df['notification_date'].min()),
                np.datetime64(df['notification_date'].max()) + 1,
            )
        }

        for _, row in df[df['lga_name19'] == lga].iterrows():
            cases_by_date[np.datetime64(row['notification_date'])] += 1

        dates = np.array(list(cases_by_date.keys()))
        new = np.array(list(cases_by_date.values()))

        new = new[dates >= start_date]
        dates = dates[dates >= start_date]

        cases_by_lga[lga.split(' (')[0]] = new

    # Last day is incomplete data, ignore it:
    dates = dates[:-1]
    cases_by_lga = {lga: cases[:-1] for lga, cases in cases_by_lga.items()}
    return dates, cases_by_lga 


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

    AUG = np.datetime64('2021-08-01').astype(int) - dates[-1].astype(int)
    SEP = np.datetime64('2021-09-01').astype(int) - dates[-1].astype(int)
    OCT = np.datetime64('2021-10-01').astype(int) - dates[-1].astype(int)
    NOV = np.datetime64('2021-11-01').astype(int) - dates[-1].astype(int)

    # History of previously projected rates, so I can remake old projections:
    if dates[-1] >= np.datetime64('2021-11-21'):
        JUL_RATE = None
        AUG_RATE = None
        SEP_RATE = None
        OCT_RATE = None
        NOV_RATE = 0.10
    elif dates[-1] >= np.datetime64('2021-10-30'):
        JUL_RATE = None
        AUG_RATE = None
        SEP_RATE = None
        OCT_RATE = 0.33
        NOV_RATE = 0.33
    elif dates[-1] >= np.datetime64('2021-10-22'):
        JUL_RATE = None
        AUG_RATE = None
        SEP_RATE = None
        OCT_RATE = 0.7
        NOV_RATE = 0.7
    elif dates[-1] >= np.datetime64('2021-08-28'):
        JUL_RATE = None
        AUG_RATE = 1.4
        SEP_RATE = 1.6
        OCT_RATE = 1.8
        NOV_RATE = 1.8
    elif dates[-1] >= np.datetime64('2021-08-16'):
        JUL_RATE = None
        AUG_RATE = 1.2
        SEP_RATE = 1.4
        OCT_RATE = 1.6
        NOV_RATE = 1.6
    elif dates[-1] >= np.datetime64('2021-08-08'):
        JUL_RATE = None
        AUG_RATE = 1.01
        SEP_RATE = 0.92
        OCT_RATE = 1.26
        NOV_RATE = 1.26
    else:
        JUL_RATE = 0.63
        AUG_RATE = 0.76
        SEP_RATE = 0.85
        OCT_RATE = 1.06
        NOV_RATE = 1.29

    doses_per_100 = np.zeros_like(t)
    doses_per_100[0] = historical_doses_per_100[-1]
    for i in range(1, len(doses_per_100)):
        if i < AUG:
            doses_per_100[i] = doses_per_100[i - 1] + JUL_RATE
        if i < SEP:
            doses_per_100[i] = doses_per_100[i - 1] + AUG_RATE
        elif i < OCT:
            doses_per_100[i] = doses_per_100[i - 1] + SEP_RATE
        elif i < NOV:
            doses_per_100[i] = doses_per_100[i - 1] + OCT_RATE
        else:
            doses_per_100[i] = doses_per_100[i - 1] + NOV_RATE

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


LGAs_OF_CONCERN = [
    'Blacktown',
    'Campbelltown',
    'Canterbury-Bankstown',
    'Cumberland',
    'Fairfield',
    'Georges River',
    'Liverpool',
    'Parramatta',
    'Penrith',
    'Bayside',
    'Strathfield',
    'Burwood',
]

HUNTER_LGAS = [
    "Cessnock",
    "Lake Macquarie",
    "Dungog",
    "Maitland",
    "Mid-Coast",
    "Muswellbrook",
    "Newcastle",
    "Port Stephens",
    "Singleton",
    "Upper Hunter Shire"
]

ILLAWARRA_LGAS = ["Wollongong", "Shoalhaven", "Shellharbour", "Kiama", "Wingecarribee"]

WESTERN_NSW_LGAS = [
    # Central West
    "Bathurst Regional",
    "Blayney",
    "Cabonne",
    "Cowra",
    "Forbes",
    "Lachlan",
    "Lithgow",
    "Mid-Western Regional",
    "Oberon",
    "Orange",
    "Parkes",
    "Weddin",
    # North Western
    "Bogan",
    "Bourke",
    "Brewarrina",
    "Cobar",
    "Coonamble",
    "Dubbo Regional",
    "Gilgandra",
    "Narromine",
    "Walgett",
    "Warren",
    "Warrumbungle Shire",
    # Far West
    "Broken Hill",
    "Central Darling",
    # "Unincorporated Far West",
]

# Source: https://lpinsw.maps.arcgis.com/apps/webappviewer/index.html?id=2a8d27c8959c407396be0a3433eb4a58
GREATER_SYDNEY_LGAS = {
    "Hawkesbury",
    "Central Coast",
    "The Hills Shire",
    "Hornsby",
    "Northern Beaches",
    "Blue Mountains",
    "Penrith",
    "Blacktown",
    "Parramatta",
    "Ryde",
    "Willoughby",
    "Lane Cove",
    "Hunters Hill",
    "North Sydney",
    "Mosman",
    "Wollondilly",
    "Liverpool",
    "Fairfield",
    "Cumberland",
    "Strathfield",
    "Burwood",
    "Canada Bay",
    "Inner West",
    "Sydney",
    "Woollahra",
    "Waverley",
    "Camden",
    "Campbelltown",
    "Canterbury-Bankstown",
    "Georges River",
    "Bayside",
    "Randwick",
    "Sutherland Shire",
    "Wollongong",
    "Shellharbour"
}

dates, cases_by_lga = lga_data()

if LGA_IX is not None or OTHERS or CONCERN or SYDNEY or NOT_SYDNEY or HUNTER or ILLAWARRA or WESTERN_NSW:
    dates, cases_by_lga = lga_data()
    # Sort LGAs in reverse order by last 14d cases
    sorted_lgas_of_concern = sorted(
        LGAs_OF_CONCERN, key=lambda k: -cases_by_lga[k][-14:].sum()
    )
    # print(sorted_lgas_of_concern)
    # for lga in sorted_lgas:
    #     print(lga, cases_by_lga[lga][-14:].sum())

    # Quick-and-dirty check - these LGAs are exluded from the Western NSW list above to
    # ensure we don't crash because they haven't had any COVID cases - but once they do
    # have cases we want to include them. But I can't be sure in advance how they will
    # be named in the dataset - e.g. two may or may not have "Shire" in the name.
    for lga in cases_by_lga:
        if "Far West" in lga:
            WESTERN_NSW_LGAS.append(lga)
            print(f"There are cases in {lga} now, add it to the list!")
if LGA_IX is not None:
    LGA = sorted_lgas_of_concern[LGA_IX]
    new = cases_by_lga[LGA]
elif OTHERS:
    # Sum over all LGAs *not* of concern
    new = sum(cases_by_lga[lga] for lga in cases_by_lga if lga not in LGAs_OF_CONCERN)
elif CONCERN:
    # Sum over all LGAs of concern
    new = sum(cases_by_lga[lga] for lga in cases_by_lga if lga in LGAs_OF_CONCERN) 
elif SYDNEY:
    # Sum over all LGAs in Greater Sydney
    new = sum(cases_by_lga[lga] for lga in cases_by_lga if lga in GREATER_SYDNEY_LGAS) 
elif NOT_SYDNEY:
    # Sum over all LGAs *not* in Greater Sydney
    new = sum(cases_by_lga[lga] for lga in cases_by_lga if lga not in GREATER_SYDNEY_LGAS) 
elif HUNTER:
    new = sum(cases_by_lga[lga] for lga in cases_by_lga if lga in HUNTER_LGAS) 
elif ILLAWARRA:
    new = sum(cases_by_lga[lga] for lga in cases_by_lga if lga in ILLAWARRA_LGAS) 
elif WESTERN_NSW:
    new = sum(cases_by_lga[lga] for lga in cases_by_lga if lga in WESTERN_NSW_LGAS) 
else:
    dates, new = covidlive_case_data('NSW', start_date=np.datetime64('2021-06-10'))

if dates[-1] >= np.datetime64('2022-01-09'):
    TEST_DETECTION_RATE = 0.27
else:
    TEST_DETECTION_RATE = 0.2

START_VAX_PROJECTIONS = 42  # July 22nd, when I started making vaccine projections
all_dates = dates
all_new = new

doses_per_100 = covidlive_doses_per_100(
    n=len(dates),
    state='NSW',
    population=POP_OF_NSW,
)

if OLD:
    dates = dates[:START_VAX_PROJECTIONS + OLD_END_IX]
    new = new[:START_VAX_PROJECTIONS + OLD_END_IX]
    doses_per_100 = doses_per_100[:START_VAX_PROJECTIONS + OLD_END_IX]

START_PLOT = np.datetime64('2021-06-13')
END_PLOT = np.datetime64('2022-05-01') if VAX else dates[-1] + 28

tau = 5  # reproductive time of the virus in days
R_clip = 50

immune = projected_vaccine_immune_population(np.arange(100), doses_per_100)
s = 1 - immune
dk_dt = 1 / tau * (s[1] / s[0] - 1)

# Keep the old methodology for old plots:
if dates[-1] >= np.datetime64('2022-01-04'):
    padding_model = lambda x, A, k: exponential_with_infection_immunity(
        x,
        A,
        k,
        cumulative_cases=new.sum(),
        tau=tau,
        effective_population=TEST_DETECTION_RATE * POP_OF_NSW,
    )
elif dates[-1] >= np.datetime64('2021-10-27'):
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
        population=POP_OF_NSW,
        test_detection_rate=TEST_DETECTION_RATE,
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


MASKS = np.datetime64('2021-06-23')
LGA_LOCKDOWN = np.datetime64('2021-06-26')
LOCKDOWN = np.datetime64('2021-06-27')
TIGHTER_LOCKDOWN = np.datetime64('2021-07-10')
NONCRITICAL_RETAIL_CLOSED = np.datetime64('2021-07-18')
STATEWIDE = np.datetime64('2021-08-15')
CURFEW = np.datetime64('2021-08-23')
END_CURFEW = np.datetime64('2021-09-16')
END_LOCKDOWN = np.datetime64('2021-10-11')
EASING_80 = np.datetime64('2021-10-18')
END_MASKS = np.datetime64('2021-12-15')
MASKS_AGAIN = np.datetime64('2021-12-24')
DENSITY_LIMITS = np.datetime64('2021-12-27')


fig1 = plt.figure(figsize=(10, 6))
ax1 = plt.axes()

ax1.fill_betweenx(
    [-10, 10],
    [MASKS, MASKS],
    [LGA_LOCKDOWN, LGA_LOCKDOWN],
    color=whiten("yellow", 0.5),
    linewidth=0,
    label="Density limits/70% easing",
)

ax1.fill_betweenx(
    [-10, 10],
    [LGA_LOCKDOWN, LGA_LOCKDOWN],
    [LOCKDOWN, LOCKDOWN],
    color=whiten("yellow", 0.5),
    edgecolor=whiten("orange", 0.5),
    linewidth=0,
    hatch="//////",
    label="East Sydney LGA lockdown",
)

ax1.fill_betweenx(
    [-10, 10],
    [LOCKDOWN, LOCKDOWN],
    [TIGHTER_LOCKDOWN, TIGHTER_LOCKDOWN],
    color=whiten("orange", 0.5),
    linewidth=0,
    label="Greater Sydney lockdown",
)

ax1.fill_betweenx(
    [-10, 10],
    [TIGHTER_LOCKDOWN, TIGHTER_LOCKDOWN],
    [NONCRITICAL_RETAIL_CLOSED, NONCRITICAL_RETAIL_CLOSED],
    color=whiten("orange", 0.5),
    edgecolor=whiten("red", 0.35),
    linewidth=0,
    hatch="//////",
    label="Lockdown tightened",
)

ax1.fill_betweenx(
    [-10, 10],
    [NONCRITICAL_RETAIL_CLOSED, NONCRITICAL_RETAIL_CLOSED],
    [STATEWIDE, STATEWIDE],
    color=whiten("red", 0.35),
    linewidth=0,
    label="Noncritical retail closed",
)

ax1.fill_betweenx(
    [-10, 10],
    [STATEWIDE, STATEWIDE],
    [CURFEW, CURFEW],
    color=whiten("red", 0.35),
    edgecolor=whiten("red", 0.45),
    hatch="//////",
    linewidth=0,
    label="Regional lockdowns",
)

ax1.fill_betweenx(
    [-10, 10],
    [CURFEW, CURFEW],
    [END_CURFEW, END_CURFEW],
    color="red",
    alpha=0.45,
    linewidth=0,
    label="LGA curfew",
)

ax1.fill_betweenx(
    [-10, 10],
    [END_CURFEW, END_CURFEW],
    [END_LOCKDOWN, END_LOCKDOWN],
    color=whiten("red", 0.35),
    edgecolor=whiten("red", 0.45),
    hatch="//////",
    linewidth=0,
)

ax1.fill_betweenx(
    [-10, 10],
    [END_LOCKDOWN, END_LOCKDOWN],
    [EASING_80, EASING_80],
    color=whiten("yellow", 0.5),
    linewidth=0,
)

ax1.fill_betweenx(
    [-10, 10],
    [EASING_80, EASING_80],
    [END_MASKS, END_MASKS],
    color=whiten("green", 0.5),
    linewidth=0,
    label="80% easing/mask mandate",
)

ax1.fill_betweenx(
    [-10, 10],
    [END_MASKS, END_MASKS],
    [MASKS_AGAIN, MASKS_AGAIN],
    color=whiten("green", 0.25),
    linewidth=0,
    label="End mandatory masks",
)

ax1.fill_betweenx(
    [-10, 10],
    [MASKS_AGAIN, MASKS_AGAIN],
    [DENSITY_LIMITS, DENSITY_LIMITS],
    color=whiten("green", 0.5),
    linewidth=0,
)
ax1.fill_betweenx(
    [-10, 10],
    [DENSITY_LIMITS, DENSITY_LIMITS],
    [END_PLOT, END_PLOT],
    color=whiten("yellow", 0.5),
    linewidth=0,
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
ax1.axis(xmin=START_PLOT, xmax=END_PLOT, ymin=0, ymax=5)
ax1.grid(True, linestyle=":", color='k', alpha=0.5)

ax1.set_ylabel(R"$R_\mathrm{eff}$")

u_R_latest = (R_upper[-1] - R_lower[-1]) / 2

R_eff_string = fR"$R_\mathrm{{eff}}={R[-1]:.02f} \pm {u_R_latest:.02f}$"

latest_update_day = datetime.fromisoformat(str(dates[-1] + 1))
latest_update_day = f'{latest_update_day.strftime("%B")} {th(latest_update_day.day)}'

if VAX:
    if OTHERS:
        region = "New South Wales (excluding LGAs of concern)"
    elif CONCERN:
        region = "New South Wales LGAs of concern"
    elif SYDNEY:
        region = "Greater Sydney"
    elif NOT_SYDNEY:
        region = "New South Wales (excluding Greater Sydney)"
    elif HUNTER:
        region = "the Hunter region"
    elif ILLAWARRA:
        region = "the Illawarra region"
    elif WESTERN_NSW:
        region = "Western New South Wales"
    else:
        region = "New South Wales"
    title_lines = [
        f"SIR model of {region} as of {latest_update_day}",
        f"Starting from currently estimated {R_eff_string}",
    ]
else:
    if LGA:
        region = LGA
    elif OTHERS:
        region = "New South Wales (excluding LGAs of concern)"
    elif CONCERN:
        region = "New South Wales LGAs of concern"
    elif SYDNEY:
        region = "Greater Sydney"
    elif NOT_SYDNEY:
        region = "New South Wales (excluding Greater Sydney)"
    elif HUNTER:
        region = "the Hunter region"
    elif ILLAWARRA:
        region = "the Illawarra region"
    elif WESTERN_NSW:
        region = "Western New South Wales"
    else:
        region = "New South Wales"
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
    label=f'Daily cases ({"SIR projection" if VAX else "exponential trend"})',
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
ax2.axis(ymin=1, ymax=100_000)
fig1.tight_layout(pad=1.8)

handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles += handles2
labels += labels2

if VAX:
    order = [9, 11, 10, 12, 13, 15, 14, 8, 7, 0, 1, 2, 3, 4, 5, 6]
else:
    order = [9, 10, 11, 12, 14, 13, 8, 7, 0, 1, 2, 3, 4, 5, 6]
ax2.legend(
    # handles,
    # labels,
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    ncol=2 if VAX else 1,
    prop={'size': 8}
)


ax2.yaxis.set_major_formatter(mticker.EngFormatter())
ax2.yaxis.set_minor_formatter(mticker.EngFormatter())
ax2.tick_params(axis='y', which='minor', labelsize='x-small')
plt.setp(ax2.get_yminorticklabels()[1::2], visible=False)
locator = mdates.DayLocator([1, 15])
ax1.xaxis.set_major_locator(locator)
formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
ax1.xaxis.set_major_formatter(formatter)

ax2.tick_params(axis='y', colors='purple', which='both')
ax1.spines['right'].set_color('purple')
ax2.spines['right'].set_color('purple')
ax2.yaxis.label.set_color('purple')

ax1.tick_params(axis='y', colors='C0', which='both')
ax1.spines['left'].set_color('C0')
ax2.spines['left'].set_color('C0')
ax1.yaxis.label.set_color('C0')

axpos = ax1.get_position()

text = fig1.text(
    0.99,
    0.02,
    "@chrisbilbo | chrisbillington.net/COVID_NSW",
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
        0.63,
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
    if OTHERS:
        suffix = "_others_vax"
    elif CONCERN:
        suffix = "_concern_vax"
    elif SYDNEY:
        suffix = "_sydney_vax"
    elif NOT_SYDNEY:
        suffix = "_not_sydney_vax"
    elif HUNTER:
        suffix = "_hunter_vax"
    elif ILLAWARRA:
        suffix = "_illawarra_vax"
    elif WESTERN_NSW:
        suffix = "_wnsw_vax"
    else:
        suffix = '_vax'
elif LGA:
    suffix=f'_LGA_{LGA_IX}'
elif OTHERS:
    suffix='_LGA_others'
elif CONCERN:
    suffix = '_LGA_concern'
elif SYDNEY:
    suffix = '_sydney'
elif NOT_SYDNEY:
    suffix = '_not_sydney'
elif HUNTER:
    suffix = '_hunter'
elif ILLAWARRA:
    suffix = "_illawarra"
elif WESTERN_NSW:
    suffix = '_wnsw'
else:
    suffix = ''

if OLD:
    fig1.savefig(f'nsw_animated/{OLD_END_IX:04d}.png', dpi=133)
else:
    fig1.savefig(f'COVID_NSW{suffix}.svg')
    fig1.savefig(f'COVID_NSW{suffix}.png', dpi=133)

if VAX or not (LGA or OTHERS or CONCERN or SYDNEY or NOT_SYDNEY or HUNTER or ILLAWARRA or WESTERN_NSW):
    ax2.set_yscale('linear')
    if OLD and dates[-1] < np.datetime64('2021-12-10'):
        ymax = 2_500
    elif VAX:
        ymax = 100_000
    else:
        ymax = 60_000
    ax2.axis(ymin=0, ymax=ymax)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(ymax / 10))
    ax2.yaxis.set_major_formatter(mticker.EngFormatter())
    ax2.set_ylabel("Daily confirmed cases (linear scale)")
    if OLD:
        fig1.savefig(f'nsw_animated_linear/{OLD_END_IX:04d}.png', dpi=133)
    else:
        fig1.savefig(f'COVID_NSW{suffix}_linear.svg')
        fig1.savefig(f'COVID_NSW{suffix}_linear.png', dpi=133)

# Save some deets to a file for the auto reddit posting to use:
try:
    # Add to existing file if already present
    stats = json.loads(Path("latest_nsw_stats.json").read_text())
except FileNotFoundError:
    stats = {}

if CONCERN and not VAX:
    stats['R_eff_concern'] = R[-1]
    stats['u_R_eff_concern'] = u_R_latest
    stats['new_concern'] = new_smoothed[-1]
    stats['cov_concern'] = cov.tolist()
    stats['initial_cumulative_concern'] = int(new.sum())
elif OTHERS and not VAX:
    stats['R_eff_others'] = R[-1]
    stats['u_R_eff_others'] = u_R_latest
    stats['new_others'] = new_smoothed[-1]
    stats['cov_others'] = cov.tolist()
    stats['initial_cumulative_others'] = int(new.sum())
elif SYDNEY and not VAX:
    stats['R_eff_sydney'] = R[-1]
    stats['u_R_eff_sydney'] = u_R_latest
    stats['new_sydney'] = new_smoothed[-1]
    stats['cov_sydney'] = cov.tolist()
    stats['initial_cumulative_sydney'] = int(new.sum())
elif NOT_SYDNEY and not VAX:
    stats['R_eff_not_sydney'] = R[-1]
    stats['u_R_eff_not_sydney'] = u_R_latest
    stats['new_not_sydney'] = new_smoothed[-1]
    stats['cov_not_sydney'] = cov.tolist()
    stats['initial_cumulative_not_sydney'] = int(new.sum())
elif HUNTER and not VAX:
    stats['R_eff_hunter'] = R[-1]
    stats['u_R_eff_hunter'] = u_R_latest
    stats['new_hunter'] = new_smoothed[-1]
    stats['cov_hunter'] = cov.tolist()
    stats['initial_cumulative_hunter'] = int(new.sum())
elif ILLAWARRA and not VAX:
    stats['R_eff_illawarra'] = R[-1]
    stats['u_R_eff_illawarra'] = u_R_latest
    stats['new_illawarra'] = new_smoothed[-1]
    stats['cov_illawarra'] = cov.tolist()
    stats['initial_cumulative_illawarra'] = int(new.sum())
elif WESTERN_NSW and not VAX:
    stats['R_eff_wnsw'] = R[-1]
    stats['u_R_eff_wnsw'] = u_R_latest
    stats['new_wnsw'] = new_smoothed[-1]
    stats['cov_wnsw'] = cov.tolist()
    stats['initial_cumulative_wnsw'] = int(new.sum())
elif not (LGA or CONCERN or SYDNEY or NOT_SYDNEY or HUNTER or ILLAWARRA or OTHERS):
    stats['R_eff'] = R[-1]
    stats['u_R_eff'] = u_R_latest
    stats['today'] = str(np.datetime64(datetime.now(), 'D'))

if VAX and not (
    LGA
    or OTHERS
    or CONCERN
    or SYDNEY
    or NOT_SYDNEY
    or HUNTER
    or ILLAWARRA
    or WESTERN_NSW
):
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
    Path("latest_nsw_stats.json").write_text(json.dumps(stats, indent=4))

    # Update the date in the HTML
    html_file = 'COVID_NSW.html'
    html_lines = Path(html_file).read_text().splitlines()
    now = datetime.now(timezone('Australia/Sydney')).strftime('%Y-%m-%d %H:%M')
    for i, line in enumerate(html_lines):
        if 'Last updated' in line:
            html_lines[i] = f'    Last updated: {now} Sydney time'
    Path(html_file).write_text('\n'.join(html_lines) + '\n')
    plt.show()
