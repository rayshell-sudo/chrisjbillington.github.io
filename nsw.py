import sys
from datetime import datetime
from pytz import timezone
from pathlib import Path

from scipy.optimize import curve_fit
from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import pandas as pd

# Our uncertainty calculations are stochastic. Make them reproducible, at least:
np.random.seed(0)

converter = mdates.ConciseDateConverter()

munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime] = converter


POP_OF_SYD = 5_312_163
POP_OF_NSW = 8.166e6


NONISOLATING = 'noniso' in sys.argv
VAX = 'vax' in sys.argv
ACCELERATED_VAX = 'accel_vax' in sys.argv
OTHERS = 'others' in sys.argv
CONCERN = 'concern' in sys.argv
LGA_IX = None
LGA = None

if not (NONISOLATING or VAX or ACCELERATED_VAX or OTHERS or CONCERN) and sys.argv[1:]:
    if len(sys.argv) == 2:
        LGA_IX = int(sys.argv[1])
    else:
        raise ValueError(sys.argv[1:])

if ACCELERATED_VAX:
    VAX = True

# Data from covidlive by date announced to public
def covidlive_data(start_date=np.datetime64('2021-06-10')):
    df = pd.read_html('https://covidlive.com.au/report/daily-source-overseas/nsw')[1]

    df = df[:200]

    if df['NET'][0] == '-':
        df = df[1:200]

    dates = np.array(
        [
            np.datetime64(datetime.strptime(date, "%d %b %y"), 'D') - 1
            for date in df['DATE']
        ]
    )
    cases = np.array(df['NET'].astype(int))
    cases = cases[dates >= start_date][::-1]
    dates = dates[dates >= start_date][::-1]

    return dates, cases


def covidlive_doses_per_100():
    df = pd.read_html("https://covidlive.com.au/report/daily-vaccinations/nsw")[1]
    doses = df['DOSES'][0]
    # We're assuming that doses are evenly distributed NSW-wide. The extent that we want
    # to assume Sydney is prioritised over regional is taken into account by the factor
    # of two in the "accelerated vaccination" scenario. In the regular scenario it's
    # even state-wide.
    return 100 * doses / POP_OF_NSW


# Data from NSW Health by test notification date
def nswhealth_data(start_date=np.datetime64('2021-06-10')):
    url = (
        "https://data.nsw.gov.au/data/dataset/"
        "c647a815-5eb7-4df6-8c88-f9c537a4f21e/"
        "resource/2f1ba0f3-8c21-4a86-acaf-444be4401a6d/"
        "download/confirmed_cases_table3_likely_source.csv"
    )
    df = pd.read_csv(url)

    LOCAL = [
        'Locally acquired - no links to known case or cluster',
        'Locally acquired - investigation ongoing',
        'Locally acquired - linked to known case or cluster',
    ]

    cases_by_date = {
        d: 0
        for d in np.arange(
            np.datetime64(df['notification_date'].min()),
            np.datetime64(df['notification_date'].max()) + 1,
        )
    }

    for _, row in df.iterrows():
        if row['likely_source_of_infection'] in LOCAL:
            cases_by_date[np.datetime64(row['notification_date'])] += 1


    dates = np.array(list(cases_by_date.keys()))
    new = np.array(list(cases_by_date.values()))

    return dates[dates >= start_date], new[dates >= start_date]

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


def nonisolating_data():
    DATA = """
        2021-06-10 0
        2021-06-11 0
        2021-06-12 0
        2021-06-13 0
        2021-06-14 0
        2021-06-15 0
        2021-06-16 0
        2021-06-17 4
        2021-06-18 1
        2021-06-19 2
        2021-06-20 1
        2021-06-21 0
        2021-06-22 2
        2021-06-23 12
        2021-06-24 3
        2021-06-25 9
        2021-06-26 12
    """

    def unpack_data(s):
        dates = []
        values = []
        for line in s.splitlines():
            if line.strip() and not line.strip().startswith('#'):
                date, value = line.strip().split(maxsplit=1)
                dates.append(np.datetime64(date) - 1)
                values.append(eval(value))
        return np.array(dates), np.array(values)

    manual_dates, manual_cases = unpack_data(DATA)

    df = pd.read_html('https://covidlive.com.au/report/daily-wild-cases/nsw')[1]

    if df['TOTAL'][0] == '-':
        df = df[1:200]

    cl_dates = np.array(
        [
            np.datetime64(datetime.strptime(date, "%d %b %y"), 'D') - 1
            for date in df['DATE']
        ]
    )
    cl_cases = np.array(df['TOTAL'].astype(int))[::-1]
    cl_dates = cl_dates[::-1]

    assert cl_dates[0] == np.datetime64('2021-06-26')

    dates = np.concatenate([manual_dates, cl_dates])
    cases = np.concatenate([manual_cases, cl_cases])
    return dates, cases


def gaussian_smoothing(data, pts):
    """gaussian smooth an array by given number of points"""
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation


def fourteen_day_average(data):
    ret = np.cumsum(data, dtype=float)
    ret[14:] = ret[14:] - ret[:-14]
    return ret / 14


def partial_derivatives(function, x, params, u_params):
    model_at_center = function(x, *params)
    partial_derivatives = []
    for i, (param, u_param) in enumerate(zip(params, u_params)):
        d_param = u_param / 1e6
        params_with_partial_differential = np.zeros(len(params))
        params_with_partial_differential[:] = params[:]
        params_with_partial_differential[i] = param + d_param
        model_at_partial_differential = function(x, *params_with_partial_differential)
        partial_derivative = (model_at_partial_differential - model_at_center) / d_param
        partial_derivatives.append(partial_derivative)
    return partial_derivatives


def model_uncertainty(function, x, params, covariance):
    u_params = [np.sqrt(abs(covariance[i, i])) for i in range(len(params))]
    derivs = partial_derivatives(function, x, params, u_params)
    squared_model_uncertainty = sum(
        derivs[i] * derivs[j] * covariance[i, j]
        for i in range(len(params))
        for j in range(len(params))
    )
    return np.sqrt(squared_model_uncertainty)

def stochastic_sir(
    initial_caseload,
    initial_cumulative_cases,
    initial_R_eff,
    tau,
    population_size,
    vaccine_immunity,
    n_days,
    n_trials=10000,
    cov_caseload_R_eff=None,
    confidence_interval=0.68,
):
    """Run n trials of a stochastic SIR model, starting from an initial caseload and
    cumulative cases, for a population of the given size, an initial observed R_eff
    (i.e. the actual observed R_eff including the effects of the current level of
    immunity), a mean generation time tau, and an array `vaccine_immunity` for the
    fraction of the population that is immune over time. Must have length n_days, or can
    be a constant. Runs n_trials separate trials for n_days each. cov_caseload_R_eff, if
    given, can be a covariance matrix representing the uncertainty in the initial
    caseload and R_eff. It will be used to randomly draw an initial caseload and R_eff
    from a multivariate Gaussian distribution each trial. Returns the median and the
    given confidence interval of daily infections, cumulative infections, and R_eff over
    time.
    """
    if not isinstance(vaccine_immunity, np.ndarray):
        vaccine_immunity = np.full(n_days, vaccine_immunity)
    # Our results dataset over all trials, will extract conficence intervals at the end.
    trials_infected_today = np.zeros((n_trials, n_days))
    trials_R_eff = np.zeros((n_trials, n_days))
    for i in range(n_trials):
        print(f"trial {i}")
        # Randomly choose an R_eff and caseload from the distribution
        if cov_caseload_R_eff is not None:
            caseload, R_eff = np.random.multivariate_normal(
                [initial_caseload, initial_R_eff], cov_caseload_R_eff
            )
        else:
            caseload, R_eff = initial_caseload, initial_R_eff
        cumulative = initial_cumulative_cases
        # First we back out an R0 from the R_eff and existing immunity. In this context,
        # R0 is the rate of spread *including* the effects of restrictions and
        # behavioural change, which are assumed constant here, but excluding immunity
        # due to vaccines or previous infection.
        R0 = R_eff / ((1 - vaccine_immunity[0]) * (1 - cumulative / population_size))
        # Initial pops in each compartment
        infectious = int(round(caseload * tau / R_eff))
        recovered = cumulative - infectious
        for j, vax_immune in enumerate(vaccine_immunity):
            # vax_immune is as fraction of the population, recovered and infectious are
            # in absolute nubmers so need to be normalised by population to get
            # susceptible fraction
            s = (1 - vax_immune) * (1 - (recovered + infectious) / population_size)
            R_eff = s * R0
            infected_today = np.random.poisson(infectious * R_eff / tau)
            recovered_today = np.random.binomial(infectious, 1 / tau)
            infectious += infected_today - recovered_today
            recovered += recovered_today
            cumulative += infected_today
            trials_infected_today[i, j] = infected_today
            trials_R_eff[i, j] = R_eff 

    trials_infected_today.sort(axis=0)
    cumulative_infected = trials_infected_today.cumsum(axis=1) + initial_cumulative_cases
    cumulative_infected.sort(axis=0)
    trials_R_eff.sort(axis=0)

    ix_median = n_trials // 2
    ix_lower = int((n_trials * (1 - confidence_interval)) // 2)
    ix_upper = n_trials - ix_lower

    daily_median = trials_infected_today[ix_median, :]
    daily_lower = trials_infected_today[ix_lower, :]
    daily_upper = trials_infected_today[ix_upper, :]

    cumulative_median = cumulative_infected[ix_median, :]
    cumulative_lower = cumulative_infected[ix_lower, :]
    cumulative_upper = cumulative_infected[ix_upper, :]

    R_eff_median = trials_R_eff[ix_median, :]
    R_eff_lower = trials_R_eff[ix_lower, :]
    R_eff_upper = trials_R_eff[ix_upper, :]

    return (
        daily_median,
        (daily_lower, daily_upper),
        cumulative_median,
        (cumulative_lower, cumulative_upper),
        R_eff_median,
        (R_eff_lower, R_eff_upper),
    )


def projected_vaccine_immune_population(t, current_doses_per_100):
    """compute projected future susceptible population, starting with the current doses
    per 100 population, and assuming a certain vaccine efficacy and rollout schedule"""
    AUG = np.datetime64('2021-08-01').astype(int) - dates[-1].astype(int)
    SEP = np.datetime64('2021-09-01').astype(int) - dates[-1].astype(int)
    OCT = np.datetime64('2021-10-01').astype(int) - dates[-1].astype(int)
    NOV = np.datetime64('2021-11-01').astype(int) - dates[-1].astype(int)

    # My national projections of doses per 100 people per day are:
    # Jul 140k per day = 0.55 %
    # Aug 165k per day = 0.66 %
    # Sep 185k per day = 0.74 %
    # Oct 230k per day = 0.92 %
    # Nov 280k per day = 1.12 %

    # NSW currently exceeding national rates by 22%, so let's go with that:
    PRIORITY_FACTOR = 1.22

    if ACCELERATED_VAX:
        # What if we give NSW double the supply, or if their rollout is prioritised such
        # that each dose reduces spread twice as much as for an average member of the
        # population?
        PRIORITY_FACTOR = 2


    doses_per_100 = np.zeros_like(t)
    doses_per_100[0] = current_doses_per_100
    for i in range(1, len(doses_per_100)):
        if i < SEP:
            doses_per_100[i] = doses_per_100[i - 1] + 0.83 * PRIORITY_FACTOR
        elif i < OCT:
            doses_per_100[i] = doses_per_100[i - 1] + 0.75 * PRIORITY_FACTOR
        elif i < NOV:
            doses_per_100[i] = doses_per_100[i - 1] + 1.03 * PRIORITY_FACTOR
        else:
            doses_per_100[i] = doses_per_100[i - 1] + 1.03 * PRIORITY_FACTOR

    doses_per_100 = np.clip(doses_per_100, 0, 85 * 2)
    immune = 0.4 * doses_per_100 / 100
    return immune

# dates, new = nswhealth_data()

# for d, n in zip(dates, new):
#     print(d, n)

# Last day is incomplete data
# dates = dates[:-1]
# new = new[:-1]

# If NSW health data not updated yet, use covidlive data:
# cl_dates, cl_new = covidlive_data(start_date=dates[-1] + 1)
# dates = np.append(dates, cl_dates)
# new = np.append(new, cl_new)

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
]

if LGA_IX is not None or OTHERS or CONCERN:
    dates, cases_by_lga = lga_data()
    # Sort LGAs in reverse order by last 14d cases
    sorted_lgas_of_concern = sorted(
        LGAs_OF_CONCERN, key=lambda k: -cases_by_lga[k][-14:].sum()
    )
    # print(sorted_lgas_of_concern)
    # for lga in sorted_lgas:
    #     print(lga, cases_by_lga[lga][-14:].sum())
if LGA_IX is not None:
    LGA = sorted_lgas_of_concern[LGA_IX]
    new = cases_by_lga[LGA]
elif OTHERS:
    # Sum over all LGAs *not* of concern
    new = sum(cases_by_lga[lga] for lga in cases_by_lga if lga not in LGAs_OF_CONCERN)
elif CONCERN:
   # Sum over all LGAs of concern
    new = sum(cases_by_lga[lga] for lga in cases_by_lga if lga in LGAs_OF_CONCERN) 
elif NONISOLATING:
    dates, new = nonisolating_data()
else:
    dates, new = covidlive_data()

# Current vaccination level:
current_doses_per_100 = covidlive_doses_per_100()

# for d, n in zip(dates, new):
#     print(d, n)

# if not NONISOLATING:
#     dates = np.append(dates, [dates[-1] + 1])
#     new = np.append(new, [98])

START_PLOT = np.datetime64('2021-06-13')
END_PLOT = np.datetime64('2022-01-01') if VAX else np.datetime64('2021-09-01')

SMOOTHING = 4
PADDING = 3 * int(round(3 * SMOOTHING))
new_padded = np.zeros(len(new) + PADDING)
new_padded[: -PADDING] = new


def exponential(x, A, k):
    return A * np.exp(k * x)


# def linear(x, A, B):
#     return A * x + B


tau = 5  # reproductive time of the virus in days

# Smoothing requires padding to give sensible results at the right edge. Compute an
# exponential fit to daily cases over the last fortnight, and pad the data with the
# fit results prior to smoothing.

FIT_PTS = min(20, len(dates[dates >= START_PLOT]))
x0 = -14
delta_x = 1
fit_x = np.arange(-FIT_PTS, 0)
fit_weights = 1 / (1 + np.exp(-(fit_x - x0) / delta_x))
pad_x = np.arange(PADDING)

def clip_params(params):
    # Clip exponential fit params to be within a reasonable range to suppress when
    # unlucky points lead us to an unrealistic exponential blowup. Modifies array
    # in-place.
    R_CLIP = 5 # Limit the exponential fits to a maximum of R=5
    params[0] = min(params[0], 2 * new[-FIT_PTS:].max() + 1)
    params[1] = min(params[1], np.log(R_CLIP ** (1 / tau)))


params, cov = curve_fit(exponential, fit_x, new[-FIT_PTS:], sigma=1 / fit_weights)
clip_params(params)
fit = exponential(pad_x, *params).clip(0.1, None)

# Linear fit for now
# params, cov = curve_fit(linear, fit_x, new[-FIT_PTS:], sigma=1 / fit_weights)
# fit = linear(pad_x, *params).clip(0.1, None)


new_padded[-PADDING:] = fit
new_smoothed = gaussian_smoothing(new_padded, SMOOTHING)[: -PADDING]
# new_smoothed = gaussian_smoothing(new, SMOOTHING)
R = (new_smoothed[1:] / new_smoothed[:-1]) ** tau

# def correct_smoothing(new_smoothed, R):
#     # Gaussian smoothing creates a consistent bias whenever there is curvature. Measure
#     # and correct for it
#     f = R ** (SMOOTHING / 5)
#     bias =  (new_smoothed[1:] * f - new_smoothed[1:] / f) / 2 - new_smoothed[1:]
#     new_smoothed[1:] -= bias
#     new_smoothed[0] -= bias[0]
#     return new_smoothed

# new_smoothed = correct_smoothing(new_smoothed, R)

N_monte_carlo = 1000
variance_R = np.zeros_like(R)
variance_new_smoothed = np.zeros_like(new_smoothed)
cov_R_new_smoothed = np.zeros_like(R)
# Monte-carlo of the above with noise to compute variance in R, new_smoothed,
# and their covariance:
u_new = np.sqrt((0.2 * new) ** 2 + new)  # sqrt(N) and 20%, added in quadrature
for i in range(N_monte_carlo):
    new_with_noise = np.random.normal(new, u_new).clip(0.1, None)
    params, cov = curve_fit(
        exponential,
        fit_x,
        new_with_noise[-FIT_PTS:],
        sigma=1 / fit_weights,
        maxfev=20000,
    )
    clip_params(params)
    scenario_params = params # np.random.multivariate_normal(params, cov)
    clip_params(scenario_params)
    fit = exponential(pad_x, *scenario_params).clip(0.1, None)

    # Linear for now:
    # params, cov = curve_fit(
    #     linear,
    #     fit_x,
    #     new_with_noise[-FIT_PTS:],
    #     sigma=1 / fit_weights,
    #     maxfev=20000,
    # )
    # scenario_params = np.random.multivariate_normal(params, cov)
    # fit = linear(pad_x, *scenario_params).clip(0.1, None)


    new_padded[:-PADDING] = new_with_noise
    new_padded[-PADDING:] = fit
    new_smoothed_noisy = gaussian_smoothing(new_padded, SMOOTHING)[:-PADDING]
    # new_smoothed_noisy = gaussian_smoothing(new_with_noise, SMOOTHING)
    variance_new_smoothed += (new_smoothed_noisy - new_smoothed) ** 2 / N_monte_carlo
    R_noisy = (new_smoothed_noisy[1:] / new_smoothed_noisy[:-1]) ** tau
    variance_R += (R_noisy - R) ** 2 / N_monte_carlo
    cov_R_new_smoothed += (
        (new_smoothed_noisy[1:] - new_smoothed[1:]) * (R_noisy - R) / N_monte_carlo
    )


# Fudge what would happen with a different R_eff:
# cov_R_new_smoothed[-1] *= 0.05 / np.sqrt(variance_R[-1])
# R[-1] = 0.75
# variance_R[-1] = 0.05**2

u_R = np.sqrt(variance_R)
R_upper = R + u_R
R_lower = R - u_R

u_new_smoothed = np.sqrt(variance_new_smoothed)
new_smoothed_upper = new_smoothed + u_new_smoothed
new_smoothed_lower = new_smoothed - u_new_smoothed

R_upper = R_upper.clip(0, 10)
R_lower = R_lower.clip(0, 10)
R = R.clip(0, None)

new_smoothed_upper = new_smoothed_upper.clip(0, None)
new_smoothed_lower = new_smoothed_lower.clip(0, None)
new_smoothed = new_smoothed.clip(0, None)


# Projection of daily case numbers:
days_projection = (END_PLOT + 200 - dates[-1]).astype(int)
t_projection = np.linspace(0, days_projection, days_projection + 1)

# Construct a covariance matrix for the latest estimate in new_smoothed and R:
cov = np.array(
    [
        [variance_new_smoothed[-1], cov_R_new_smoothed[-1]],
        [cov_R_new_smoothed[-1], variance_R[-1]],
    ]
)

if VAX:
    # Fancy stochastic SIR model
    results = stochastic_sir(
        initial_caseload=new_smoothed[-1],
        initial_cumulative_cases=new.sum(),
        initial_R_eff=R[-1],
        tau=tau,
        population_size=POP_OF_SYD,
        vaccine_immunity=projected_vaccine_immune_population(
            t_projection, current_doses_per_100
        ),
        n_days=days_projection + 1,
        n_trials=10000,
        cov_caseload_R_eff=cov,
        confidence_interval=0.68,
    )
    (
        new_projection,
        ci_new_projection,
        total_projection,
        ci_total_projection,
        R_eff_projection,
        ci_R_eff_projection,
    ) = results

    new_projection_lower, new_projection_upper = ci_new_projection
    R_eff_projection_lower, R_eff_projection_upper = ci_R_eff_projection

    total_cases = total_projection[-1]
    total_cases_lower = ci_total_projection[0][-1]
    total_cases_upper = ci_total_projection[1][-1]

    print(f"total cases: {total_cases/1000:.02f}k")
    print(
        f"1 sigma range:  {total_cases_lower/1000:.02f}k—{total_cases_upper/1000:.02f}k"
    )


else:

    # Simple model, no vaccines or community immunity
    def log_projection_model(t, A, R):
        return np.log(A * R ** (t / tau))

    new_projection = np.exp(log_projection_model(t_projection, new_smoothed[-1], R[-1]))
    log_new_projection_uncertainty = model_uncertainty(
        log_projection_model, t_projection, (new_smoothed[-1], R[-1]), cov
    )
    new_projection_upper = np.exp(
        np.log(new_projection) + log_new_projection_uncertainty
    )
    new_projection_lower = np.exp(
        np.log(new_projection) - log_new_projection_uncertainty
    )

    

# Examining whether the smoothing and uncertainty look decent
# plt.bar(dates, new)
# plt.fill_between(
#     dates,
#     new_smoothed_lower,
#     new_smoothed_upper,
#     color='orange',
#     alpha=0.5,
#     zorder=5,
#     linewidth=0,
# )
# plt.plot(dates, new_smoothed, color='orange', zorder=6)
# plt.plot(
#     dates[-1] + 24 * t_projection.astype('timedelta64[h]'),
#     new_projection,
#     color='orange',
#     zorder=6,
# )
# plt.fill_between(
#     dates[-1] + 24 * t_projection.astype('timedelta64[h]'),
#     new_projection_lower,
#     new_projection_upper,
#     color='orange',
#     alpha=0.5,
#     zorder=5,
#     linewidth=0,
# )
# params, cov = curve_fit(exponential, fit_x, new[-FIT_PTS:], sigma=1 / fit_weights)
# clip_params(params)
# fit = exponential(fit_x, *params).clip(0.1, None)

# plt.plot(dates[-1] + 1 + fit_x, fit)
# plt.grid(True)
# plt.axis(xmin=dates[0], xmax=dates[-1] + 14, ymin=0, ymax=2 * new[-1])
# plt.show()

MASKS = np.datetime64('2021-06-21')
LGA_LOCKDOWN = np.datetime64('2021-06-26')
LOCKDOWN = np.datetime64('2021-06-27')
TIGHTER_LOCKDOWN = np.datetime64('2021-07-10')
NONCRITICAL_RETAIL_CLOSED = np.datetime64('2021-07-18')
END_LOCKDOWN = np.datetime64('2021-07-31') + 28 # extended

def whiten(color, f):
    """Mix a color with white where f is how much of the original colour to keep"""
    white = np.array(mcolors.to_rgb("white"))
    return (1 - f) * white + f * np.array(mcolors.to_rgb(color))


fig1 = plt.figure(figsize=(10, 6))
ax1 = plt.axes()

fig2 = plt.figure(figsize=(10, 6))
ax3 = plt.axes()

for ax in [ax1, ax3]:

    ax.fill_betweenx(
        [-10, 10] if ax is ax1 else [0, 5000],
        [MASKS, MASKS],
        [LGA_LOCKDOWN, LGA_LOCKDOWN],
        color=whiten("yellow", 0.5),
        linewidth=0,
        label="Initial restrictions",
    )


    ax.fill_betweenx(
        [-10, 10] if ax is ax1 else [0, 5000],
        [LGA_LOCKDOWN, LGA_LOCKDOWN],
        [LOCKDOWN, LOCKDOWN],
        color=whiten("yellow", 0.5),
        edgecolor=whiten("orange", 0.5),
        linewidth=0,
        hatch="//////",
        label="East Sydney LGA lockdown",
    )
    ax.fill_betweenx(
        [-10, 10] if ax is ax1 else [0, 5000],
        [LOCKDOWN, LOCKDOWN],
        [TIGHTER_LOCKDOWN, TIGHTER_LOCKDOWN],
        color=whiten("orange", 0.5),
        linewidth=0,
        label="Greater Sydney lockdown",
    )

    ax.fill_betweenx(
        [-10, 10] if ax is ax1 else [0, 5000],
        [TIGHTER_LOCKDOWN, TIGHTER_LOCKDOWN],
        [NONCRITICAL_RETAIL_CLOSED, NONCRITICAL_RETAIL_CLOSED],
        color=whiten("orange", 0.5),
        edgecolor=whiten("red", 0.35),
        linewidth=0,
        hatch="//////",
        label="Lockdown tightened",
    )

    ax.fill_betweenx(
        [-10, 10] if ax is ax1 else [0, 5000],
        [NONCRITICAL_RETAIL_CLOSED, NONCRITICAL_RETAIL_CLOSED],
        [END_LOCKDOWN, END_LOCKDOWN],
        color=whiten("red", 0.35),
        linewidth=0,
        label="Noncritical retail closed",
    )


    for i in range(30):
        ax.fill_betweenx(
            [-10, 10] if ax is ax1 else [0, 5000],
            [END_LOCKDOWN.astype(int) + i / 3] * 2,
            [END_LOCKDOWN.astype(int) + (i + 1) / 3] * 2,
            color=whiten("red", 0.25 * (30 - i) / 30),
            linewidth=0,
            zorder=-10,
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
        # linewidth=0,
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
        # linewidth=0,
        hatch="////",
    )


ax1.axhline(1.0, color='k', linewidth=1)
for ax in [ax1, ax3]:
    ax.axis(xmin=START_PLOT, xmax=END_PLOT, ymin=0, ymax=4)
    ax.grid(True, linestyle=":", color='k', alpha=0.5)

ax1.set_ylabel(R"$R_\mathrm{eff}$")

u_R_latest = (R_upper[-1] - R_lower[-1]) / 2

R_eff_string = fR"$R_\mathrm{{eff}}={R[-1]:.02f} \pm {u_R_latest:.02f}$"

if not VAX:
    if LGA:
        region = LGA
    elif OTHERS:
        region = "New South Wales (excluding LGAs of concern)"
    elif CONCERN:
        region = "New South Wales LGAs of concern"
    else:
        region = "New South Wales"
    title_lines = [
        f"$R_\\mathrm{{eff}}$ in {region}, "
        "with Sydney restriction levels and daily cases",
        f"Latest estimate: {R_eff_string}",
    ]
    if NONISOLATING:
        title_lines[0] += ' (non-isolating cases only)'
elif ACCELERATED_VAX:
    title_lines = [
        "Projected effect of 2× accelerated/prioritised New South Wales vaccination rollout",
        f"Starting from currently estimated {R_eff_string}",
    ]
else:
    title_lines = [
        "Projected effect of New South Wales vaccination rollout",
        f"Starting from currently estimated {R_eff_string}",
    ]

for ax in [ax1, ax3]:
    ax.set_title('\n'.join(title_lines))

ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
ax3.yaxis.set_major_locator(mticker.MultipleLocator(100))
ax2 = ax1.twinx()

for ax in [ax2, ax3]:
    ax.step(dates + 1, new + 0.02, color='purple', label='Daily cases')
    ax.plot(
        dates.astype(int) + 0.5,
        new_smoothed,
        color='magenta',
        label='Daily cases (smoothed)',
    )

    ax.fill_between(
        dates.astype(int) + 0.5,
        new_smoothed_lower,
        new_smoothed_upper,
        color='magenta',
        alpha=0.3,
        linewidth=0,
        zorder=10,
        label=f'Smoothing/{"projection" if VAX else "trend"} uncertainty',
    )
    ax.plot(
        dates[-1].astype(int) + 0.5 + t_projection,
        new_projection,
        color='magenta',
        linestyle='--',
        label=f'Daily cases ({"projection" if VAX else "trend"})',
    )
    ax.fill_between(
        dates[-1].astype(int) + 0.5 + t_projection,
        new_projection_lower,
        new_projection_upper,
        color='magenta',
        alpha=0.3,
        linewidth=0,
    )

    ax.set_ylabel(
        f"Daily {'non-isolating' if NONISOLATING else 'confirmed'} cases (log scale)"
    )

ax2.set_yscale('log')
ax2.axis(ymin=1, ymax=10000)
ax3.axis(ymin=0, ymax=1000 if ACCELERATED_VAX else 800)

for fig in [fig1, fig2]:
    fig.tight_layout(pad=1.8)

handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles += handles2
labels += labels2

if VAX:
    order = [5, 7, 6, 8, 9, 10, 11, 0, 1, 2, 3, 4]
else:
    order = [5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]
ax2.legend(
    # handles,
    # labels,
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    ncol=2,
    prop={'size': 8}
)

handles, labels = ax3.get_legend_handles_labels()
order = [0, 1, 2, 8, 3, 4, 5, 6, 7]
ax3.legend(
    # handles,
    # labels,
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    ncol=2,
    prop={'size': 8}
)


ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax2.yaxis.set_minor_formatter(mticker.ScalarFormatter())
ax2.tick_params(axis='y', which='minor', labelsize='x-small')
plt.setp(ax2.get_yminorticklabels()[1::2], visible=False)
for fig, ax in [(fig1, ax1), (fig2, ax3)]:
    locator = mdates.DayLocator([1, 15] if VAX else [1, 5, 10, 15, 20, 25])
    ax.xaxis.set_major_locator(locator)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    ax.xaxis.set_major_formatter(formatter)

    axpos = ax.get_position()

    text = fig.text(
        # axpos.x0 + axpos.width - 0.01,
        # axpos.y0 + 0.02,
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
    for fig in [fig1, fig2]:
        text = fig.text(
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
    if ACCELERATED_VAX:
        suffix = '_accel_vax'
elif NONISOLATING:
    suffix = '_noniso'
elif LGA:
    suffix=f'_LGA_{LGA_IX}'
elif OTHERS:
    suffix=f'_LGA_others'
elif CONCERN:
    suffix = f'_LGA_concern'
else:
    suffix = ''

fig1.savefig(f'COVID_NSW{suffix}.svg')
fig1.savefig(f'COVID_NSW{suffix}.png', dpi=133)
ax2.set_yscale('linear')
ax2.axis(ymin=0, ymax=1600 if VAX else 800)
ax2.set_ylabel(f"Daily confirmed cases (linear scale)")
fig1.savefig(f'COVID_NSW{suffix}_linear.svg')
fig1.savefig(f'COVID_NSW{suffix}_linear.png', dpi=133)

# if VAX:
#     fig2.savefig(f'COVID_NSW{suffix}_linear.svg')
#     fig2.savefig(f'COVID_NSW{suffix}_linear.png', dpi=133)


# Update the date in the HTML
html_file = 'COVID_NSW.html'
html_lines = Path(html_file).read_text().splitlines()
now = datetime.now(timezone('Australia/Melbourne')).strftime('%Y-%m-%d-%H:%M')
for i, line in enumerate(html_lines):
    if 'Last updated' in line:
        html_lines[i] = f'    Last updated: {now} AEST'
Path(html_file).write_text('\n'.join(html_lines) + '\n')
plt.show()
