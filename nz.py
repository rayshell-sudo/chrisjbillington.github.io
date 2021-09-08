import sys
from datetime import datetime
from pytz import timezone
from pathlib import Path
import json
from subprocess import check_output
import io

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


POP_OF_NZ = 4.917e6
POP_OF_AUCKLAND = 1.657e6


NONISOLATING = "noniso" in sys.argv
VAX = 'vax' in sys.argv

if not (NONISOLATING or VAX) and sys.argv[1:]:
    if len(sys.argv) == 2:
        LGA_IX = int(sys.argv[1])
    else:
        raise ValueError(sys.argv[1:])

def get_data():

    today = datetime.now().strftime('%Y-%m-%d')

    # My saved data of cumulative counts by date
    data = json.loads(Path('nz_cases.json').read_text())
    
    if today not in [item['date'] for item in data]:
        # Get today's data and add it to the file
        URL = f"https://www.health.govt.nz/system/files/documents/pages/covid_cases_{today}.csv"

        # Pandas gets 403 forbidden on the URL directly - curl seems fine though.
        df = pd.read_csv(io.BytesIO(check_output(["curl", URL])))
        
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

def midnight_to_midnight_data():

    today = datetime.now().strftime('%Y-%m-%d')
    URL = f"https://www.health.govt.nz/system/files/documents/pages/covid_cases_{today}.csv"

    # Pandas gets 403 forbidden on the URL directly - curl seems fine though.
    df = pd.read_csv(io.BytesIO(check_output(["curl", URL])))
    
    df = df[
        (df["DHB"] != "Managed Isolation & Quarantine") & (df["Historical"] != "Yes")
    ]

    # Deliberately excluding today, as it is incomplete data
    dates = np.arange(np.datetime64('2021-08-10'), np.datetime64(today))
    counts = df['Report Date'].value_counts()
    counts = np.array([counts[str(d)] if str(d) in counts.index else 0 for d in dates])
    return dates, counts

def owid_doses_per_hundred(n):
    REPO_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master"
    DATA_DIR = "public/data/vaccinations"
    df = pd.read_csv(f"{REPO_URL}/{DATA_DIR}/vaccinations.csv")
    df = df[df['location']=="New Zealand"]
    doses_per_100 = np.array(df['total_vaccinations_per_hundred'])
    # Remove NaNs from the dataset, duplicate prev. day instead
    for i, val in enumerate(doses_per_100):
        if np.isnan(val):
            doses_per_100[i] = doses_per_100[i-1]
    return np.array(doses_per_100)[-n:]


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


def get_confidence_interval(data, confidence_interval=0.68, axis=0):
    """Return median (lower, upper) for a confidence interval of the data along the
    given axis"""
    n = data.shape[axis]
    ix_median = n // 2
    ix_lower = int((n * (1 - confidence_interval)) // 2)
    ix_upper = n - ix_lower
    sorted_data = np.sort(data, axis=axis)
    median = sorted_data.take(ix_median, axis=axis)
    lower = sorted_data.take(ix_lower, axis=axis)
    upper = sorted_data.take(ix_upper, axis=axis)
    return median, (lower, upper)


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
):
    """Run n trials of a stochastic SIR model, starting from an initial caseload and
    cumulative cases, for a population of the given size, an initial observed R_eff
    (i.e. the actual observed R_eff including the effects of the current level of
    immunity), a mean generation time tau, and an array `vaccine_immunity` for the
    fraction of the population that is immune over time. Must have length n_days, or can
    be a constant. Runs n_trials separate trials for n_days each. cov_caseload_R_eff, if
    given, can be a covariance matrix representing the uncertainty in the initial
    caseload and R_eff. It will be used to randomly draw an initial caseload and R_eff
    from a multivariate Gaussian distribution each trial. Returns the full dataset of
    daily infections, cumulative infections, and R_eff over time, with the first axis of
    each array being the trial number, and the second axis the day.
    """
    if not isinstance(vaccine_immunity, np.ndarray):
        vaccine_immunity = np.full(n_days, vaccine_immunity)
    # Our results dataset over all trials, will extract conficence intervals at the end.
    trials_infected_today = np.zeros((n_trials, n_days))
    trials_R_eff = np.zeros((n_trials, n_days))
    for i in range(n_trials):
        # print(f"trial {i}")
        # Randomly choose an R_eff and caseload from the distribution
        if cov_caseload_R_eff is not None:
            caseload, R_eff = np.random.multivariate_normal(
                [initial_caseload, initial_R_eff], cov_caseload_R_eff
            )
            R_eff = max(0.1, R_eff)
            caseload = max(0, caseload)
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

    cumulative_infected = trials_infected_today.cumsum(axis=1) + initial_cumulative_cases

    return trials_infected_today, cumulative_infected, trials_R_eff


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
    for i in range(1, len(doses_per_100)):
        if i < SEP:
            doses_per_100[i] = doses_per_100[i - 1] + 1.0
        elif i < OCT:
            doses_per_100[i] = doses_per_100[i - 1] + 1.6
        else:
            doses_per_100[i] = doses_per_100[i - 1] + 1.8

    doses_per_100 = np.clip(doses_per_100, 0, 85 * 2)

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


# dates, new = get_data()

dates, new = midnight_to_midnight_data()

# Current vaccination level:
doses_per_100 = owid_doses_per_hundred(n=5*len(dates))

START_PLOT = np.datetime64('2021-08-16')
END_PLOT = np.datetime64('2022-01-01') if VAX else dates[-1] + 28

SMOOTHING = 4
PADDING = 3 * int(round(3 * SMOOTHING))
new_padded = np.zeros(len(new) + PADDING)
new_padded[: -PADDING] = new


def exponential(x, A, k):
    return A * np.exp(k * x)


tau = 5  # reproductive time of the virus in days

# Smoothing requires padding to give sensible results at the right edge. Compute an
# exponential fit to daily cases over the last fortnight, and pad the data with the
# fit results prior to smoothing.

FIT_PTS = min(20, len(dates[dates >= START_PLOT]))
x0 = -10
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


new_padded[-PADDING:] = fit
new_smoothed = gaussian_smoothing(new_padded, SMOOTHING)[: -PADDING]
R = (new_smoothed[1:] / new_smoothed[:-1]) ** tau

N_monte_carlo = 1000
variance_R = np.zeros_like(R)
variance_new_smoothed = np.zeros_like(new_smoothed)
cov_R_new_smoothed = np.zeros_like(R)

# Uncertainty in new cases is whatever multiple of Poisson noise puts them on average 1
# sigma away from the smoothed new cases curve. Only use data when smoothed data > 1.0
valid = new_smoothed > 1.0
if valid.sum():
    SHOT_NOISE_FACTOR = np.sqrt(
        ((new[valid] - new_smoothed[valid]) ** 2 / new_smoothed[valid]).mean()
    )
else:
    SHOT_NOISE_FACTOR = 1.0
u_new = SHOT_NOISE_FACTOR * np.sqrt(new)

# Monte-carlo of the above with noise to compute variance in R, new_smoothed,
# and their covariance:

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
    scenario_params = np.random.multivariate_normal(params, cov)
    clip_params(scenario_params)
    fit = exponential(pad_x, *scenario_params).clip(0.1, None)

    new_padded[:-PADDING] = new_with_noise
    new_padded[-PADDING:] = fit
    new_smoothed_noisy = gaussian_smoothing(new_padded, SMOOTHING)[:-PADDING]
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
days_projection = (np.datetime64('2022-02-01') - dates[-1]).astype(int)
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
    trials_infected_today, trials_cumulative, trials_R_eff = stochastic_sir(
        initial_caseload=new_smoothed[-1],
        initial_cumulative_cases=new.sum(),
        initial_R_eff=R[-1],
        tau=tau,
        population_size=POP_OF_AUCKLAND,
        vaccine_immunity=projected_vaccine_immune_population(
            t_projection, doses_per_100
        ),
        n_days=days_projection + 1,
        n_trials=10000,
        cov_caseload_R_eff=cov,
    )

    new_projection, (
        new_projection_lower,
        new_projection_upper,
    ) = get_confidence_interval(trials_infected_today)

    cumulative_median, (cumulative_lower, cumulative_upper) = get_confidence_interval(
        trials_cumulative,
    )

    R_eff_projection, (
        R_eff_projection_lower,
        R_eff_projection_upper,
    ) = get_confidence_interval(trials_R_eff)

    total_cases = cumulative_median[-1]
    total_cases_lower = cumulative_lower[-1]
    total_cases_upper = cumulative_upper[-1]

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


ALERT_LEVEL_1 = np.datetime64('2021-06-29')
ALERT_LEVEL_4 = np.datetime64('2021-08-17')
END_LOCKDOWN = dates[-1] + 14 # Who knows?

def whiten(color, f):
    """Mix a color with white where f is how much of the original colour to keep"""
    white = np.array(mcolors.to_rgb("white"))
    return (1 - f) * white + f * np.array(mcolors.to_rgb(color))

ORANGERED = (np.array(mcolors.to_rgb("orange")) + np.array(mcolors.to_rgb("red"))) / 2

fig1 = plt.figure(figsize=(10, 6))
ax1 = plt.axes()

ax1.fill_betweenx(
    [-10, 10],
    [ALERT_LEVEL_1, ALERT_LEVEL_1],
    [ALERT_LEVEL_4, ALERT_LEVEL_4],
    color=whiten("yellow", 0.35),
    linewidth=0,
    label="Alert Level 1",
)

ax1.fill_betweenx(
    [-10, 10],
    [ALERT_LEVEL_4, ALERT_LEVEL_4],
    [END_LOCKDOWN, END_LOCKDOWN],
    color=ORANGERED,
    alpha=0.45,
    linewidth=0,
    label="Alert Level 4",
)

for i in range(30):
    ax1.fill_betweenx(
        [-10, 10],
        [END_LOCKDOWN.astype(int) + i / 3] * 2,
        [END_LOCKDOWN.astype(int) + (i + 1) / 3] * 2,
        color=ORANGERED,
        alpha=0.45 * (30 - i) / 30,
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

if VAX:
    title_lines = [
        "Projected effect of New Zealand vaccination rollout",
        f"Starting from currently estimated {R_eff_string}",
    ]
else:
    region = "New Zealand"
    title_lines = [
        f"$R_\\mathrm{{eff}}$ in {region}, with Auckland alert level and daily cases"
        + (" (nonisolating cases only)" if NONISOLATING else ""),
        f"Latest estimate: {R_eff_string}",
    ]
    
ax1.set_title('\n'.join(title_lines))

ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
ax2 = ax1.twinx()

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

ax2.set_ylabel(
    f"Daily {'non-isolating' if NONISOLATING else 'confirmed'} cases (log scale)"
)

ax2.set_yscale('log')
ax2.axis(ymin=1, ymax=10_000)
fig1.tight_layout(pad=1.8)

handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles += handles2
labels += labels2
if VAX:
    order = [2, 3, 4, 5, 6, 7, 8, 0, 1]
else:
    order = [2, 3, 4, 5, 6, 7, 0, 1]
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
elif NONISOLATING:
    suffix = "_noniso"
else:
    suffix = ''

fig1.savefig(f'COVID_NZ{suffix}.svg')
fig1.savefig(f'COVID_NZ{suffix}.png', dpi=133)
if True: # Just to keep the diff with nsw.py sensible here
    ax2.set_yscale('linear')
    if VAX:
        ymax = 160
    else:
        ymax = 160
    ax2.axis(ymin=0, ymax=ymax)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(ymax / 8))
    ax2.set_ylabel("Daily confirmed cases (linear scale)")
    fig1.savefig(f'COVID_NZ{suffix}_linear.svg')
    fig1.savefig(f'COVID_NZ{suffix}_linear.png', dpi=133)

# Save some deets to a file for the auto reddit posting to use:
try:
    # Add to existing file if already present
    stats = json.loads(Path("latest_nz_stats.json").read_text())
except FileNotFoundError:
    stats = {}

if NONISOLATING:
    stats['R_eff_noniso'] = R[-1] 
    stats['u_R_eff_noniso'] = u_R_latest
else:
    stats['R_eff'] = R[-1] 
    stats['u_R_eff'] = u_R_latest
    stats['today'] = str(np.datetime64(datetime.now(), 'D'))

if VAX:
    # Case number predictions
    stats['projection'] = []
    # in case I ever want to get the orig projection range not expanded - like to
    # compare past projections:
    stats['SHOT_NOISE_FACTOR'] = SHOT_NOISE_FACTOR 
    for i, cases in enumerate(new_projection):
        date = dates[-1] + i
        lower = new_projection_lower[i]
        upper = new_projection_upper[i]
        lower = SHOT_NOISE_FACTOR * (lower - cases) + cases
        upper = SHOT_NOISE_FACTOR * (upper - cases) + cases
        stats['projection'].append(
            {'date': str(date), 'cases': cases, 'upper': upper, 'lower': lower}
        )
        if i < 8:
            print(f"{cases:.0f} {lower:.0f}—{upper:.0f}")

Path("latest_nz_stats.json").write_text(json.dumps(stats, indent=4))

# Update the date in the HTML
html_file = 'COVID_NZ.html'
html_lines = Path(html_file).read_text().splitlines()
now = datetime.now(timezone('NZ')).strftime('%Y-%m-%d %H:%M')
for i, line in enumerate(html_lines):
    if 'Last updated' in line:
        html_lines[i] = f'    Last updated: {now} NZST'
Path(html_file).write_text('\n'.join(html_lines) + '\n')
plt.show()

