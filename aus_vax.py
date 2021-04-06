import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
from pathlib import Path
from pytz import timezone

converter = mdates.ConciseDateConverter()
locator = mdates.DayLocator([1])
formatter = mdates.ConciseDateFormatter(locator)

munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime] = converter

def n_day_average(data, n=14):
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


STATES = ['aus', 'nsw', 'vic', 'sa', 'wa', 'tas', 'qld', 'nt', 'act']

state = 'aus'

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


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def linear(x, a, b):
    return a * x + b

def gaussian_smoothing(data, pts):
    """gaussian smooth an array by given number of points"""
    from scipy.signal import convolve
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return convolve(data / normalisation, kernel, mode='same')


TREND = False
START_DATE = np.datetime64('2021-02-22')
PHASE_1B = np.datetime64('2021-03-22')

doses_by_state = {}
for s in STATES:
    print(f"getting data for {s}")
    df = pd.read_html(f"https://covidlive.com.au/report/daily-vaccinations/{s}")[1]
    dates = np.array(df['DATE'][::-1])
    state_doses = np.array(df['DOSES'][::-1])
    dates = np.array([np.datetime64(datetime.strptime(d, '%d %b %y'), 'D') for d in dates])
    state_doses = state_doses[dates >= START_DATE]
    dates = dates[dates >= START_DATE]
    doses_by_state[s] = state_doses

# Move back some late-announced data to the dates which it applies to:
# known_national_dailies = [
#     ['2021-04-01', 73979],
#     ['2021-04-02', 79283],
#     ['2021-04-03', 4435],
# ]
# for date, daily_doses in known_national_dailies:
#     i = np.argwhere(dates == np.datetime64(date))[0, 0]
#     doses_by_state['aus'][i] = doses_by_state['aus'][i - 1] + daily_doses

doses_by_state['fed'] = doses_by_state['aus'] - sum(
    doses_by_state[s] for s in STATES if s != 'aus'
)


doses = doses_by_state[state]

# dates = dates[:-1]
# doses = doses[:-1]

# doses = np.append(doses, [doses[-1] + 85000])
# dates = np.append(dates, [dates[-1] + 1])
# doses = np.append(doses, [doses[-1] + 85000])
# dates = np.append(dates, [dates[-1] + 1])

# doses = np.append(doses, [doses[-1] + 90000])
# dates = np.append(dates, [dates[-1] + 1])

# doses = np.append(doses, [doses[-1] + 100000])
# dates = np.append(dates, [dates[-1] + 1])

smoothed_doses = gaussian_smoothing(np.diff(doses, prepend=0), 2).cumsum()

# doses = n_day_average(np.diff(doses, prepend=0), 7).cumsum()

N_DAYS_PROJECT = 250

days = (dates - dates[0]).astype(float)
days_model = np.linspace(days[0], days[-1] + N_DAYS_PROJECT, 1000)

fig1 = plt.figure(figsize=(8, 6))

plt.fill_between(
        dates + 1,
        smoothed_doses / 1e6,
        label='Cumulative doses (smoothed)',
        step='pre',
        color='C0',
        zorder=10
    )

# plt.bar(dates, smoothed_doses / 1e6, width=1, label='Cumulative doses (smoothed)')

ax1 = plt.gca()
target = 160000 * days_model
plt.plot(
    days_model + dates[0].astype(int),
    target / 1e6,
    'k--',
    label='Target',
)

popt, pcov = curve_fit(quadratic, days, smoothed_doses)
vaccinated_model = quadratic(days_model, *popt)
u_vaccinated_model = model_uncertainty(quadratic, days_model, popt, pcov)

if TREND:
    plt.plot(
        days_model + dates[0].astype(int),
        vaccinated_model / 1e6,
        color='k',
        alpha=0.5,
        label='Quadratic trend',
    )
    plt.fill_between(
        days_model + dates[0].astype(int),
        (vaccinated_model + 2 * u_vaccinated_model) / 1e6,
        (vaccinated_model - 2 * u_vaccinated_model) / 1e6,
        alpha=0.3,
        color='k',
        # edgecolor='grey',
        linewidth=0,
    )


plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=dates[0].astype(int) + 250,
    ymin=0,
    ymax=40,
)

plt.title(f'AUS cumulative doses. Total to date: {doses[-1]/1e3:.1f}k')
plt.ylabel('Cumulative doses (millions)')
# plt.yscale('log')


fig2 = plt.figure(figsize=(8, 6))
daily_doses = np.diff(smoothed_doses, prepend=0)

plt.fill_between(
        dates + 1,
        daily_doses / 1000,
        label='Daily doses (smoothed)',
        step='pre',
        color='C0',
        zorder=10,
    )

# plt.bar(
#     dates, daily_doses / 1000, width=1.0, label='Daily doses (smoothed)'
# )
ax2 = plt.gca()
plt.title(
    f'{state.upper()} daily doses. Latest rate: {daily_doses[-1] / 1000:.1f}k per day'
)
plt.axhline(160, color='k', linestyle='--', label="Target")

if TREND:
    dt = days_model[1] - days_model[0]
    plt.plot(
        days_model + dates[0].astype(int),
        np.diff(vaccinated_model, prepend=0) / dt / 1000,
        color='k',
        label="Trend"
    )

    plt.fill_between(
        days_model + dates[0].astype(int),
        np.diff(vaccinated_model + 2 * u_vaccinated_model, prepend=0) / dt / 1000,
        np.diff(vaccinated_model - 2 * u_vaccinated_model, prepend=0) / dt / 1000,
        alpha=0.3,
        color='k',
        # edgecolor='grey',
        linewidth=0,
    )

plt.axis(
    xmin=dates[0].astype(int) + 1,
    xmax=dates[0].astype(int) + 250,
    ymin=0,
    ymax=210,
)
plt.ylabel('Daily doses (thousands)')

fig3 = plt.figure(figsize=(8, 6))

MOST_RECENT_FED_UPDATE = np.datetime64('2021-04-06')
FED_CLIP = len(dates) - 1 - np.argwhere(dates == MOST_RECENT_FED_UPDATE)[0, 0]

cumsum = np.zeros(len(dates))
colours = list(reversed([f'C{i}' for i in range(9)]))
for i, state in enumerate(['nt', 'act', 'tas', 'sa', 'wa', 'qld', 'vic', 'nsw', 'fed']):
    doses = doses_by_state[state]
    if state == 'fed' and FED_CLIP:
        smoothed_doses = gaussian_smoothing(np.diff(doses[:-FED_CLIP], prepend=0), 2).cumsum()
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
ax3 = plt.gca()

for ax in [ax1, ax2, ax3]:
    ax.fill_betweenx(
        [0, 1e9],
        2 * [START_DATE.astype(int)],
        2 * [PHASE_1B.astype(int)],
        color='red',
        alpha=0.5,
        linewidth=0,
        label='Phase 1a',
    )

    ax.fill_betweenx(
        [0, 1e9],
        2 * [PHASE_1B.astype(int)],
        2 * [dates[-1].astype(int) + 30],
        color='orange',
        alpha=0.5,
        linewidth=0,
        label='Phase 1b',
    )

    for i in range(10):
        ax.fill_betweenx(
            [0, 1e9],
            2 * [dates[-1].astype(int) + 30 + i],
            2 * [dates[-1].astype(int) + 31 + i],
            color='orange',
            alpha=0.5 * (10 - i) / 10,
            linewidth=0,
        )


# plt.gca().tick_params(axis='x', rotation=90)
# plt.axis(
#     xmin=dates[0].astype(int),
#     xmax=dates[-1].astype(int),
#     ymin=0,
#     ymax=30000,
# )


handles, labels = ax1.get_legend_handles_labels()
order = [0, 1, 4, 3, 2] if TREND else [1, 0, 2, 3]
ax1.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    ncol=2,
)

handles, labels = ax2.get_legend_handles_labels()
order = [0, 1, 4, 3, 2] if TREND else [1, 0, 2, 3]
ax2.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    ncol=2,
)

for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.get_xaxis().get_major_formatter().show_offset = False
    ax.grid(True, linestyle=":", color='k')
    # ax.tick_params(axis='x', rotation=90)


ax3.legend(loc='lower right')

# plt.figure()

# plt.plot(dates[2:], 100 * (smoothed_doses[2:] / smoothed_doses[1:-1] - 1))
# plt.grid(True, linestyle=':', color='k', alpha=0.5)
# plt.axis(ymin=0, ymax=100)
# plt.ylabel('growth rate of doses (% / day)')
# plt.gca().xaxis.set_major_locator(mdates.DayLocator([1]))
# plt.gca().get_xaxis().get_major_formatter().show_offset = False
# plt.axis(
#     xmin=dates[0].astype(int),
#     xmax=dates[0].astype(int) + 250,
# )






# Update the date in the HTML
html_file = 'aus_vaccinations.html'
html_lines = Path(html_file).read_text().splitlines()
now = datetime.now(timezone('Australia/Melbourne')).strftime('%Y-%m-%d-%H:%M')
for i, line in enumerate(html_lines):
    if 'Last updated' in line:
        html_lines[i] = f'    Last updated: {now} Melbourne time'
Path(html_file).write_text('\n'.join(html_lines) + '\n')

fig1.savefig('cumulative_doses.svg')
fig2.savefig('daily_doses.svg')
fig3.savefig('daily_doses_by_state.svg')

fig1.savefig('cumulative_doses.png')
fig2.savefig('daily_doses.png')
fig3.savefig('daily_doses_by_state.png')

plt.show()
