import json
from pathlib import Path

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

from datetime import datetime

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime] = converter

import sys

SKIP_FIGS =  'skip_figs' in sys.argv
if not SKIP_FIGS and sys.argv[1:]:
    raise ValueError(sys.argv[1:])

# ABS Estimated Resident Population, June 2020.
# Note AUS doesn't equal sum of states/territories
TOTAL_POPS = {
    'NSW': 8_164_128,
    'VIC': 6_694_884,
    'QLD': 5_174_437,
    'WA': 2_661_936,
    'SA': 1_769_319,
    'TAS': 540_569,
    'ACT': 431_114,
    'NT': 245_980,
    'AUS': 25_687_041
}

# ABS Estimated Resident Population, June 2020
# https://www.abs.gov.au/statistics/people/population/national-state-and-territory-population/jun-2020
POP_DATA = {
    '5_11': 2258250,
    '12_15': 1206862,
    '16_19': 1195636,
    '20_24': 1712736,
    '25_29': 1906551,
    '30_34': 1924355,
    '35_39': 1835579,
    '40_44': 1620761,
    '45_49': 1676720,
    '50_54': 1564167,
    '55_59': 1552740,
    '60_64': 1432211,
    '65_69': 1254318,
    '70_74': 1102983,
    '75_79': 772923,
    '80_84': 528108,
    '85_89': 316098,
    '90_94': 158412,
    '95_PLUS': 52906,
}


def datefmt(d):
    """Format a np.datetime64 as e.g."August 5th" """
    d = d.astype(datetime)
    n = d.day
    th = "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f'{d.strftime("%B").rjust(8)} {str(n).rjust(2)}{th}'

def add_national_data(data):
    """Convert national data to same format as the per-state data and add to the
    dataset"""
    AIR_JSON = "https://vaccinedata.covid19nearme.com.au/data/air.json"
    AIR_data = json.loads(requests.get(AIR_JSON).content)
    start_date = min(row['DATE_AS_AT'] for row in data)
    age_12_15_start_date = min(
        row['DATE_AS_AT'] for row in AIR_data if 'AIR_12_15_FIRST_DOSE_COUNT' in row
    )
    age_5_11_start_date = min(
        row['DATE_AS_AT'] for row in AIR_data if 'AIR_AUS_5_11_FIRST_DOSE_COUNT' in row
    )
    pop_16_plus = sum(POP_DATA.values()) - POP_DATA['12_15'] - POP_DATA['5_11']
    for row in AIR_data:
        if row['DATE_AS_AT'] < start_date:
            continue

        first_doses_16_plus = 0
        second_doses_16_plus = 0
        for agegroup in POP_DATA:
            if agegroup == '12_15' and row['DATE_AS_AT'] < age_12_15_start_date:
                continue
            if agegroup == '5_11' and row['DATE_AS_AT'] < age_5_11_start_date:
                continue
            age_lower, age_upper = agegroup.split('_')
            age_lower = int(age_lower)
            age_upper = 999 if age_upper == 'PLUS' else int(age_upper)
            if age_lower == 5:
                first_doses = row[f'AIR_AUS_{agegroup}_FIRST_DOSE_COUNT']
                second_doses = row.get(f'AIR_AUS_{agegroup}_SECOND_DOSE_COUNT', 0)
            else:
                first_doses = row[f'AIR_{agegroup}_FIRST_DOSE_COUNT']
                second_doses = row[f'AIR_{agegroup}_SECOND_DOSE_COUNT']
            if agegroup not in ('12_15', '5_11'):
                first_doses_16_plus += first_doses
                second_doses_16_plus += second_doses
            new_row = {
                'STATE': 'AUS',
                'DATE_AS_AT': row['DATE_AS_AT'],
                'AGE_LOWER': age_lower,
                'AGE_UPPER': age_upper,
                'ABS_ERP_JUN_2020_POP': POP_DATA[agegroup],
                'AIR_RESIDENCE_FIRST_DOSE_APPROX_COUNT': first_doses,
                'AIR_RESIDENCE_SECOND_DOSE_APPROX_COUNT': second_doses,
            }
            data.append(new_row)

        new_row = {
                'STATE': 'AUS',
                'DATE_AS_AT': row['DATE_AS_AT'],
                'AGE_LOWER': 16,
                'AGE_UPPER': 999,
                'ABS_ERP_JUN_2020_POP': pop_16_plus,
                'AIR_RESIDENCE_FIRST_DOSE_COUNT': first_doses_16_plus,
                'AIR_RESIDENCE_SECOND_DOSE_COUNT': second_doses_16_plus,
            }
        data.append(new_row)


url = "https://vaccinedata.covid19nearme.com.au/data/air_residence.json"
data = json.loads(requests.get(url).content)

add_national_data(data)

# Convert dates to np.datetime64
for row in data:
    row['DATE_AS_AT'] = np.datetime64(row['DATE_AS_AT'])

data.sort(key=lambda row: row['DATE_AS_AT'])

# First date when data for ages 12-15 became available:
AGES_12_15_FROM = np.datetime64('2021-09-13')
AGES_5_11_FROM = np.datetime64('2022-01-10')
data_dates = set([row['DATE_AS_AT'] for row in data])
# dates = np.array(sorted(set([row['DATE_AS_AT'] for row in data])))
dates = np.arange(min(data_dates), max(data_dates) + 1)
dates_12_15 = dates[dates >= AGES_12_15_FROM]
dates_5_11 = dates[dates >= AGES_5_11_FROM]

STATES = ['AUS', 'NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT', 'NT']
AGE_GROUPS = ['0+', '12+', '16+', 5, 12, 16, 20, 30, 40, 50, 60, 70, 80, 90][::-1]

html_table_content = {}

for state in STATES:

    html_table_content[state] = {}

    labels_by_age = []
    first_dose_coverage_by_age = []
    second_dose_coverage_by_age = []
    dates_by_age = []

    for group_start in AGE_GROUPS:
        if group_start in ['0+', '12+']:
            ranges = [(12, 15), (16, 999)]
        elif group_start == '16+':
            ranges = [(16, 999)]
        elif group_start == 5:
            ranges = [(5, 11)]
        elif group_start == 12:
            ranges = [(12, 15)]
        elif group_start == 16:
            ranges = [(16, 19)]
        elif group_start == 90:
            ranges = [(90, 94), (95, 999)]
        else:
            ranges = [
                (group_start, group_start + 4),
                (group_start + 5, group_start + 9),
            ]
        first_dose_coverage = []
        second_dose_coverage = []

        pop = 0
        if group_start == 5:
            dates_by_age.append(dates_5_11)
        elif group_start == 12:
            dates_by_age.append(dates_12_15)
        else:
            dates_by_age.append(dates)

        first_doses = np.zeros(len(dates_by_age[-1]))
        second_doses = np.zeros(len(dates_by_age[-1]))

        for lower, upper in ranges:

            if (lower, upper) == (16, 999):
                FIRST_KEY = 'AIR_RESIDENCE_FIRST_DOSE_COUNT'
                SECOND_KEY = 'AIR_RESIDENCE_SECOND_DOSE_COUNT'
            else:
                FIRST_KEY = 'AIR_RESIDENCE_FIRST_DOSE_APPROX_COUNT'
                SECOND_KEY = 'AIR_RESIDENCE_SECOND_DOSE_APPROX_COUNT'

            # Filter for the rows we want
            rows = [
                row
                for row in data
                if row['AGE_LOWER'] == lower
                and row['AGE_UPPER'] == upper
                and row['STATE'] == state
            ]

            # Duplicate rows for days with missing data:
            i = 0
            while i < len(rows) - 1:
                if rows[i + 1]['DATE_AS_AT'] != rows[i]['DATE_AS_AT'] + 1:
                    rows.insert(i + 1, rows[i].copy())
                    rows[i + 1]['DATE_AS_AT'] = rows[i]['DATE_AS_AT'] + 1
                i += 1

            pop += rows[0]['ABS_ERP_JUN_2020_POP']

            # Now one row for each date. We do this backwards and then reverse the
            # arrays afterwards, since the 12–15 age group only has data for more recent
            # dates, and we need them to line up when calculating the numbers for all
            # the 12+ population
            for i, row in enumerate(rows[::-1]):
                first_doses[i] += row[FIRST_KEY]
                second_doses[i] += row.get(SECOND_KEY, 0)

        first_doses = first_doses[::-1]
        second_doses = second_doses[::-1]

        if group_start == '0+':
            pop = TOTAL_POPS[state]

        first_dose_coverage_by_age.append(100 * first_doses / pop)
        second_dose_coverage_by_age.append(100 * second_doses / pop)

        if group_start == '16+':
            label = 'Ages 16+'
        elif group_start == '12+':
            label = 'Ages 12+'
        elif group_start == '0+':
            label = 'All ages'
        elif group_start == 90:
            label = 'Ages 90+'
        else:
            label = f'Ages {ranges[0][0]}–{ranges[-1][-1]}'
        labels_by_age.append(label)

    fig9 = plt.figure(figsize=(12, 5))
    plt.subplot(121)
    for d, coverage, label in zip(
        dates_by_age, first_dose_coverage_by_age, labels_by_age
    ):
        if label == 'Ages 16+':
            args = ['k-']
        elif label == 'Ages 12+':
            args = ['k--']
        elif label == 'All ages':
            args = ['k:']
        else:
            args = []
        if label in ['Ages 16+', 'Ages 12+', 'All ages'] or state == 'AUS':
            coverage_string = f"{coverage[-1]:.1f}"
        else:
            coverage_string = f"{coverage[-1]:.1f}" if coverage[-1] < 94.9 else ">95"
        plt.plot(d, coverage, *args, label=f"{label} ({coverage_string} %)")

    plt.legend(loc='lower right', prop={'size': 8})
    plt.grid(True, linestyle=':', color='k', alpha=0.5)
    locator = mdates.DayLocator([1, 15])
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.axis(
        xmin=np.datetime64('2021-07-28'),
        xmax=np.datetime64('2022-06-01'),
        ymin=0,
        ymax=100,
    )
    plt.title(f"{state} first dose coverage by age group")
    plt.ylabel("Vaccine coverage (%)")

    plt.subplot(122)
    for d, coverage, label in zip(
        dates_by_age, second_dose_coverage_by_age, labels_by_age
    ):
        if label == 'Ages 16+':
            args = ['k-']
        elif label == 'Ages 12+':
            args = ['k--']
        elif label == 'All ages':
            args = ['k:']
        else:
            args = []
        if label in ['Ages 16+', 'Ages 12+', 'All ages'] or state == 'AUS':
            coverage_string = f"{coverage[-1]:.1f}"
        else:
            coverage_string = f"{coverage[-1]:.1f}" if coverage[-1] < 94.9 else ">95"
        plt.plot(d, coverage, *args, label=f"{label} ({coverage_string} %)")

    plt.legend(loc='lower right', prop={'size': 8})
    plt.grid(True, linestyle=':', color='k', alpha=0.5)
    locator = mdates.DayLocator([1, 15])
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.axis(
        xmin=np.datetime64('2021-07-28'),
        xmax=np.datetime64('2022-06-01'),
        ymin=0,
        ymax=100,
    )
    plt.title(f"{state} second dose coverage by age group")
    plt.ylabel("Vaccine coverage (%)")

    plt.tight_layout()

    for extension in ['png', 'svg']:
        if not SKIP_FIGS:
            plt.savefig(f'{state}_coverage_by_age.{extension}')

    print(f"{state}")
    for agegroup in ['16+', '12+']:
        agegroup_ix = AGE_GROUPS.index(agegroup)
        first = first_dose_coverage_by_age[agegroup_ix]
        second = second_dose_coverage_by_age[agegroup_ix]
        d = dates_by_age[agegroup_ix]

        # Get the dosing interval and project targets:

        interval = len(second) - np.argwhere(first > second[-1])[0][0]
        coverage = first[-1]
        rate = (first[-1] - first[-8]) / 7

        second_coverage = second[-1]
        second_rate = (second[-1] - second[-8]) / 7

        levels = [60, 70, 80, 90]

        html_lines = []
        print(f"  Ages {agegroup}:")
        print(f"    {coverage:.1f}% first-dose coverage 💉")
        print(f"    {second_coverage:.1f}% second-dose coverage 💉💉")
        print(f"    {rate:.2f}%/day first-dose rate")
        print(f"    {second_rate:.2f}%/day second-dose rate")
        print(f"    {interval:.0f} days average dosing interval")

        html_lines.append(f"<b>Ages {agegroup}</b>")
        html_lines.append(f"  {coverage:.1f}% first-dose coverage 💉")
        html_lines.append(f"  {second_coverage:.1f}% second-dose coverage 💉💉")
        html_lines.append(f"  {rate:.2f}%/day first-dose rate")
        html_lines.append(f"  {second_rate:.2f}%/day second-dose rate")
        html_lines.append(f"  {interval:.0f} days average dosing interval")

        today = np.datetime64(datetime.now().strftime('%Y-%m-%d'))

        print("    1st dose targets 💉")
        html_lines.append("  <b>1st dose targets</b> 💉")
        for level in levels:
            if coverage > level:
                date = d[first > level][0]
                t = (today - date).astype(int)
                datestr = f"✅  {datefmt(date)} ({t} days ago)"
            else:
                t = (level - coverage) / rate
                date = d[-1] + int(np.ceil(t))
                t_from_today = (date - today).astype(int)
                datestr = f"{datefmt(date)} ({t_from_today:.0f} days)"
            print(f"      {level}%: {datestr}")
            html_lines.append(f"    {level}%: {datestr}")

        print("    2nd dose targets 💉💉:")
        html_lines.append("  <b>2nd dose targets</b> 💉💉")

        for level in levels:
            if second_coverage > level:
                date = d[second > level][0]
                t = (today - date).astype(int)
                datestr = f"✅  {datefmt(date)} ({t} days ago)"
            elif coverage > level:
                date = d[first > level][0] + int(np.ceil(interval))
                t_from_today = (date - today).astype(int)
                datestr = f"{datefmt(date)} ({t_from_today:.0f} days)"
            else:
                t = (level - coverage) / rate + interval
                date = d[-1] + int(np.ceil(t))
                t_from_today = (date - today).astype(int)
                datestr = f"{datefmt(date)} ({t_from_today:.0f} days)"
            print(f"      {level}%: {datestr}")
            html_lines.append(f"    {level}%: {datestr}")

        html_table_content[state][agegroup] = '\n'.join(html_lines)

# Update html:
html = Path('aus_vaccinations.html').read_text(encoding='utf8')
for state in STATES:
    for agegroup in ['12+', '16+']:
        start_marker = f'<!-- BEGIN {state} {agegroup} STATS -->\n'
        end_marker = f'<!-- END {state} {agegroup} STATS -->'
        pre, _ = html.split(start_marker, 1)
        _, post = html.split(end_marker, 1)
        html = (
            pre + start_marker + html_table_content[state][agegroup] + end_marker + post
        )
Path('aus_vaccinations.html').write_text(html, encoding='utf8')

# plt.show()
