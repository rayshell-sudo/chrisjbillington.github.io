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


def datefmt(d):
    """Format a np.datetime64 as e.g."August 5th" """
    d = d.astype(datetime)
    n = d.day
    th = "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f'{d.strftime("%B").rjust(8)} {str(n).rjust(2)}{th}'


url = "https://vaccinedata.covid19nearme.com.au/data/air_residence.json"

data = json.loads(requests.get(url).content)
# Convert dates to np.datetime64 and sort:
for row in data:
    row['DATE_AS_AT'] = np.datetime64(row['DATE_AS_AT'])
data.sort(key=lambda row: row['DATE_AS_AT'])

# First date when data for ages 12-15 became available:
AGES_12_15_FROM = np.datetime64('2021-09-13')

dates = np.array(sorted(set([row['DATE_AS_AT'] for row in data])))
dates_12_15 = dates[dates >= AGES_12_15_FROM]


STATES = ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT', 'NT']

html_table_content = {}

for state in STATES:

    html_table_content[state] = {}

    labels_by_age = []
    first_dose_coverage_by_age = []
    second_dose_coverage_by_age = []
    dates_by_age = []

    for group_start in ['12+', '16+', 12, 16, 20, 30, 40, 50, 60, 70, 80, 90][::-1]:
        if group_start == '12+':
            ranges = [(12, 15), (16, 999)]
        elif group_start == '16+':
            ranges = [(16, 999)]
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
        dates_by_age.append(dates_12_15 if group_start == 12 else dates)
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

            pop += rows[0]['ABS_ERP_JUN_2020_POP']
            # Now one row for each date
            for i, row in enumerate(rows):
                first_doses[i] += row[FIRST_KEY]
                second_doses[i] += row[SECOND_KEY]

        first_dose_coverage_by_age.append(100 * first_doses / pop)
        second_dose_coverage_by_age.append(100 * second_doses / pop)

        if group_start == '16+':
            label = 'Ages 16+'
        elif group_start == '12+':
            label = 'Ages 12+'
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
            args = ['k--']
        elif label == 'Ages 12+':
            args = ['k:']
        else:
            args = []
        plt.plot(d, coverage, *args, label=f"{label} ({coverage[-1]:.1f} %)")

    plt.legend(loc='upper right', prop={'size': 9})
    plt.grid(True, linestyle=':', color='k', alpha=0.5)
    locator = mdates.DayLocator([1, 15])
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.axis(
        xmin=np.datetime64('2021-07-28'),
        xmax=np.datetime64('2022-01-01'),
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
            args = ['k--']
        elif label == 'Ages 12+':
            args = ['k:']
        else:
            args = []
        plt.plot(d, coverage, *args, label=f"{label} ({coverage[-1]:.1f} %)")

    plt.legend(loc='upper right', prop={'size': 9})
    plt.grid(True, linestyle=':', color='k', alpha=0.5)
    locator = mdates.DayLocator([1, 15])
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.axis(
        xmin=np.datetime64('2021-07-28'),
        xmax=np.datetime64('2022-01-01'),
        ymin=0,
        ymax=100,
    )
    plt.title(f"{state} second dose coverage by age group")
    plt.ylabel("Vaccine coverage (%)")

    plt.tight_layout()

    for extension in ['png', 'svg']:
        plt.savefig(f'{state}_coverage_by_age.{extension}')

    print(f"{state}")
    for agegroup in ['16+', '12+']:
        if agegroup == '12+':
            first = first_dose_coverage_by_age[-1]
            second = second_dose_coverage_by_age[-1]
            d = dates_by_age[-1]
        elif agegroup == '16+':
            first = first_dose_coverage_by_age[-2]
            second = second_dose_coverage_by_age[-2]
            d = dates_by_age[-2]
        else:
            raise ValueError(agegroup)

        # Get the dosing interval and project targets:

        interval = len(second) - np.argwhere(first > second[-1])[0][0]
        coverage = first[-1]
        rate = (first[-1] - first[-8]) / 7

        second_coverage = second[-1]
        # second_rate = (second[-1] - second[-8]) / 7

        levels = [70, 80, 90]

        html_lines = []
        print(f"  Ages {agegroup}:")
        print(f"    {coverage:.1f}% first-dose coverage 💉")
        print(f"    {second_coverage:.1f}% second-dose coverage 💉💉")
        print(f"    {rate:.2f}%/day first-dose rate")
        print(f"    {interval:.0f} days average dosing interval")

        html_lines.append(f"<b>Ages {agegroup}</b>")
        html_lines.append(f"  {coverage:.1f}% first-dose coverage 💉")
        html_lines.append(f"  {second_coverage:.1f}% second-dose coverage 💉💉")
        html_lines.append(f"  {rate:.2f}%/day first-dose rate")
        html_lines.append(f"  {interval:.0f} days average dosing interval")

        today = np.datetime64(datetime.now().strftime('%Y-%m-%d'))

        print("    1st dose targets 💉 (@ current 1st dose rate)")
        html_lines.append("  <b>1st dose targets</b> 💉 (@ current 1st dose rate)")
        for level in levels:
            if coverage > level:
                date = d[first > level][0]
                t = (today - date).astype(int)
                datestr = f"✅  {datefmt(date)} ({t} days ago)"
            else:
                t = (level - coverage) / rate
                date = d[-1] + int(round(t))
                t_from_today = (date - today).astype(int)
                datestr = f"{datefmt(date)} ({t_from_today:.0f} days)"
            print(f"      {level}%: {datestr}")
            html_lines.append(f"    {level}%: {datestr}")

        print("    2nd dose targets 💉💉 (@ current 1st dose rate + dosing interval):")
        html_lines.append("  <b>2nd dose targets</b> 💉💉 (@ current 1st dose rate + interval)")

        for level in levels:
            if second_coverage > level:
                date = d[second > level][0]
                t = (d[-1] - date).astype(int)
                datestr = f"✅  {datefmt(date)} ({t} days ago)"
            elif coverage > level:
                date = d[first > level][0] + int(round(interval))
                t_from_today = (date - today).astype(int)
                datestr = f"{datefmt(date)} ({t_from_today:.0f} days)"
            else:
                t = (level - coverage) / rate + interval
                date = d[-1] + int(round(t))
                datestr = f"{datefmt(date)} ({t:.0f} days)"
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
