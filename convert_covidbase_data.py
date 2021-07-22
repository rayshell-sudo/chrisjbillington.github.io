import pandas as pd
import io
from subprocess import check_output
from datetime import datetime
import numpy as np
import json

# population data from here:

# from ABS, released 11 June 2021: https:FEMALE//www.abs.gov.au/statistics/people/
# population/national-state-and-territory-population/latest-release
POP_DATA = {
    '16_19': {'MALE': 620_871, 'FEMALE': 587_540},
    '20_24': {'MALE': 899_060, 'FEMALE': 850_721},
    '25_29': {'MALE': 957_746, 'FEMALE': 949_146},
    '30_34': {'MALE': 933_587, 'FEMALE': 959_375},
    '35_39': {'MALE': 885_449, 'FEMALE': 896_946},
    '40_44': {'MALE': 793_623, 'FEMALE': 802_591},
    '45_49': {'MALE': 825_686, 'FEMALE': 854_071},
    '50_54': {'MALE': 750_782, 'FEMALE': 785_017},
    '55_59': {'MALE': 757_941, 'FEMALE': 790_197},
    '60_64': {'MALE': 677_332, 'FEMALE': 714_617},
    '65_69': {'MALE': 596_458, 'FEMALE': 631_045},
    '70_74': {'MALE': 518_927, 'FEMALE': 539_265},
    '75_79': {'MALE': 351_089, 'FEMALE': 383_275},
    '80_84': {'MALE': 227_926, 'FEMALE': 277_175},
    '85_89': {'MALE': 129_208, 'FEMALE': 184_648},
    '90_94': {'MALE': 53_978, 'FEMALE': 99_014},
    '95_PLUS': {'MALE': 13_867, 'FEMALE': 34_246},
}

for pops in POP_DATA.values():
    pops['TOTAL'] = pops['MALE'] + pops['FEMALE'] 

age_groups = list(POP_DATA.keys())

# Redirected from https://covidbaseau.com/vaccinations/download
url = (
    "https://docs.google.com/spreadsheets/d/"
    "1gStZ55jH-weWAkI-EGhOzYo-lQeEKNlvm1F_Y70E4gc/export?format=csv"
)
data = check_output(['curl', '-L', url])

df = pd.read_csv(io.StringIO(data.decode('utf8')))

processed_data = []
for i, row in df.iterrows():
    date = row['Date']
    if not isinstance(date, str):
        continue
    date = datetime.strptime(date, '%d %b %y')
    date = np.datetime64(date, 'D')
    if date < np.datetime64('2021-06-09'):
        continue
    data_for_date = {'DATE_AS_AT': str(date)}
    for age_group in age_groups:
        doses_in_age_group = 0
        for sex in ['MALE', 'FEMALE']:
            covidbase_sex = sex.capitalize()
            covidbase_age_group = age_group.replace('_PLUS', '+').replace("_", '-')
            covidase_column_name = f'{covidbase_sex}_{covidbase_age_group}_%_Vaccinated'
            doses_percent = float(row[covidase_column_name].rstrip('%'))
            doses = doses_percent * POP_DATA[age_group][sex] / 100.0
            data_for_date[f'AIR_{age_group}_{sex}_PCT'] = doses_percent
            doses_in_age_group += doses
        percent_in_age_group = 100 * doses_in_age_group / POP_DATA[age_group]['TOTAL']
        doses_in_age_group = int(round(doses_in_age_group))
        percent_in_age_group = round(percent_in_age_group, 1)
        data_for_date[f'AIR_{age_group}_FIRST_DOSE_COUNT'] = doses_in_age_group
        data_for_date[f'AIR_{age_group}_FIRST_DOSE_PCT'] = percent_in_age_group
    processed_data.append(data_for_date)

with open('covidbase_data.json', 'w') as f:
    json.dump(processed_data, f, indent=4)
