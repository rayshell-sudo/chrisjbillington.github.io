import pandas as pd
import io
from subprocess import check_output
from datetime import datetime
import numpy as np
import json

# population data from here:

# ABS Estimated Resident Population, June 2020
# https://www.abs.gov.au/statistics/people/population/national-state-and-territory-population/jun-2020
POP_DATA = {
    # '12_14': {'MALE': 486_620, 'FEMALE': 461_308},
    # '15_19': {'MALE': 767_801, 'FEMALE': 725_052},
    '16_19': {'MALE': 614_430, 'FEMALE': 581_206},
    '20_24': {'MALE': 880_327, 'FEMALE': 832_409},
    '25_29': {'MALE': 960_798, 'FEMALE': 945_753},
    '30_34': {'MALE': 948_799, 'FEMALE': 975_556},
    '35_39': {'MALE': 909_066, 'FEMALE': 926_513},
    '40_44': {'MALE': 805_245, 'FEMALE': 815_516},
    '45_49': {'MALE': 825_971, 'FEMALE': 850_749},
    '50_54': {'MALE': 763_177, 'FEMALE': 800_990},
    '55_59': {'MALE': 759_231, 'FEMALE': 793_509},
    '60_64': {'MALE': 695_820, 'FEMALE': 736_391},
    '65_69': {'MALE': 607_161, 'FEMALE': 647_157},
    '70_74': {'MALE': 539_496, 'FEMALE': 563_487},
    '75_79': {'MALE': 370_469, 'FEMALE': 402_454},
    '80_84': {'MALE': 239_703, 'FEMALE': 288_405},
    '85_89': {'MALE': 130_999, 'FEMALE': 185_099},
    '90_94': {'MALE': 57_457, 'FEMALE':  100_955},
    '95_PLUS': {'MALE': 15_720, 'FEMALE': 37_186},
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
    if date < np.datetime64('2021-05-09'):
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
