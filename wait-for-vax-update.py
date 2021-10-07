import json
import requests
import numpy as np
from pathlib import Path
import time


def latest_covidlive_date():
    """Return a np.datetime64 for the date covidlive most recently updated its
    vaccination data"""
    COVIDLIVE = 'https://covidlive.com.au/covid-live.json'
    covidlivedata = json.loads(requests.get(COVIDLIVE).content)

    STATES = ['AUS', 'NSW', 'VIC', 'SA', 'WA', 'TAS', 'QLD', 'NT', 'ACT']

    # We want the most recent date common to all jurisdictions
    maxdates = []
    for state in STATES:
        maxdate = max(
            np.datetime64(report['REPORT_DATE'])
            for report in covidlivedata
            if report['CODE'] == state and report['VACC_DOSE_CNT'] is not None
        )
        maxdates.append(maxdate)

    return min(maxdates)

def latest_AIR_date():
    AIR_JSON = "https://vaccinedata.covid19nearme.com.au/data/air.json"
    AIR_data = json.loads(requests.get(AIR_JSON).content)
    return np.datetime64(max(row["DATE_AS_AT"] for row in AIR_data)) + 1


def latest_air_residence_date():
    url = "https://vaccinedata.covid19nearme.com.au/data/air_residence.json"
    data = json.loads(requests.get(url).content)
    return np.datetime64(max(row["DATE_AS_AT"] for row in data)) + 1

def latest_site_update():
    """Return the date in latest-vax-stats.json as a np.datetime64"""
    return np.datetime64(json.loads(Path('latest_vax_stats.json').read_text())['today'])

def updates():
    return [latest_covidlive_date(), latest_AIR_date(), latest_air_residence_date()]

if __name__ == '__main__':
    # Check every 10 minutes if we're out of date:
    while True:
        try:
            outdated = min(updates()) > latest_site_update()
        except Exception as e:
            print(e)
            continue
        if outdated:
            break
        time.sleep(600)
    print("ready!")
