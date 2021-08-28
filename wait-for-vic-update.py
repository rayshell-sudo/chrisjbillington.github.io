# script to wait for VIC case nubmers for today to appear on covidlive.

import pandas as pd
import time

def covidlive_updated_today():
    """Check covidlive for VIC net and wild numbers for today, and return bool for
    whether they're there."""
    df1 = pd.read_html('https://covidlive.com.au/report/daily-source-overseas/vic')[1]
    df2 = pd.read_html('https://covidlive.com.au/report/daily-wild-cases/vic')[1]
    return df1['NET'][0] != '-' and df2['TOTAL'][0] != '-'

if __name__ == '__main__':
    # Hit covidlive once a minute checking if it's updated:
    while not covidlive_updated_today():
        time.sleep(60)
    # After it is updated, wait an extra minute. Sometimes the initial update is like,
    # zero or something.
    time.sleep(60)
    print("ready!")
