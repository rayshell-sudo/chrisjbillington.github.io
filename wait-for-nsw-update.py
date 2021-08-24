# script to wait for NSW case nubmers for today to appear on covidlive.

import pandas as pd
import time

def covidlive_updated_today():
    """Check covidlive for NSW case numbers for today, and return bool for whether
    they're there."""
    df = pd.read_html('https://covidlive.com.au/report/daily-source-overseas/nsw')[1]
    return df['NET'][0] != '-'

if __name__ == '__main__':
    # Hit covidlive once a minute checking if it's updated:
    while not covidlive_updated_today():
        time.sleep(60)
    # After it is updated, wait an extra minute. Sometimes the initial update is like,
    # zero or something.
    time.sleep(60)
    print("ready!")
