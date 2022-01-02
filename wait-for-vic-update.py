# script to wait for VIC case nubmers for today to appear on covidlive.

import pandas as pd
import time

def covidlive_updated_today():
    """Check covidlive for VIC net and ~~wild~~ numbers for today, and return bool for
    whether they're there."""
    try:
        df = pd.read_html('https://covidlive.com.au/report/daily-source-overseas/vic')[1]
    except Exception as e:
        print(str(e))
        return False
    return df['LOCAL'][0] not in  ['-', df['LOCAL'][1]]

if __name__ == '__main__':
    # Hit covidlive once every 5 minutes checking if it's updated:
    while not covidlive_updated_today():
        time.sleep(300)
    # After it is updated, wait an extra minute. Sometimes the initial update is like,
    # zero or something.
    time.sleep(60)
    print("ready!")
