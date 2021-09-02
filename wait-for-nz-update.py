# script to wait for NZ case numbers for today to appear on MOH official dataset.

import pandas as pd
import time
from datetime import datetime
import io
from subprocess import check_output

def moh_updated_today():
    """Check if NZ MOH dataset for today exists yet, return True if it does"""
    today = datetime.now().strftime('%Y-%m-%d')
    URL = f"https://www.health.govt.nz/system/files/documents/pages/covid_cases_{today}.csv"
    try:
        pd.read_csv(io.BytesIO(check_output(["curl", URL])))
    except pd.errors.ParserError:
        return False
    except Exception as e:
        print(e)
        return False
    return True

if __name__ == '__main__':
    # Hit MoH once every 15 minutes checking if it's updated:
    while not moh_updated_today():
        time.sleep(900)
    print("ready!")
