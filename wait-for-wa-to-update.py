# script to wait for WA case numbers to be released

import time
from datetime import datetime, timedelta
from get_wa_cases import get_cases

def press_release_today():
    yesterday = f"{datetime.now() - timedelta(days=1):%Y-%m-%d}"
    if yesterday in get_cases():
        return True
    return False


if __name__ == '__main__':
    # Check once every 5 minutes for a press release
    while not press_release_today():
        time.sleep(300)
    print("ready!")
