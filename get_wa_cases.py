import requests
from datetime import datetime, timedelta
import time
import json
from pathlib import Path

def get_cases_for_day(url):
    page = requests.get(url).content.decode('utf8')
    page = '\n'.join(
        l for l in page.splitlines() if '<meta name="description"' not in l
    )
    suffixes = ['new local', 'were local', 'local']
    for suffix in suffixes:
        if suffix in page:
            cases = page.split(suffix, 1)[0].split()[-1]
            break
    else:
        cases = 0
    try:
        return int(cases)
    except ValueError:
        try:
            return {
                'no': 0,
                'one': 1,
                'two': 2,
                'three': 3,
                'four': 4,
                'five': 5,
                'six': 6,
                'seven': 7,
                'eight': 8,
                'nine': 9,
                'ten': 10,
            }[cases]
        except KeyError:
            print(url, cases)
            raise


def get_cases():
    try:
        # Add to existing file if already present
        data = json.loads(Path("wa_local_cases.json").read_text(encoding='utf8'))
    except FileNotFoundError:
        data = {}

    url = "https://ww2.health.wa.gov.au/News/Media-releases-listing-page"
    page = requests.get(url).content.decode('utf8')
    for line in page.splitlines():
        if 'title="COVID-19 update ' in line:
            date = line.split('title="COVID-19 update ', 1)[1].split('"', 1)[0]
            url = line.split('href="', 1)[1].split('"', 1)[0]
            url = f"https://ww2.health.wa.gov.au{url}"
            try:
                date = datetime.strptime(date, "%d %B %Y") - timedelta(days=1)
            except ValueError:
                continue
            date = f"{date:%Y-%m-%d}"
            if date in data or date < "2021-12-10":
                break
            cases = get_cases_for_day(url)
            print(f"{date}: {cases}")
            data[date] = cases
            time.sleep(0.5)

    Path("wa_local_cases.json").write_text(
        json.dumps(data, indent=4, sort_keys=True), encoding='utf8'
    )
    return {d: c for d, c in sorted(data.items())}
