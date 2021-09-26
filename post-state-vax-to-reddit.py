import sys
import json
from datetime import datetime
from pathlib import Path

import praw


def th(n):
    """Ordinal of an integer, eg "1st", "2nd" etc"""
    return str(n) + (
        "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )


def make_title():
    """Title of the reddit post"""
    vax_stats = json.loads(Path("latest_vax_stats.json").read_text())
    today = vax_stats['today']  # The date of the last update - likely yesterday by now

    today = datetime.fromisoformat(today)
    today = f'{today.strftime("%B")} {th(today.day)}'

    title = f"""Per state vaccination coverage by age group as of {today}. With
        projected dates for 16+ and 12+ coverage targets in comments."""
    return " ".join(title.split())


IMAGES = [
    "AUS_coverage_by_age.png",
    "NSW_coverage_by_age.png",
    "VIC_coverage_by_age.png",
    "QLD_coverage_by_age.png",
    "WA_coverage_by_age.png",
    "SA_coverage_by_age.png",
    "TAS_coverage_by_age.png",
    "ACT_coverage_by_age.png",
    "NT_coverage_by_age.png",
]

this_script_url = (
    "https://github.com/chrisjbillington/chrisjbillington.github.io/"
    + "blob/master/post-state-vax-to-reddit.py"
)


stats = '\n'.join(
    '    ' + line
    for line in Path('state_vax_stats.txt').read_text(encoding='utf8').splitlines()
)


COMMENT_1_TEXT = f"""\
Also in webpage form: https://chrisbillington.net/aus_vaccinations.html#state-age

Below is a big block of stats and projected targets, based on the current first-dose 7d
average rates and current average dosing intervals. These are also on the webpage in a
more readable form.

(split over two comments due to reddit comment length limit)

This post was made by a [bot]({this_script_url}) 🤖. Please let me know if something
looks broken.

{stats.split('    SA')[0]}
"""

COMMENT_2_TEXT = '    SA' +  stats.split('    SA')[1]

def get_flair_id(subreddit):
    for flair in subreddit.flair.link_templates:
        if flair['text'] == 'Independent Data Analysis':
            return flair['id']


if __name__ == '__main__':

    client_id = sys.argv[1]
    client_secret = sys.argv[2]
    password = sys.argv[3]

    user_agent = "linux:cvdu-chrisjbillington-autoposter:1.0.0 (by /u/chrisjbillington)"

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username="chrisjbillington",
        password=password,
    )

    subreddit = reddit.subreddit("CoronavirusDownunder")

    submission = subreddit.submit_gallery(
        title=make_title(),
        images=[{"image_path": p} for p in IMAGES],
        flair_id=get_flair_id(subreddit),
    )

    comment = submission.reply(COMMENT_1_TEXT)
    comment.reply(COMMENT_2_TEXT)
