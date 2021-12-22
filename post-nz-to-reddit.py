import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent

import numpy as np
import praw


def th(n):
    """Ordinal of an integer, eg "1st", "2nd" etc"""
    return str(n) + (
        "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )


def make_title():
    """Title of the reddit post"""
    stats = json.loads(Path("latest_nz_stats.json").read_text())
    today = stats['today']  # The date of the last update - should be today
    R_eff = stats['R_eff']
    u_R_eff = stats['u_R_eff']

    today = datetime.fromisoformat(today)
    today = f'{today.strftime("%B")} {th(today.day)}'

    title=f"""NZ R_eff as of {today}, with daily cases and restrictions. Latest
        estimate: R_eff = {R_eff:.02f} ± {u_R_eff:.02f}. (images with both linear and
        log scales)
    """
    return " ".join(title.split())

def make_comment():
    stats = json.loads(Path("latest_nz_stats.json").read_text())
    R_eff_auckland = stats['R_eff_auckland']
    u_R_eff_auckland = stats['u_R_eff_auckland']
    R_eff_notauckland = stats['R_eff_notauckland']
    u_R_eff_notauckland = stats['u_R_eff_notauckland']
    today = stats['today']
    
    proj_lines = [
        "day  cases  68% range",
        "---------------------",
    ]
    for proj in stats['projection'][1:8]:
        presser_date = datetime.fromisoformat(proj['date']) + timedelta(1)
        cases = proj['cases']
        lower = proj['lower']
        upper = proj['upper']
        weekday = presser_date.strftime("%a:")
        cases_str = f"{cases:.0f}".rjust(5)
        range_str = f"{lower:.0f}—{upper:.0f}".rjust(10)
        proj_lines.append(" ".join([weekday, cases_str, range_str]))


    proj_lines = "\n        ".join(proj_lines)

    doubling_time = 5 * np.log(2) / np.log(stats['R_eff'])

    this_script_url = (
        "https://github.com/chrisjbillington/chrisjbillington.github.io/"
        + "blob/master/post-nz-to-reddit.py"
    )

    COMMENT_TEXT = f"""\
    More info/methodology: https://chrisbillington.net/COVID_NZ.html

    First plot has case numbers on a linear scale, second plot is exactly the
    the same but with case numbers on a log scale.

    [Auckland](https://chrisbillington.net/COVID_NZ_auckland.png?dontcache={today}):
    R_eff = {R_eff_auckland:.02f} ± {u_R_eff_auckland:.02f}

    [NZ excluding
    Auckland](https://chrisbillington.net/COVID_NZ_notauckland.png?dontcache={today}):
    R_eff = {R_eff_notauckland:.02f} ± {u_R_eff_notauckland:.02f}

    Expected case numbers if the current  trend continues:

        {proj_lines}

    The current {"doubling" if doubling_time > 0 else "halving"} time is
    {abs(doubling_time):.01f} days.

    Usual disclaimer about trendlines:

    >The plotted trend is a simple extrapolation of what will happen if R_eff remains at
    >its current value. This does not take into account that things are in a state of
    >flux. As restrictions take effect, the virus should have fewer opportunities for
    >spread, and Reff will decrease. If restrictions are eased, it may increase. Contact
    >tracing may suppress spread to a greater or lesser degree over time.
    >
    >Furthermore, when case numbers are small, the random chance of how many people each
    >infected person subsequently infects can cause estimates of R_eff to vary randomly
    >in time. As such the projection should be taken with a grain of salt—it is merely
    >an indication of the trend as it is right now.

    This post was made by a [bot]({this_script_url}) 🤖. Please let me know if something
    looks broken."""

    return dedent(COMMENT_TEXT)

def get_flair_id(subreddit):
    patterns = {
        'newzealand': 'Coronavirus',
        'CoronavirusDownunder': 'Independent Data Analysis',
    }
    pattern = patterns[subreddit.display_name]
    for flair in subreddit.flair.link_templates:
        if pattern in flair['text']:
            return flair['id']
    raise LookupError(f"Couldn't find flair for subreddit {subreddit.display_name}")


IMAGES = [
    "COVID_NZ_linear.png",
    "COVID_NZ.png",
]

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

    for subreddit_name in ["newzealand", "CoronavirusDownunder"]:
        subreddit = reddit.subreddit(subreddit_name)

        submission = subreddit.submit_gallery(
            title=make_title(),
            images=[{"image_path": p} for p in IMAGES],
            flair_id=get_flair_id(subreddit),
        )
        submission.reply(make_comment())
