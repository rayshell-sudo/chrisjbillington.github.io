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
    stats = json.loads(Path("latest_nsw_stats.json").read_text())
    today = stats['today']  # The date of the last update - should be today
    R_eff = stats['R_eff']
    u_R_eff = stats['u_R_eff']

    today = datetime.fromisoformat(today)
    today = f'{today.strftime("%B")} {th(today.day)}'

    title=f"""NSW R_eff as of {today}, with daily cases and restrictions. Latest
        estimate: R_eff = {R_eff:.02f} ± {u_R_eff:.02f}. Plus SIR model projection.
        (images with both linear and log scales)
    """
    return " ".join(title.split())

def make_comment():
    stats = json.loads(Path("latest_nsw_stats.json").read_text())
    R_eff_concern = stats['R_eff_concern']
    u_R_eff_concern = stats['u_R_eff_concern']
    R_eff_others = stats['R_eff_others']
    u_R_eff_others = stats['u_R_eff_others']
    R_eff_hunter = stats['R_eff_hunter']
    u_R_eff_hunter = stats['u_R_eff_hunter']
    R_eff_illawarra = stats['R_eff_illawarra']
    u_R_eff_illawarra = stats['u_R_eff_illawarra']
    R_eff_wnsw = stats['R_eff_wnsw']
    u_R_eff_wnsw = stats['u_R_eff_wnsw']
    R_eff_sydney = stats['R_eff_sydney']
    u_R_eff_sydney = stats['u_R_eff_sydney']
    R_eff_not_sydney = stats['R_eff_not_sydney']
    u_R_eff_not_sydney = stats['u_R_eff_not_sydney']
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
        + "blob/master/post-nsw-to-reddit.py"
    )

    COMMENT_TEXT = f"""\
    More info/methodology: https://chrisbillington.net/COVID_NSW.html

    First two plots have case numbers on a linear scale, next two plots are exactly the
    same but with case numbers on a log scale.

    First (and third) plot show an exponential trendline, second (and fourth) are SIR
    models that assume 27% of infections will be detected via testing.

    [Greater
    Sydney](https://chrisbillington.net/COVID_NSW_sydney.png?dontcache={today}): R_eff =
    {R_eff_sydney:.02f} ± {u_R_eff_sydney:.02f}

    [NSW excluding Greater
    Sydney](https://chrisbillington.net/COVID_NSW_not_sydney.png?dontcache={today}):
    R_eff = {R_eff_not_sydney:.02f} ± {u_R_eff_not_sydney:.02f}
    
    [former LGAs of concern](https://chrisbillington.net/COVID_NSW_LGA_concern.png?dontcache={today}): R_eff =
    {R_eff_concern:.02f} ± {u_R_eff_concern:.02f}

    [NSW excluding former LGAs of
    concern](https://chrisbillington.net/COVID_NSW_LGA_others.png?dontcache={today}):
    R_eff = {R_eff_others:.02f} ± {u_R_eff_others:.02f}

    [Hunter region](https://chrisbillington.net/COVID_NSW_hunter.png?dontcache={today}):
    R_eff = {R_eff_hunter:.02f} ± {u_R_eff_hunter:.02f}

    [Illawarra
    region](https://chrisbillington.net/COVID_NSW_illawarra.png?dontcache={today}):
    R_eff = {R_eff_illawarra:.02f} ± {u_R_eff_illawarra:.02f}

    [Western New South
    Wales](https://chrisbillington.net/COVID_NSW_wnsw.png?dontcache={today}): R_eff =
    {R_eff_wnsw:.02f} ± {u_R_eff_wnsw:.02f}

    Note: LGA/region-specific data is several days out of date compared to total
    numbers.

    Expected case numbers if the current  trend continues:

        {proj_lines}

    The current {"doubling" if doubling_time > 0 else "halving"} time is
    {abs(doubling_time):.01f} days.

    Usual disclaimer about trendlines:

    > The plotted exponential trendlines in the above plots are simple extrapolations of
    > what will happen if Reff remains at its current value. the SIR model projection,
    > although it accounts for the effect of increasing immunity, still assumes that
    > Reff otherwise remains constant.
    > 
    > This does not take into account that things are in a state of flux. If
    > restrictions are tightened, the virus should have fewer opportunities for spread,
    > and Reff will decrease. If restrictions are eased, it may increase. Contact
    > tracing may suppress spread to a greater or lesser degree over time. The above
    > plots specifically showing the effect of population immunity do take into account
    > a reduction in Reff as immunity due to infections increases, but ignores any other
    > possible future changes in Reff.
    > 
    > Furthermore, when case numbers are small, the random chance of how many people
    > each infected person subsequently infects can cause estimates of Reff to vary
    > randomly in time. As such the projections should be taken with a grain of salt—it
    > is merely an indication of the trend as it is right now.

    This post was made [by a bot]({this_script_url}) 🤖. Please let me know if something
    looks broken."""

    return dedent(COMMENT_TEXT)

def get_flair_id(subreddit):
    for flair in subreddit.flair.link_templates:
        if flair['text'] == 'Independent Data Analysis':
            return flair['id']


IMAGES = [
    "COVID_NSW_linear.png",
    "COVID_NSW_vax_linear.png",
    "COVID_NSW.png",
    "COVID_NSW_vax.png",
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

    subreddit = reddit.subreddit("CoronavirusDownunder")

    submission = subreddit.submit_gallery(
        title=make_title(),
        images=[{"image_path": p} for p in IMAGES],
        flair_id=get_flair_id(subreddit),
    )
    submission.reply(make_comment())
