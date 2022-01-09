import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent

import numpy as np
import tweepy


def th(n):
    """Ordinal of an integer, eg "1st", "2nd" etc"""
    return str(n) + (
        "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )


def fmt(a):
    """Dedent and replace single newline characters with a space"""
    return '\n\n'.join(line.replace('\n', ' ') for line in dedent(a).split('\n\n'))


def stats():
    stats = json.loads(Path("latest_vic_stats.json").read_text())
    today = stats['today']  # The date of the last update - should be today
    R_eff = stats['R_eff']
    u_R_eff = stats['u_R_eff']
    today = datetime.fromisoformat(today)
    today = f'{today.strftime("%B")} {th(today.day)}'
    return today, R_eff, u_R_eff


def tweet_1_text():
    today, R_eff, u_R_eff = stats()
    COMMENT_TEXT = f"""\
    VIC R_eff as of {today} with daily cases and restrictions. Latest estimate:
    R_eff = {R_eff:.2f} ± {u_R_eff:.2f}

    Plus SIR model projection.

    Cases shown on a linear scale (log scale in next tweet).

    More info https://chrisbillington.net/COVID_VIC_2021.html

    #COVID19Vic"""
    return fmt(COMMENT_TEXT)

def tweet_2_text():
    today, R_eff, u_R_eff = stats()
    COMMENT_TEXT = f"""\
    VIC R_eff as of {today} with daily cases and restrictions. Latest estimate:
    R_eff = {R_eff:.2f} ± {u_R_eff:.2f}

    Plus SIR model projection.

    (Cases shown on a log scale)

    More info https://chrisbillington.net/COVID_VIC_2021.html

    #COVID19Vic"""
    return fmt(COMMENT_TEXT)

def tweet_3_text():
    stats = json.loads(Path("latest_vic_stats.json").read_text())

    proj_lines = [
        "day  cases  68% range",
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

    proj_lines = "\n    ".join(proj_lines)

    doubling_time = 5 * np.log(2) / np.log(stats['R_eff'])

    doubling_or_halving = "Doubling" if doubling_time > 0 else "Halving"

    COMMENT_TEXT = f"""\
    Cases in VIC if the current trend continues:
    
    {proj_lines}
    
    {doubling_or_halving} time is {abs(doubling_time):.1f} days.
    
    #COVID19Vic"""

    return dedent(COMMENT_TEXT)

# def tweet_4_text():
#     COMMENT_TEXT = """\
#     Note that these projections do not take into account recent/upcoming easing of
#     restrictions. If the trend changes as a result of easing, the projections will also
#     change once this is reflected in case numbers, but not in advance."""
#     return fmt(COMMENT_TEXT)

# def tweet_5_text():
#     COMMENT_TEXT = """\

#     Vaccination is the only way out of this lockdown. See where vaccines are available
#     near you: https://covid19nearme.com.au/state/vic/vaccination. If you're already
#     booked in, keep an eye out as more supply might mean you can move it up.

#     #Covid19Vic
#     """
#     return fmt(COMMENT_TEXT)

if __name__ == '__main__':
    api_key = sys.argv[1]
    api_secret_key = sys.argv[2]
    access_token = sys.argv[3]
    access_token_secret = sys.argv[4]

    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # Upload images
    linear = api.media_upload("COVID_VIC_2021_linear.png")
    vax_linear = api.media_upload("COVID_VIC_2021_vax_linear.png")
    log = api.media_upload("COVID_VIC_2021.png")
    vax_log = api.media_upload("COVID_VIC_2021_vax.png")
    disclaimer = api.media_upload("disclaimer.png")
 
    # Post tweets with images
    tweet_1 = api.update_status(
        status=tweet_1_text(),
        media_ids=[linear.media_id, vax_linear.media_id],
    )
 
    tweet_2 = api.update_status(
        status=tweet_2_text(),
        media_ids=[log.media_id, vax_log.media_id],
        in_reply_to_status_id=tweet_1.id,
        auto_populate_reply_metadata=True,
    )

    tweet_3 = api.update_status(
        status=tweet_3_text(),
        in_reply_to_status_id=tweet_2.id,
        auto_populate_reply_metadata=True,
    )

    # tweet_4 = api.update_status(
    #     status=tweet_4_text(),
    #     in_reply_to_status_id=tweet_3.id,
    #     auto_populate_reply_metadata=True,
    #     media_ids=[disclaimer.media_id],
    # )

    # tweet_5 = api.update_status(
    #     status=tweet_5_text(),
    #     in_reply_to_status_id=tweet_4.id,
    #     auto_populate_reply_metadata=True,
    # )

