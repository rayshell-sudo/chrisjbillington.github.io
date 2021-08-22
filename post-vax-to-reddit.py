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
    latest_cumulative_doses = vax_stats['latest_cumulative_doses']
    latest_daily_doses = vax_stats['latest_daily_doses']
    phase_C_date = vax_stats['phase_C_date']
    today = vax_stats['today']  # The date of the last update - likely yesterday by now

    today = datetime.fromisoformat(today)
    today = f'{today.strftime("%B")} {th(today.day)}'

    phase_C_date = datetime.fromisoformat(phase_C_date)
    phase_C_date = f'{phase_C_date.strftime("%B")} {th(phase_C_date.day)}'

    title = f"""
        AUS vaccination rollout as of {today}. {latest_daily_doses / 1000:.1f}k doses
        per day 7d average, {latest_cumulative_doses / 1e6:.2f}M total doses. With first
        and second dose coverage by age group and 2021 projections based on expected
        supply. Projected 80% 16+ coverage: {phase_C_date}.
    """
    return " ".join(title.split())


IMAGES = [
    "daily_doses_by_state.png",
    "doses_by_weekday.png",
    "coverage_by_agegroup.png",
    "coverage_2nd_by_agegroup.png",
    "cumulative_doses.png",
    "daily_doses_by_state.png",
    "projection_by_type.png",
    "projection_cumulative_by_type.png",
]

COMMENT_TEXT = """\
More info/methodology: https://chrisbillington.net/aus_vaccinations.html

This post was made by a bot 🤖. Please let me know if something looks broken."""


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
    submission.reply(COMMENT_TEXT)
