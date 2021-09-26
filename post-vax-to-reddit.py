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
        and second dose coverage by age group and state/territory, and 2021 projections
        based on expected supply. Projected 80% 16+ coverage: {phase_C_date}.
    """
    return " ".join(title.split())


IMAGES = [
    "daily_doses_by_state.png",
    "doses_by_weekday.png",
    "coverage_by_agegroup.png",
    "coverage_2nd_by_agegroup.png",
    "coverage_by_state.png",
    "coverage_2nd_by_state.png",    
    "cumulative_doses.png",
    "daily_doses_by_state_longproject.png",
    "projection_by_type.png",
    "projection_cumulative_by_type.png",
]

this_script_url = (
    "https://github.com/chrisjbillington/chrisjbillington.github.io/"
    + "blob/master/post-vax-to-reddit.py"
)

# extra = """
# Some changes in the projection.

# 1. +500k doses from Singapore. Thanks, Singapore! These have been added to supply
#    arriving by end of this week.

# 2. I found some doses under the couch pillows: The 500k COVAX doses from a few months
#    ago, I managed to accidentally misplace at some point. So I have added them back in.
#    Also, I typod a 1M doses projected Pfizer shipment in early September as only 100k
#    doses 😬. So I've bumped it back up to +1M. These two mistakes account for 1.4M doses.

# Apologies for the mistakes. After correcting these and adding the Singapore doses, the
# "dip" is all but gone. And the October spike is reduced, because at that point, we'll be
# running out of people to vaccinate.
# """

# # Schedule some extra text:
# if datetime.now().strftime('%Y-%m-%d') != '2021-09-01':
extra = ""

COMMENT_TEXT = f"""\
More info/methodology: https://chrisbillington.net/aus_vaccinations.html
{extra}
FAQ:

Q. Why did the projection for the phase C target date move back a few days?

A. I used to assume all supply would go to ages 16+ first. Now that we have to share
   with those pesky 12–15 year-olds (and now that I've updated my code to acknowledge
   this) there's less supply for the rest of us and it'll take a tad longer for us to
   reach any given threshold.

Q. How come in the "Projected cumulative 1st and 2nd doses" plot, the axis on the right
   is total eligible (12+) population, but the lines for the 70% and 80% thresholds are
   labelled "Phase B" and "Phase C", when these phases in the national plan refer to the
   16+ population?

A. This projection was originally made to project total doses based on available supply,
   and doesn't have any way of predicting which age group doses will go to first when
   they're all eligible. So I have simply marked the thresholds for the total eligible
   population. This will correspond well to the thresholds for the 16+ population only
   if ages 12–15 have similar coverage at those points in time to the rest of the
   population (uptake is very fast in the 12–15s right now, so maybe this isn't so
   unreasonable).

Q. What about hesitancy?

A. These projections completely ignore hesitancy, and represent what would be possible
   if everyone steps up for a dose when it's their turn. Without knowing what level of
   hesitancy there will be, I can't take it into account - so you should just mentally
   draw a horizontal line at whatever level of hesitancy you think there will be. For
   what it's worth, I am personally sceptical that anything like the 20% levels of
   hesitancy often cited will eventuate in Australia. Given Australia's past track
   record on vaccinations, I expect we'll end up one of the most highly vaccinated
   countries in the world against COVID—the lower takeup seen in many other
   countries is not indicative of what we can achieve here.

This post was made by a [bot]({this_script_url}) 🤖. Please let me know if something
looks broken."""


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
