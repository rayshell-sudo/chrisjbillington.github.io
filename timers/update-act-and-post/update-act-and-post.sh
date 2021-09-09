#! /bin/bash
set -euxo

# Source our secrets, clone a temporary copy of the repo and cd to it. This also gets
# the LOCKFILE variable for locking access to the main repo.
source "$(dirname "$BASH_SOURCE")/../common.sh"

# Wait for ACT data to become available:
if python wait-for-act-update.py | grep "ready!"; then
    ./act.sh
fi

# Post to reddit:
python post-act-to-reddit.py \
  "${REDDIT_CLIENT_ID}" \
  "${REDDIT_CLIENT_SECRET}" \
  "${REDDIT_PASSWORD}"

# Post to twitter:
python post-act-to-twitter.py \
  "${TWITTER_API_KEY}" \
  "${TWITTER_API_SECRET_KEY}" \
  "${TWITTER_ACCESS_TOKEN}" \
  "${TWITTER_ACCESS_TOKEN_SECRET}"

# Commit and push
git commit --all -m "ACT update"

# pull first to decrease the chances of a collision. Lockfile ensures this isn't racey
# with respect to the other automation jobs running on this computer, but if we're
# unluckly it could still collide with other pushes to remote.
flock "${LOCKFILE}" -c  "git pull --rebase --autostash; git push"
