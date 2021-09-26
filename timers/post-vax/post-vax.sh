#! /bin/bash
set -euxo

# Source our secrets, clone a temporary copy of the repo and cd to it. This also gets
# the LOCKFILE variable for locking access to the main repo.
source "$(dirname "$BASH_SOURCE")/../common.sh"

# Update html and vax-stats.txt
python vax-by-state.py skip_figs > state_vax_stats.txt

# Post national vax plots to reddit:
python post-vax-to-reddit.py \
  "${REDDIT_CLIENT_ID}" \
  "${REDDIT_CLIENT_SECRET}" \
  "${REDDIT_PASSWORD}"

# Post per-state vax plots to reddit
python post-state-vax-to-reddit.py \
  "${REDDIT_CLIENT_ID}" \
  "${REDDIT_CLIENT_SECRET}" \
  "${REDDIT_PASSWORD}"

# Commit and push
git commit --all -m "state vax stats update"

# pull first to decrease the chances of a collision. Lockfile ensures this isn't racey
# with respect to the other automation jobs running on this computer, but if we're
# unluckly it could still collide with other pushes to remote.
flock "${LOCKFILE}" -c  "git pull --rebase --autostash; git push"
