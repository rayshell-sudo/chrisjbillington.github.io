#! /bin/bash
set -euxo

# Source our secrets, clone a temporary copy of the repo and cd to it:
source "$(dirname "$BASH_SOURCE")/../common.sh"

# Wait for VIC data to become available:
if python wait-for-vic-update.py | grep "ready!"; then
    ./vic.sh
fi

# Post to reddit:
python post-vic-to-reddit.py \
  "${REDDIT_CLIENT_ID}" \
  "${REDDIT_CLIENT_SECRET}" \
  "${REDDIT_PASSWORD}"

# Commit and push
git commit --all -m "VIC update"

# pull first to decrease the chances of a collision
git pull --rebase --autostash
git push
