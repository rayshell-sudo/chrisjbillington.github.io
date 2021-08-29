#! /bin/bash
set -euxo

# Source our secrets, clone a temporary copy of the repo and cd to it:
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

# Commit and push
git commit --all -m "ACT update"

# pull first to decrease the chances of a collision
git pull --rebase --autostash
git push
