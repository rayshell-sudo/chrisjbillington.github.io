#! /bin/bash
set -euxo

# Source our secrets, clone a temporary copy of the repo and cd to it:
source "$(dirname "$0")/common.sh"

Wait for NSW data to become available:
if python wait-for-nsw-update.py | grep "ready!"; then
    ./nsw.sh
fi

# Post to reddit:
python post-nsw-to-reddit.py \
  "${REDDIT_CLIENT_ID}" \
  "${REDDIT_CLIENT_SECRET}" \
  "${REDDIT_PASSWORD}"

# Commit and push
git commit --all -m "NSW update"

# pull first to decrease the chances of a collision
git pull --rebase --autostash
git push
