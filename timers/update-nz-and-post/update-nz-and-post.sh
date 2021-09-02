#! /bin/bash
set -euxo

# Source our secrets, clone a temporary copy of the repo and cd to it:
source "$(dirname "$BASH_SOURCE")/../common.sh"

# Wait for NZ data to become available:
if python wait-for-nz-update.py | grep "ready!"; then
    ./nz.sh
fi

# Post to reddit:
python post-nz-to-reddit.py \
  "${REDDIT_CLIENT_ID}" \
  "${REDDIT_CLIENT_SECRET}" \
  "${REDDIT_PASSWORD}"

# Commit and push
git commit --all -m "NZ update"

# pull first to decrease the chances of a collision
git pull --rebase --autostash
git push
