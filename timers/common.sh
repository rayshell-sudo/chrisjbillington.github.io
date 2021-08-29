#! /bin/bash
set -euxo pipefail

# chdir to the main repo directory:
cd "$(dirname "$BASH_SOURCE")/.."

# Get our secrets
source ../secrets.sh

# Make a clean git repo in /tmp to avoid collisions with other scripts running on the
# same repo:
scratch=$(mktemp -d -t tmp.XXXXXXXXXX)

# Ensure it's deleted when we're done
function finish {
  rm -rf "$scratch"
}

trap finish EXIT

# Sync with any remote changes. The way this works is that the main repo only ever pulls
# from remote, our temporary clones are the ones pushing to remote. We clone from the
# local one to ensure we include any local (but committed) changes that might be being
# tested. The individual jobs can't just work on the same git repo because that would
# create a collision. There is still a risk of collision if two of them push at the same
# time, but it's much smaller since they all issue a 'pull --rebase' immediately prior.
git pull

# clone, cd, and set remote. --depth 1 is important because the repository is huge, so
# even though it's a local copy, we don't wan /tmp to fill up - it's a RAMdisk so can't
# fit much.
git clone --depth 1 .git "$scratch"
cd "$scratch"
git remote set-url origin git@github.com:chrisjbillington/chrisjbillington.github.io
