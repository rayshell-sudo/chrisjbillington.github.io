#! /bin/bash
set -euxo pipefail

# chdir to the main repo directory:
cd "$(dirname "$0")/.."

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
# from remote, our temporary clones are the ones pushing to remote. We would just clone
# the entire repo from remote to a tempdir for each job, but it's actually very large so
# we use the local one as an optimisation.
git pull --rebase --autostash

# clone, cd, and set remote
git clone .git "$scratch"
cd "$scratch"
git remote set-url origin git@github.com:chrisjbillington/chrisjbillington.github.io
