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

# For ensuring multiple jobs don't operate on the git repo simultaneously
LOCKFILE="/tmp/chrisbillington.net.git.lock"

# The way this works is that the main repo only ever pulls from remote, our temporary
# clones are the ones pushing to remote. We clone from the local one instead of remote
# to ensure we include any local (but committed) changes that might be being tested.
# Each job has its own copy of the repo so jobs can work simultaneously, and we lock any
# access to the main repo or the remote repo to prevent collisions.

# pull, clone, cd, and set remote. --depth 1 is important because the repository is
# huge, so even though it's a local copy, we don't want /tmp to fill up - it's a RAMdisk
# so can't fit much.
flock "${LOCKFILE}" -c "git pull; git clone --depth 1 \"file://${PWD}/.git\" \"${scratch}\""
cd "${scratch}"
git remote set-url origin git@github.com:chrisjbillington/chrisjbillington.github.io
