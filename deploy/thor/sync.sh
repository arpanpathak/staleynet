#!/usr/bin/env bash
# Copies the project and the tokenised datasets to a remote Jetson (by default
# the AGX Thor reachable as `thor`) and builds the release binary there.
#
# Raw audio and log-mel features stay on this machine: training needs only each
# cache's manifest.json, tokenizer.json and codes/. Runs, logs and samples on
# the remote machine are never touched.
#
# Files under data/ are protected from --delete. A cache that was tokenised on
# the remote machine and does not exist here would otherwise be emptied by the
# next sync, which is easy to miss while a running job still holds the codes in
# memory. Files that exist on both sides are still updated.
#
# Settings (environment variables):
#   THOR_HOST     ssh host name                  (default: thor)
#   THOR_PROJECT  project directory on the host  (default: Projects/staleynet)
set -euo pipefail

host="${THOR_HOST:-thor}"
project="${THOR_PROJECT:-Projects/staleynet}"
root="$(cd "$(dirname "$0")/../.." && pwd)"

rsync -az --delete --filter "protect data/***" \
  --exclude target --exclude .git --exclude .cargo --exclude .oxide-artifacts \
  --exclude data/raw --exclude "data/cache*/mel" --exclude data/recordings \
  --exclude studio/target --exclude studio/certs \
  --exclude logs --exclude runs --exclude samples --exclude book/book \
  "$root/" "$host:$project/"

# rust-toolchain.toml selects the nightly that cuda-oxide requires; rustup
# installs it on first use.
ssh "$host" "export PATH=\$HOME/.cargo/bin:\$PATH && cd $project && cargo oxide build -- --release"
