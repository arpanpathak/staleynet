#!/usr/bin/env bash
# Copies a remote run's best checkpoint, metrics and tokenizer to this
# machine, so that it can be sampled here (for example from the studio).
#
#   deploy/thor/pull-run.sh <run-name>
#
# Settings: THOR_HOST and THOR_PROJECT, as for sync.sh.
set -euo pipefail

host="${THOR_HOST:-thor}"
project="${THOR_PROJECT:-Projects/staleynet}"
run="${1:?usage: pull-run.sh <run-name>}"
root="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "$root/runs/$run"
rsync -az "$host:$project/runs/$run/best" "$host:$project/runs/$run/metrics.jsonl" "$host:$project/runs/$run/tokenizer.json" "$root/runs/$run/"
grep '"improved":true' "$root/runs/$run/metrics.jsonl" | tail -1
