#!/usr/bin/env bash
# Publishes training runs from the remote Jetson to the Hugging Face checkpoint
# repository, once now and then every PUBLISH_SECONDS: metrics, plots, the run
# configuration, and the best and latest checkpoints.
#
# Run it on the machine that holds the Hugging Face token. Each pass copies
# the runs from the remote machine, plots them with plot_metrics.py, and
# uploads the staging folder; files that have not changed are skipped by the Hub.
#
# Settings (environment variables):
#   THOR_HOST        ssh host of the training machine   (default: thor)
#   THOR_RUNS        runs folder on that machine          (default: Projects/stayleNET/runs)
#   HF_REPO          Hugging Face model repository        (default: arpanpathak/staleynet-checkpoints)
#   RUNS             runs to publish                       (default: "pq-large pq-xl")
#   PUBLISH_SECONDS  seconds between passes; 0 runs once  (default: 7200)
#   STAGING          local staging folder                  (default: ~/.cache/staleynet-publish)
set -uo pipefail

host="${THOR_HOST:-thor}"
remote_runs="${THOR_RUNS:-Projects/stayleNET/runs}"
repo="${HF_REPO:-arpanpathak/staleynet-checkpoints}"
runs="${RUNS:-pq-large pq-xl}"
interval="${PUBLISH_SECONDS:-7200}"
staging="${STAGING:-$HOME/.cache/staleynet-publish}"
here="$(cd "$(dirname "$0")" && pwd)"

log() { echo "$(date '+%F %T') $*"; }

# Copies one run into the staging folder in the repository's layout.
stage_run() {
  local run="$1"
  ssh "$host" "test -f $remote_runs/$run/metrics.jsonl" || return 1

  mkdir -p "$staging/checkpoints/$run" "$staging/metrics" "$staging/run-configs" "$staging/plots"
  rsync -a "$host:$remote_runs/$run/metrics.jsonl" "$staging/metrics/$run.jsonl"
  rsync -a "$host:$remote_runs/$run/config.json" "$staging/run-configs/$run.json" 2>/dev/null
  for checkpoint in best latest; do
    rsync -a --delete "$host:$remote_runs/$run/$checkpoint/" "$staging/checkpoints/$run/$checkpoint/" 2>/dev/null
  done

  python3 "$here/plot_metrics.py" "$staging/metrics/$run.jsonl" "$staging/plots/$run" 2>/dev/null
}

publish_once() {
  local staged=()
  for run in $runs; do
    stage_run "$run" && staged+=("$run")
  done
  [ ${#staged[@]} -gt 0 ] || { log "no runs to publish"; return; }

  local summary
  summary=$(for run in "${staged[@]}"; do python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print(f\"{s['run']} step {s['latest_step']}, validation {s['best_validation_loss']}\")" "$staging/plots/$run.json"; done | paste -sd ';')
  log "uploading: $summary"
  hf upload "$repo" "$staging" . --repo-type model --commit-message "Training progress: $summary" > /dev/null && log "uploaded"
}

while true; do
  publish_once
  [ "$interval" -gt 0 ] || break
  sleep "$interval"
done
