#!/usr/bin/env bash
# Starts a new language-model run once the lyrics have been merged into the
# rock corpus, so that it trains with them from its first step.
#
# after-transcription.sh writes "resumed" to its log after merging the lyrics
# and resuming the running model; this script waits for that line, then
# starts RUN beside it.
#
# Settings (environment variables):
#   RUN      run to start                        (default: pq-xl)
#   CONFIG   its configuration                   (default: configs/pq-xl-thor.json)
#   PRIMARY  its primary cache                   (default: data/cache-g4l4f1)
#   WAIT_LOG log that announces the merge        (default: logs/after-transcription.log)
set -euo pipefail
cd "$(dirname "$0")/../.."

run="${RUN:-pq-xl}"
config="${CONFIG:-configs/pq-xl-thor.json}"
primary="${PRIMARY:-data/cache-g4l4f1}"
wait_log="${WAIT_LOG:-logs/after-transcription.log}"

# Seconds between checks of the log.
poll_seconds=60

log() { echo "$(date '+%F %T') $*"; }

log "waiting for the lyrics merge (watching $wait_log)"
until grep -q "resumed" "$wait_log" 2>/dev/null; do
  if grep -q "leaving the run untouched" "$wait_log" 2>/dev/null; then
    log "transcription failed; not starting $run"
    exit 1
  fi
  sleep "$poll_seconds"
done

if [ -e "runs/$run" ]; then
  log "runs/$run already exists; not starting it again"
  exit 1
fi

log "starting $run with lyrics"
setsid nohup target/release/staleynet train --cache "$primary" --run "runs/$run" --config "$config" \
  > "logs/train-$run.log" 2>&1 < /dev/null &
log "started (pid $!)"
