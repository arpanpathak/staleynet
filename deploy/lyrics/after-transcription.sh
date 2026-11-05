#!/usr/bin/env bash
# Runs on the training machine after the transcription container starts:
# waits for it to finish, writes the lyrics into the rock cache, and resumes
# the language-model run so that its next batches include them.
#
# Settings (environment variables):
#   CONTAINER  transcription container name   (default: lyrics-fma)
#   CACHE      cache that receives the lyrics  (default: data/cache-fma-rock)
#   RUN        run to resume                    (default: pq-rock-balanced)
#   CONFIG     its configuration                (default: configs/pq-rock-balanced-thor.json)
#   PRIMARY    its primary cache                (default: data/cache-g4l4f1)
set -euo pipefail
cd "$(dirname "$0")/../.."

container="${CONTAINER:-lyrics-fma}"
cache="${CACHE:-data/cache-fma-rock}"
run="${RUN:-pq-rock-balanced}"
config="${CONFIG:-configs/pq-rock-balanced-thor.json}"
primary="${PRIMARY:-data/cache-g4l4f1}"
transcripts="data/fma-lyrics/transcripts.jsonl"

log() { echo "$(date '+%F %T') $*"; }

log "waiting for $container"
status=$(docker wait "$container")
if [ "$status" != "0" ]; then
  log "transcription exited with $status; leaving the run untouched"
  exit 1
fi

log "transcribed $(wc -l < "$transcripts") clips"
target/release/staleynet data merge-lyrics --cache "$cache" --transcripts "$transcripts"

# Stop the run by its exact arguments, then resume it from its latest checkpoint.
pattern="target/release/staleynet train --cache $primary --run runs/$run "
pid=$(pgrep -f "^$pattern" | head -1 || true)
if [ -n "$pid" ]; then
  log "stopping $run (pid $pid)"
  kill "$pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 1; done
fi

log "resuming $run with lyrics"
setsid nohup target/release/staleynet train --cache "$primary" --run "runs/$run" --config "$config" --resume \
  >> "logs/train-$run.log" 2>&1 < /dev/null &
log "resumed (pid $!)"
