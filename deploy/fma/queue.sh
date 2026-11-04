#!/usr/bin/env bash
# Finishes the FMA rock corpus and runs the training queue on it:
#
#   1. waits for a running `staleynet prepare` of the corpus, if any;
#   2. tokenizes the corpus with the band-group tokenizer;
#   3. waits for the run named in WAIT_FOR_RUN to finish;
#   4. trains pq-large-rock, warm-started from that run's best checkpoint;
#   5. starts pq-xl.
#
# Run from the project directory. Each step logs a line; a failed step stops
# the queue rather than training on incomplete data.
#
# Settings (environment variables):
#   STALEYNET     the staleynet binary       (default: target/release/staleynet)
#   WAIT_FOR_RUN  run to wait for and start from  (default: runs/pq-large)
set -euo pipefail

bin="${STALEYNET:-target/release/staleynet}"
base="${WAIT_FOR_RUN:-runs/pq-large}"
cache=data/cache-fma-rock

# Seconds between checks while waiting.
poll_seconds=60

log() { echo "$(date '+%F %T') $*"; }

# True while a process whose arguments match the regular expression $1 runs.
# The grep processes themselves carry the pattern in their arguments, so they
# are filtered out first.
running() { ps -eo args | grep -v grep | grep -q -- "$1"; }

log "waiting for the spectrograms of the corpus"
while running "staleynet prepare --folder data/fma-rock"; do sleep "$poll_seconds"; done
test -f "$cache/manifest.json" || { log "no manifest in $cache: prepare did not finish"; exit 1; }
log "prepared: $(python3 -c "import json; print(len(json.load(open('$cache/manifest.json'))['tracks']))") tracks"

if running "staleynet tokenize --cache $cache"; then
  log "waiting for the tokenizer already running on the corpus"
  while running "staleynet tokenize --cache $cache"; do sleep "$poll_seconds"; done
else
  log "tokenizing with the band-group tokenizer"
  "$bin" tokenize --cache "$cache" --reuse data/cache-g4l4f1/tokenizer.json > logs/tokenize-fma-rock.log 2>&1
fi
test -f "$cache/tokenizer_report.json" || { log "no tokenizer report in $cache: tokenizing did not finish"; exit 1; }
log "tokenized: $(tail -n 2 logs/tokenize-fma-rock.log | tr '\n' ' ')"

log "waiting for $base to finish"
while running "staleynet train .*--run $base "; do sleep "$poll_seconds"; done

log "training pq-large-rock from $base/best"
"$bin" train --cache data/cache-g4l4f1 --run runs/pq-large-rock --config configs/pq-large-rock-thor.json \
  --fine-tune-from "$base/best" > logs/train-pq-large-rock.log 2>&1
log "pq-large-rock finished"

log "starting pq-xl"
nohup "$bin" train --cache data/cache-g4l4f1 --run runs/pq-xl --config configs/pq-xl-thor.json > logs/train-pq-xl.log 2>&1 &
log "pq-xl started as process $!"
