#!/usr/bin/env bash
# Trains the neural vocoder on the FMA rock corpus:
#
#   1. writes every track's samples into the cache (resumable);
#   2. waits for the corpus to be tokenized, since training reads its codes;
#   3. trains, resuming from runs/vocoder-rock/latest when it exists.
#
# Run from the project directory. Settings (environment variables):
#   STALEYNET  the staleynet binary  (default: target/release/staleynet)
set -euo pipefail

bin="${STALEYNET:-target/release/staleynet}"
cache=data/cache-fma-rock
run=runs/vocoder-rock

# Seconds between checks while waiting.
poll_seconds=60

log() { echo "$(date '+%F %T') $*"; }

# True while a process whose arguments match the regular expression $1 runs
# (the grep processes themselves are filtered out first).
running() { ps -eo args | grep -v grep | grep -q -- "$1"; }

log "writing waveforms into $cache"
nice "$bin" vocoder prepare --cache "$cache" > logs/vocoder-prepare.log 2>&1
log "waveforms: $(tail -n 1 logs/vocoder-prepare.log)"

log "waiting for the corpus to be tokenized"
while running "staleynet tokenize --cache $cache"; do sleep "$poll_seconds"; done
test -f "$cache/tokenizer_report.json" || { log "no tokenizer report in $cache: tokenizing did not finish"; exit 1; }

log "training $run"
"$bin" vocoder train --cache "$cache" --run "$run" --config configs/vocoder-rock-thor.json >> logs/train-vocoder-rock.log 2>&1
log "$run finished"
