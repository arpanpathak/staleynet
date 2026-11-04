#!/usr/bin/env bash
# Builds the FMA rock cache on the training machine and starts the pq-xl run
# once the run named in WAIT_FOR_RUN has finished.
#
# Run from the project directory, so that the relative data paths of the
# configurations resolve. Expects fma_metadata/, rock.tsv (from select.py) and
# a download of fma_large.zip that writes DOWNLOAD_DONE to download.log.
#
# Settings (environment variables):
#   STALEYNET     the staleynet binary         (default: target/release/staleynet)
#   FMA           folder of the FMA download    (default: data/raw/fma)
#   WAIT_FOR_RUN  run directory to wait for     (default: runs/pq-large)
set -euo pipefail

bin="${STALEYNET:-target/release/staleynet}"
fma="${FMA:-data/raw/fma}"
wait_for="${WAIT_FOR_RUN:-runs/pq-large}"
here="$(cd "$(dirname "$0")" && pwd)"

# Seconds between checks while waiting.
poll_seconds=30

log() { echo "$(date '+%F %T') $*"; }

log "waiting for the download"
until grep -q DOWNLOAD_DONE "$fma/download.log"; do
  sleep "$poll_seconds"
done
log "fma_large.zip sha1 $(sha1sum "$fma/fma_large.zip" | cut -d' ' -f1)"

log "extracting the selected clips"
python3 "$here/arrange.py" "$fma/fma_large.zip" "$fma/rock.tsv" data/fma-rock

log "computing log-mel spectrograms"
"$bin" prepare --folder data/fma-rock --cache data/cache-fma-rock > logs/prepare-fma-rock.log 2>&1

log "tokenising with the band-group tokenizer"
"$bin" tokenize --cache data/cache-fma-rock --reuse data/cache-g4l4f1/tokenizer.json > logs/tokenize-fma-rock.log 2>&1

log "waiting for $wait_for to finish"
while ps -eo args | grep -q "staleynet train .*--run $wait_for "; do
  sleep "$poll_seconds"
done

log "starting pq-xl"
nohup "$bin" train --cache data/cache-g4l4f1 --run runs/pq-xl --config configs/pq-xl-thor.json > logs/train-pq-xl.log 2>&1 &
log "pq-xl started as process $!"
