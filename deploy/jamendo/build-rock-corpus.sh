#!/usr/bin/env bash
# Builds the MTG-Jamendo rock corpus from scratch: downloads the complete rock
# songs, arranges them into artist folders with tag captions, computes their
# log-mel spectrograms, tokenizes them with the FMA rock tokenizer (so the two
# corpora can be trained on together), and stores the waveforms that the
# Whisper lyrics pass and the vocoder read.
#
# Needs the MTG-Jamendo metadata repository cloned into
# data/raw/mtg-jamendo-dataset and the FMA rock cache's tokenizer.
#
# Run from the project directory. Every step is resumable or idempotent.
set -euo pipefail
cd "$(dirname "$0")/../.."

bin=target/release/staleynet
downloads=data/raw/jamendo-rock
songs=data/jamendo-rock
cache=data/cache-jamendo-rock
tokenizer=data/cache-fma-rock/tokenizer.json

log() { echo "$(date '+%F %T') $*"; }

log "downloading the rock songs"
"$bin" data jamendo-download --out "$downloads"

log "arranging them into artist folders"
"$bin" data jamendo-arrange --downloads "$downloads" --out "$songs"

log "computing log-mel spectrograms of the complete songs"
"$bin" prepare --folder "$songs" --cache "$cache"

log "tokenizing with the FMA rock tokenizer"
"$bin" tokenize --cache "$cache" --reuse "$tokenizer"

log "storing waveforms"
"$bin" vocoder prepare --cache "$cache"

log "ROCK_CORPUS_DONE"
