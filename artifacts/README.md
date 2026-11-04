# Artifacts

This directory is the GitHub-side index of StaleyNET's trained artifacts. The
large binaries live in public Hugging Face repositories; this directory keeps
the run metadata, logs, checksums, tokenizer, and sample reports that are small
enough to version with the code.

## Where everything lives

| Artifact | Location | Visibility |
|---|---|---|
| Code, paper, book, configs | <https://github.com/arpanpathak/staleynet> | public |
| Checkpoints, optimizer states, metrics, training logs | <https://huggingface.co/arpanpathak/staleynet-checkpoints> | public |
| Derived caches, tokenizer, samples, own recordings | <https://huggingface.co/datasets/arpanpathak/staleynet-data> | public |

The Hugging Face repositories carry `SHA256SUMS`, `LICENSE`, `NOTICE`, and a
card describing the layout and the upstream dataset terms.

## Checkpoints

| Run | `best` step | `latest` step | Parameters | Training tracks |
|---|---:|---:|---:|---:|
| `thor-base` | 0 | — | 7,099,904 | 667 |
| `smoke` | 0 | 1,500 | 7,099,904 | 667 |
| `overfit-one-take` | 200 | 200 | 7,099,904 | 1 |
| `vocal-scratch` | 5,500 | 6,000 | 7,099,904 | 281 |
| `vocal-finetune` | 250 | — | 7,099,904 | 281 |
| `lyrics-orin` | 3,750 | 4,000 | 7,165,440 | 543 (73 with lyrics) |
| `lyrics-thor` | 4,750 | 8,000 | 7,165,440 | 543 (73 with lyrics) |
| `lyrics-thor-long` | 19,750 | 20,000 | 7,165,440 | 543 (73 with lyrics) |
| `pq-large` | 7,000 | 7,000 | 42,858,496 | 742 (75 with lyrics) |

`best/` is weight-only; `latest/` also stores the AdamW moments and is the
checkpoint to resume from. Step values are read from each `model.json`.

`pq-large` is a snapshot of a run that is still training on the AGX Thor
(step 7,000 of 20,000). It uses the band-group tokenizer (4 groups × 4 levels,
16 codebooks), so its losses cannot be compared with the 4-codebook runs.

`runs/<run>/` here contains the small metadata: `config.json` where the run
saved one, `best-model.json`, `latest-model.json`, and `metrics.jsonl`. The
matching `tensors.bin` files are on Hugging Face.

## Data

The dataset repository contains:

- `cache/` — SDD log-mel and RVQ codes (1,415 files, 2.0 GB).
- `cache-jamendolyrics/` — JamendoLyrics cache (161 files, 408 MB).
- `cache-g4l4f1/` — SDD codes under the band-group tokenizer (708 files,
  246 MB).
- `cache-pq-jamendolyrics/`, `cache-pq-recordings/` — JamendoLyrics and the
  author's own takes under the same tokenizer (82 and 5 files, 62 MB).
- `cache-l16f1/` — tokenizer-study control with 16 residual levels and no band
  groups (709 files, 263 MB).
- These four caches reuse the `mel/` of their source cache, which is omitted
  here; the dataset card lists the symlinks to recreate.
- `cache-overfit/`, `cache-recordings/` — small debugging caches.
- `recordings/`, `overfit/` — the author's own takes; all rights reserved.
- `samples/` — generated and reconstruction audio (79 files, 34 MB).
- `raw/` — SDD and JamendoLyrics metadata/licence tables only; no raw audio.

The raw third-party audio is **not** redistributed. `raw/README.md` in the
dataset repository records the download and rebuild commands, and the code
README's `prepare`/`tokenize` steps reproduce the caches.

## Hydrate a working tree

From the repository root:

```bash
hf download arpanpathak/staleynet-data \
  --repo-type dataset --local-dir data

hf download arpanpathak/staleynet-checkpoints \
  --repo-type model --local-dir runs
```

The data lands at `data/cache`, `data/cache-jamendolyrics`, and so on. The
checkpoints land at `runs/checkpoints/<run>/{best,latest}`. Point the CLI at
that path directly, for example:

```bash
staleynet sing \
  --cache data/cache \
  --run runs/checkpoints/thor-base/best \
  --prompt "slow acoustic ballad, warm female voice" \
  --out song.wav
```

Recreate the omitted `cache-g4l4f1/mel` symlink with:

```bash
ln -s ../cache/mel data/cache-g4l4f1/mel
```

## Integrity

`checksums/checkpoints.sha256` and `checksums/data.sha256` are the exact
`SHA256SUMS` files published alongside the Hugging Face artifacts. Run them
from the root of the downloaded HF repository:

```bash
cd staleynet-checkpoints && sha256sum -c SHA256SUMS
cd ../staleynet-data && sha256sum -c SHA256SUMS
```

The copies here are for auditing without downloading the binaries.

## Licensing

- Code and scripts: Apache-2.0 (`LICENSE`).
- Paper and book: CC BY 4.0 (`LICENSE-CC-BY-4.0.txt`).
- Checkpoints, caches, and generated audio: CC BY-SA 4.0 to the extent the
  upstream dataset licences allow (`LICENSE-CC-BY-SA-4.0.txt`).
- The author's own recordings (`recordings/`, `overfit/`, `cache-recordings/`)
  are not licensed for reuse; all rights reserved.

SDD and JamendoLyrics include CC BY-NC-ND and CC BY-NC-SA recordings. Treat the
checkpoints as research artifacts and do not use them commercially where the
upstream licence forbids it. See `NOTICE` and the HF cards for details.
