# Reproducibility

This chapter lists what is needed to rebuild the system and the commands that
produce each result reported in this book. Because sampling and
initialisation are seeded and the GPU code adds numbers in a fixed order, a
command repeated on the same device produces the same numbers.

## Requirements

The code runs on an NVIDIA Jetson (Orin or Thor) or any other CUDA GPU
supported by cuda-oxide. Building it requires the cuda-oxide toolchain: the
nightly Rust compiler named in `rust-toolchain.toml` and the `cargo-oxide`
tool described in the cuda-oxide README. The datasets occupy about 5 GB on
disk and the prepared features about 3 GB.

## Building and testing

```bash
cargo oxide build -- --release                    # builds target/release/staleynet
cargo test --release -p staleynet-audio           # signal processing on the CPU
cargo oxide test -- --release -p staleynet-cuda   # every GPU kernel against its CPU reference
cargo oxide test -- --release -p staleynet-model  # gradients, cached decoding, memorisation, checkpoints, warm start
cargo oxide test -- --release -p staleynet-data   # GPU spectrogram, tokenizer, batches, lyrics alignment
```

## Obtaining the data

The Song Describer Dataset is published on Zenodo. Its audio carries
per-track Creative Commons licences, listed in `audio_licenses.txt`, and its
captions are licensed CC BY-SA 4.0.

```bash
mkdir -p data/raw/song-describer && cd data/raw/song-describer
for file in song_describer.csv audio.zip; do
  curl -L -o "$file" "https://zenodo.org/records/10072001/files/$file?download=1"
done
unzip -q audio.zip -d audio && cd -
```

JamendoLyrics MultiLang is distributed as a Git repository whose songs and
lyrics carry the Creative Commons licence stated for each song in
`JamendoLyrics.csv`.

```bash
git clone --depth 1 https://github.com/f90/jamendolyrics data/raw/jamendolyrics
```

Recordings made with the studio are stored in `data/recordings/` in the layout
that `staleynet prepare --folder` reads: each audio file is accompanied by a
`.txt` file holding its description and a `.lyrics.txt` file holding the words
sung.

## The pipeline, step by step

```bash
# Spectrograms and tokens of the main dataset, and the tokenizer itself
staleynet prepare  --song-describer data/raw/song-describer --cache data/cache
staleynet tokenize --cache data/cache --listen samples/tokenizer

# The lyrics corpus, encoded with the same tokenizer
staleynet prepare  --jamendo-lyrics data/raw/jamendolyrics --cache data/cache-jamendolyrics
staleynet tokenize --cache data/cache-jamendolyrics --reuse data/cache/tokenizer.json

# Training runs (see configs/ for the settings of each experiment)
staleynet train --cache data/cache --run runs/vocal-scratch --config configs/vocal-scratch.json
staleynet train --run runs/lyrics-orin --config configs/lyrics-orin.json --fine-tune-from runs/vocal-scratch/best

# Analyses
staleynet reconstruct --cache data/cache --out samples/reconstruction
staleynet evaluate    --run runs/lyrics-orin --cache data/cache-jamendolyrics
staleynet bench       --out benchmarks/results.json

# Listening
staleynet sing --run runs/lyrics-orin --prompt "slow grunge song with a raw male voice" \
               --lyrics-file song.txt --seconds 15 --out samples/song.wav
```

A run interrupted at any point continues with `--resume`. The resumed run
restores the optimiser state and draws the same batches it would have drawn
without the interruption.

The single-excerpt memorisation experiment of
[Experiments, Section 5](experiments.md#memorisation-of-a-single-excerpt) uses
`configs/overfit-one-take.json` on a folder containing one excerpt rendered by
`staleynet reconstruct`, and compares the generated tokens with the excerpt's
own tokens through `staleynet sing --save-codes`.

## Where the results are stored

| Result | Location |
|---|---|
| Tokenizer error per track | `data/cache/tokenizer_report.json` |
| Tokenizer listening examples | `samples/tokenizer/` |
| Reconstruction analysis and its audio | `samples/reconstruction/` |
| Memorisation experiment audio and tokens | `samples/overfit/` |
| Training and validation curves | `runs/<run>/metrics.jsonl` |
| Checkpoints | `runs/<run>/latest/`, `runs/<run>/best/` |
| Lyrics evaluation | `runs/<run>/lyrics-evaluation-<split>.json` |
| Generated audio and timing | `samples/*.wav` with a `.json` report beside each |
| Kernel benchmarks | `benchmarks/*.json` |
