# Experiments

## Experimental setup

**Hardware.** Two devices were used, often concurrently with other
workloads:

* NVIDIA Jetson Orin NX 16 GB (Super developer kit), CUDA 12.6;
* NVIDIA Jetson AGX Thor with 128 GB of memory shared by its CPU and GPU. In
  the early runs other GPU services shared the device; the larger models were
  trained with most of the device to themselves.

Unless stated otherwise, figures were measured on the Orin NX.

**Data.** The Song Describer Dataset and JamendoLyrics MultiLang, partitioned
by artist as described in [Training](training.md#data-and-partitions).

**Metrics.** Losses are mean cross-entropies in nats per code (chance level
6.93). Spectral errors are mean absolute differences of natural-log mel
energies; one unit is a factor of e in energy, about 4.3 dB. Throughput is given in
sequence positions per second.

**Procedures.** Every number in this chapter is produced by a command listed
in [Reproducibility](reproducing.md). Raw results are stored in
`benchmarks/`, `runs/*/metrics.jsonl`, `samples/reconstruction/report.json` and
`runs/*/lyrics-evaluation-*.json`.

## 1. Correctness

All 169 tests pass on the Orin NX, and the kernel tests also pass on the Thor.

| Test suite | Scope | Tests |
|---|---|---|
| `staleynet-cuda` | Every kernel against a CPU reference; the GEMM in all transpose, batch and accumulation modes, for all three implementations, on ragged sizes | 40 |
| `staleynet-model` | Gradients against central finite differences for every parameter tensor; cached decoding against the full forward pass (≤ 10⁻⁴), with and without a lyrics field; memorization of a fixed batch; exact checkpoint round trip including optimizer moments; warm start into a model with lyrics; sequence layout and lyrics windowing | 38 |
| `staleynet-data` | GPU log-mel against the rustfft reference (≤ 10⁻² on audible bins); monotone RVQ error; band groups; next-position targets; lyrics cropping and even spreading of untimed lines; weighted sampling; split groups | 31 |
| `staleynet-audio` | STFT/iSTFT round trip ≤ 10⁻⁴; resampler accuracy in band and > 80 dB anti-aliasing; Griffin-Lim preserves the log-mel; every example of the crate's documentation | 45 |
| `staleynet-train`, `staleynet-cli` | Configuration defaults, metrics parsing, paired statistics of the lyrics test | 15 |

## 2. GPU kernels

Matrix products at the shapes used by the first model (a batch of 4 × 512
positions), measured with `staleynet bench` on the Orin NX. Each timing is the
mean of five runs after one warm-up run, and each entry is the median of three
such measurements, because the clock frequency of the device varies noticeably
from run to run. These are the figures of the paper's appendix.

| Product | Shape (M × N × K) | Direct GFLOP/s | Tiled GFLOP/s | Register-blocked GFLOP/s | Blocked / tiled |
|---|---|---|---|---|---|
| Attention projections | 2048 × 768 × 256 | 46.8 | 258.6 | 579.0 | 2.2× |
| Feed-forward | 2048 × 1024 × 256 | 55.3 | 255.5 | 450.6 | 1.8× |
| Output heads | 2048 × 4096 × 256 | 31.2 | 222.1 | 458.5 | 2.1× |
| Spectrogram | 4096 × 513 × 1024 | 47.7 | 236.1 | 541.7 | 2.3× |

A forward and backward training step on 4 × 512 positions takes 2 930 ms with
the direct kernel, 576 ms with the tiled kernel and 315 ms with the
register-blocked kernel: 1.8 times faster than tiling and 9.3 times faster
than the direct kernel.

**On the Thor.** Measured during training runs of identical shape, register
blocking reduced a training step of the first model on 4 × 768 positions from
238 ms to 147 ms, raising throughput from 12 900 to 20 800 positions per
second. A step of the 42.9-million-parameter model on 8 × 1 408 positions
takes about 2.0 seconds.

**Feature extraction.** The log-mel spectrogram of 60 s of audio takes 47 ms
on the GPU and 73 ms with the multithreaded rustfft reference on the Orin's
eight Cortex-A78AE cores. The \\(O(N^2)\\) cost of the DFT-as-GEMM formulation,
about 100 times the arithmetic of an FFT, consumes much of the GPU's advantage
on this device.

## 3. Key/value caching

Sampling 200 audio frames (203 delayed steps) with a randomly initialized
default-size model:

| Attention | ms per step |
|---|---|
| Recomputation of the full context | 17.1 |
| Key/value cache | 6.2 |

The cache is 2.8 times faster at this context length, and the gap grows with
the context, since recomputation repeats the work of every earlier position.

## 4. The tokenizer

The tokenizer was trained on 3.69 million steps from the 667 training tracks
(subsampled to 200 000 vectors) with 20 Lloyd iterations per level. Mean
absolute log-mel error, averaged over tracks, after decoding with the first
*n* codebooks:

| Codebooks used | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Training (667 tracks) | 1.040 | 0.906 | 0.828 | 0.771 |
| Validation (39 tracks, unseen artists) | 1.044 | 0.911 | 0.833 | 0.776 |
| JamendoLyrics validation, reused codebooks | 1.089 | 0.959 | 0.879 | 0.821 |

The validation error lies within 0.7 % of the training error, so the
codebooks generalize to unseen artists, and they transfer to the lyrics corpus
with a modest increase. The absolute error is nevertheless large: 0.77
natural-log units correspond to about 3.4 dB per mel band (one unit is a
factor of e in energy, 10 log₁₀ e ≈ 4.3 dB), and each additional level removes
less error than the previous one (−0.13, −0.08, −0.06).

## 5. The audio pipeline in isolation

Listening to generated samples suggested pitched, melodic content with a
blurred and unintelligible voice. Two experiments were designed to locate the
source of the blur without involving the generative behavior of the model.

### Reconstruction without a model

`staleynet reconstruct` takes 12-second excerpts of held-out vocal tracks and
renders them through the pipeline in stages: the exact log-mel through
Griffin-Lim, the tokens decoded with one codebook, and the tokens decoded with
all four codebooks. The last rendering is the best output that any language
model trained on these tokens could produce. Errors are measured against the
exact log-mel of the excerpt.

| Excerpt (validation) | Griffin-Lim on exact mel | First tokenizer, 1 codebook | First tokenizer, 4 codebooks | First tokenizer through Griffin-Lim | Final tokenizer, 16 codebooks | Final tokenizer through Griffin-Lim |
|---|---|---|---|---|---|---|
| sdd-1014963 (guitar, brass, female voice) | 0.230 | 1.055 | 0.807 | 0.821 | 0.425 | 0.480 |
| sdd-103089 (folk metal, male voice) | 0.152 | 0.973 | 0.707 | 0.703 | 0.353 | 0.378 |
| sdd-1116696 (electronic, vocal sample) | 0.170 | 0.801 | 0.569 | 0.554 | 0.268 | 0.338 |
| sdd-1245191 (bass, piano, wordless vocals) | 0.222 | 0.984 | 0.732 | 0.763 | 0.390 | 0.460 |
| **Mean** | **0.194** | **0.953** | **0.704** | **0.710** | **0.359** | **0.414** |

The final tokenizer, described below, is included for comparison.

![Where the spectral detail is lost: mean error of each stage](figures/paper-errors.svg)

The tokenizer introduces between three and four times the spectral error of
Griffin-Lim, and the end-to-end error of a perfect language model is almost
entirely that of the tokenizer. An error of 0.7 units, about 3 dB per mel
band on average, is consistent with the blurred quality heard in generated
voices, which depend on fine spectral detail (harmonics and formants). The
measure does not capture the phase artifacts of Griffin-Lim, which are audible
in the second rendering although its spectral error is small; the comparison
therefore establishes the tokenizer as the dominant *spectral* limitation,
not as the only audible one. The renderings are in `samples/reconstruction/`.

### Memorization of a single excerpt

To test the language model independently of data scarcity, a fresh
default-size model was trained on a single 12-second excerpt (sdd-103089,
562 frames) with its caption, using batch 4, peak learning rate
10⁻³ and no weight decay (`configs/overfit-one-take.json`).

| Step | Training loss (nats per code) |
|---|---|
| 50 | 2.493 |
| 100 | 0.054 |
| 150 | 0.008 |
| 200 | 0.002 |

The model was then asked to generate 12 seconds from the caption without
guidance, and the sampled codes were compared with the excerpt's codes:

| Decoding | Agreement per codebook | Identical frames from the start |
|---|---|---|
| Greedy | 1.000, 1.000, 1.000, 1.000 | 562 of 562 |
| Top-k 250, seed 1 | 0.996, 1.000, 1.000, 0.996 | 26 |
| Top-k 250, seed 2 | 1.000, 0.998, 0.993, 0.998 | 26 |

Greedy free-running generation reproduces the excerpt token for token, and
top-k sampling departs from it only at isolated tokens without accumulating
error. The architecture, the delay pattern, the key/value-cached decoder and
the sampler are therefore capable of reproducing audio exactly, and a model
that has learned an excerpt produces precisely the tokenizer reconstruction
measured above. Together, the two experiments indicate that improving the
audio representation is a precondition for intelligible singing: further
training of the language model on the present tokens cannot exceed the
fidelity of the reconstruction. The renderings are in `samples/overfit/`.

### A finer tokenizer

Guided by these two experiments, tokenizers were compared that differ in the
number of frames per step, the number of band groups (the 128 mel bands are
divided into groups of neighboring bands, each quantized with its own
codebooks) and the number of residual levels in each group. All use 1 024
prototypes per codebook and were trained on the same songs; the error is the
mean over all 39 held-out songs, in log units per mel band.

| Frames per step | Band groups | Levels | Codes per step | Held-out error |
|---|---|---|---|---|
| 2 | 1 | 4 | 4 | 0.776 |
| 2 | 1 | 8 | 8 | 0.643 |
| 1 | 1 | 4 | 4 | 0.745 |
| 1 | 1 | 8 | 8 | 0.588 |
| 1 | 4 | 2 | 8 | 0.597 |
| 1 | 8 | 1 | 8 | 0.612 |
| 1 | 1 | 16 | 16 | 0.427 |
| 1 | 8 | 2 | 16 | 0.403 |
| 1 | 4 | 4 | 16 | **0.401** |

Quantizing one frame per step in 4 band groups with 4 levels each lowered the
error from 0.776 to 0.401, about 1.7 dB per band. At 8 codes per step,
dividing the bands into groups did not help; with 16 codes it helped a little
(0.427 without groups against 0.401 with them). Most of the improvement
therefore comes from describing each step with more codes, and a smaller part,
about 6 %, from quantizing the band groups separately. On the four vocal
excerpts above, the error of the tokens fell from 0.70 to 0.36 and the
end-to-end error from 0.71 to 0.41. The price is a vocabulary of 16 codebooks
instead of 4.

## 6. Training

### Initial run (Orin NX)

`configs/smoke.json`: the default 7.1-million-parameter model, batch 8 × 512
positions, 1 500 steps (about 6.1 million positions, or 1.7 passes over the
3.69 million training steps), warm-up of 100 steps and cosine decay. Wall-clock
time was 25 minutes at 4 100 positions per second. Validation loss on the 39
held-out tracks:

| Step | Mean | Codebook 0 | Codebook 1 | Codebook 2 | Codebook 3 |
|---|---|---|---|---|---|
| 250 | 5.93 | 4.68 | 6.32 | 6.42 | 6.30 |
| 500 | 5.44 | 3.55 | 5.75 | 6.26 | 6.20 |
| 750 | 5.21 | 3.33 | 5.30 | 6.11 | 6.10 |
| 1 000 | 5.06 | 3.22 | 5.04 | 5.97 | 6.01 |
| 1 250 | 4.98 | 3.15 | 4.90 | 5.88 | 5.96 |
| 1 500 | **4.93** | 3.12 | 4.84 | 5.84 | 5.94 |

The validation loss decreased at every evaluation and remained close to the
training loss, indicating that the run ended under-fitted. Codebook 0 is the
most predictable (3.1 nats, a perplexity of about 23 of 1 024); the finest
codebooks are the least predictable (about 5.9 nats), since they encode
residual detail that is close to noise.

### Extended run (AGX Thor)

`configs/thor.json` (batch 4 × 768, planned for 30 000 steps) ran at 12 900
positions per second, 3.3 times the throughput of the Orin NX, and was stopped
at step 15 625 to free the device for the subsequent experiments. The best
validation loss on the 39 held-out tracks was **4.02 nats at step 15 000**,
against 4.93 for the initial run; with the model and data unchanged, the
improvement is attributable to tenfold longer training.

### Vocal subset: fine-tuning against training from scratch

Two runs were trained on the vocal subset (281 training and 19 validation
tracks):

| Run | Device | Initialization | Schedule | Best validation | At step |
|---|---|---|---|---|---|
| `vocal-finetune` | Thor | extended run | 8 000 steps, peak learning rate 3 × 10⁻⁴ | **4.23** | 250 |
| `vocal-scratch` | Orin NX | random | 6 000 steps, batch 8 × 512, peak 6 × 10⁻⁴ | 4.72 | 5 500 |

These losses are measured on the vocal validation subset and are not
comparable with the 4.02 of the extended run, which was measured on all 39
validation tracks.

Initialization from the extended run is worth about 0.5 nats per code. The
fine-tuned model reached its best validation loss after only 250 steps;
thereafter its validation loss rose to about 4.30 and remained there while its
training loss continued to fall, which indicates that 281 tracks do not
support prolonged fine-tuning at this learning rate. The model trained from
scratch improved until near its final step and ended with a training loss of
3.53 against a validation loss of 4.72, a gap that likewise points to
insufficient data. On informal listening, samples from `vocal-scratch` were
perceived as blurred but melodic voices; this is a single listener's
impression rather than a controlled evaluation.

### A larger model on the finer tokens (AGX Thor)

With the finer tokens, a larger model was trained on the Thor
(`configs/pq-large-thor.json`): 8 blocks of width 512 with 8 heads, 42.9
million parameters, with the lyrics field enabled from the start. It was
trained on all 742 training songs of the combined corpora, with JamendoLyrics
weighted 4-fold and the author's two takes weighted 10-fold, in batches of 8
sequences. A step took 2.0 seconds and about 11 GB of memory. Its best held-out
loss was 5.062 nats per code, averaged over its 16 codebooks, at step 18 500 of
20 000; because the vocabulary differs, this figure is not comparable with the
losses above. Its curves are published on
[Hugging Face](https://huggingface.co/arpanpathak/staleynet-checkpoints).

![pq-large training curves](https://huggingface.co/arpanpathak/staleynet-checkpoints/resolve/main/plots/pq-large.png)

The per-codebook panel shows the coarse-to-fine structure directly: the four
codebooks of the coarsest level are predicted at about 3.0 nats, each finer
level is harder, and the finest approach the uniform guess of 6.93, because
they encode residual detail that is close to noise.

#### The model overfitted

The table compares the training loss, averaged over the 1 000 steps before each
point, with the held-out loss at that point. The training loss is measured with
the prompt dropped from 10 % of the sequences, which raises it slightly, so the
true gap is, if anything, larger.

| Step | Training loss | Held-out loss | Gap |
|---|---|---|---|
| 2 500 | 5.080 | 5.333 | 0.253 |
| 5 000 | 4.714 | 5.178 | 0.464 |
| 10 000 | 4.443 | 5.091 | 0.648 |
| 15 000 | 4.137 | 5.066 | 0.929 |
| 20 000 | 4.091 | 5.066 | 0.975 |

From step 15 000 the held-out loss no longer improved, while the training loss
kept falling: everything the model learned in its last quarter applied only to
its training songs. The gap grew to almost one nat per code, which means the
model assigned to each code of its training songs, as a geometric mean,
\\(e^{0.975} \approx 2.7\\) times the probability it assigned to a code of a song
it had not heard. Listening
agreed: different prompts produced nearly the same song, and a line of lyrics
from a training song recurred. The cause was the sampling weights, under which
each lyric-bearing song was drawn about 650 times and each of the author's
takes more than 1 600 times
([Training](training.md#overfitting-and-how-often-each-song-is-seen)).

#### A balanced run

A second run, `pq-large-rock`, added the 29 314 rock tracks of the Free Music
Archive but kept the heavy weights on the small datasets, and was stopped
after 1 000 steps for that reason. Its replacement, `pq-rock-balanced`
(`configs/pq-rock-balanced-thor.json`), trains the same 42.9-million-parameter
model on 30 054 songs for 30 000 steps, with weights chosen so that no song is
drawn more than about 16 times. It is in progress; its curves are published
alongside those of `pq-large`, and the quantity to watch is the gap between the
two losses.

## 7. Lyrics conditioning

Models with a 256-byte lyrics field were warm-started from the two vocal
models and trained on a mixture of the vocal subset, JamendoLyrics and, after
the first 750 (Orin) and 4 750 (Thor) steps, the rock subset with weight 3.
The use of the lyrics was measured with the paired protocol of
[Lyrics Conditioning](lyrics.md#evaluation-protocol) on 48 batches of four
crops.

**First runs.** With JamendoLyrics at weight 2 (73 training songs against 281
vocal-subset tracks), about one third of training sequences were drawn from
the lyrics corpus. The Thor run (8 000 steps, warm-started from
`vocal-finetune`) gave the following losses at its best checkpoint (step
4 750):

| Split | Aligned | Removed | Mismatched | Mismatched − aligned |
|---|---|---|---|---|
| JamendoLyrics validation (6 songs) | 4.2962 | 4.3031 | 4.2964 | +0.0002 |
| JamendoLyrics training (73 songs) | 3.9728 | 3.9779 | 3.9776 | +0.0048 |

The difference between mismatched and aligned lyrics is negligible on held-out
songs and very small on training songs. The model responds weakly to the
presence of text (the removed condition is slightly worse) but not to its
content: it has not learned to use the lyrics. The likely causes are the small
share of lyric-bearing sequences, the short schedule, and the difficulty of
learning a text-to-audio alignment from 78 songs in four languages.

**Second run.** A Thor run of 20 000 steps with JamendoLyrics at weight 8 and
the register-blocked GEMM (`configs/lyrics-thor-long.json`) overfitted
quickly: its training loss fell to 2.66 while its held-out loss stayed near
4.34, and its best held-out checkpoint is that of step 250. The paired test on
that checkpoint, now reported with the standard error of the paired
difference, gave:

| Songs | Correct | None | Other song | Other minus correct |
|---|---|---|---|---|
| JamendoLyrics held-out (6) | 4.3345 | 4.3409 | 4.3345 | +0.0000 ± 0.0004 |
| JamendoLyrics training (73) | 4.0337 | 4.0395 | 4.0378 | +0.0042 ± 0.0006 |

On held-out songs the correct and the wrong lyrics give the same loss, so the
model does not use the words. On training songs the difference is
statistically clear but small, which points to memorization of those songs
rather than a general relation between words and sound. In both cases the
model scores slightly worse with no lyrics than with any lyrics: it reacts to
the presence of text, not to its content.

## 8. Generation

Three prompts of 12 seconds each, generated from the initial checkpoint with
top-k 250, temperature 1.0, guidance scale 3 and seed 1 on the Orin NX:

| Prompt | Sampling | Griffin-Lim | Real-time factor |
|---|---|---|---|
| upbeat pop song with female vocals and electric guitar | 3.3 s | 1.0 s | 0.36 |
| slow acoustic folk ballad with a male singer and fingerpicked guitar | 3.4 s | 1.0 s | 0.37 |
| energetic electronic dance music with a heavy kick drum | 3.4 s | 1.1 s | 0.37 |

Generation runs about three times faster than real time, including guidance,
which doubles the number of forward passes.

![Spectrograms of the three generated samples](figures/smoke-spectrograms.png)

The spectrograms show rhythmic onsets and, for the pop prompt, sustained
harmonic lines between 300 and 600 Hz. The prompt influences the output: the
share of energy below 250 Hz is 0.57 for the pop prompt, 0.88 for folk and
0.90 for dance. Broadband clicks, visible as vertical stripes, are present in
all three samples; in the light of Section 5 they are attributed in part to
the tokenizer and Griffin-Lim, although the contribution of each factor has
not been isolated. No formal listening test has been conducted, so these
results do not establish perceived quality, musicality or intelligibility.
