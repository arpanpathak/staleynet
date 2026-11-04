# staleynet-audio

Audio decoding, resampling, log-mel spectrograms and Griffin-Lim reconstruction
in pure Rust, on the CPU.

This crate is the audio layer of [StaleyNET](https://github.com/arpanpathak/staleynet),
a small research system that generates music with singing from a text
description. StaleyNET never works with raw waveforms directly: it turns every
recording into a log-mel spectrogram, learns to predict spectrograms, and turns
the prediction back into sound. This crate performs both conversions, and it is
also the reference against which StaleyNET's GPU feature extractor is checked.
It has no GPU dependency and can be used on its own in any Rust program that
needs to look at, or listen to, the frequency content of audio.

```text
 audio file ──► decode_mono ──► resample ──► MelAnalyzer::log_mel ──► spectrogram
 (mp3, flac,     mono samples    24 kHz       128 bands, about          (what a
  wav, ogg ...)  at any rate                  94 frames per second       model sees)

 spectrogram ──► griffin_lim ──► write_mono_pcm16 ──► out.wav
                 recovers the phase the spectrogram does not store
```

## Install

```bash
cargo add staleynet-audio
```

The crate depends only on pure-Rust libraries: Symphonia for decoding, RustFFT
for the Fourier transforms, rayon for parallelism and hound for WAV output.

## A first example: a note, its picture, and back again

The example below builds one second of the note A4 (440 Hz) at the CD sample
rate, converts it to the analysis rate of 24 kHz, computes its spectrogram, and
reconstructs a waveform from the spectrogram alone. It needs no input file.

```rust
use staleynet_audio::griffin_lim::{GriffinLimConfig, griffin_lim};
use staleynet_audio::resample::resample;
use staleynet_audio::wav::write_mono_pcm16;
use staleynet_audio::{MelAnalyzer, MelConfig};
use std::f32::consts::TAU;

fn main() -> anyhow::Result<()> {
    // One second of A4 at 44,100 samples per second, the rate of a CD.
    let cd_rate = 44_100;
    let note: Vec<f32> = (0..cd_rate)
        .map(|n| 0.5 * (TAU * 440.0 * n as f32 / cd_rate as f32).sin())
        .collect();

    // The analysis expects 24,000 samples per second.
    let samples = resample(&note, cd_rate as u32, 24_000);
    assert_eq!(samples.len(), 24_000);

    // One spectrogram frame every 256 samples gives 1 + 24,000 / 256 = 94
    // frames, and each frame holds 128 band energies.
    let analyzer = MelAnalyzer::new(MelConfig::default());
    let log_mel = analyzer.log_mel(&samples);
    assert_eq!(log_mel.len(), 94 * 128);

    // The spectrogram keeps how strong each frequency is, but not its phase.
    // Griffin-Lim searches for a phase that is consistent with the magnitudes.
    let settings = GriffinLimConfig { iterations: 64, seed: 7 };
    let waveform = griffin_lim(&analyzer, &log_mel, settings);
    write_mono_pcm16(&std::env::temp_dir().join("a4.wav"), &waveform, 24_000)?;
    Ok(())
}
```

## The ideas behind each step

### Decoding

Audio files store sound in many containers and codecs. `decode_mono` opens any
file whose extension is listed in `decode::SUPPORTED_EXTENSIONS` (WAV, MP3,
FLAC, M4A, AAC, OGG and MP4), decodes it with Symphonia, and averages the
channels into one. The result is a `MonoAudio`: a list of samples between −1
and 1, together with the rate at which they were recorded.

### Resampling

Recordings arrive at 44.1 kHz, 48 kHz and other rates, while the analysis
assumes 24 kHz. Changing the rate means estimating the signal at instants that
fall between the original samples. The ideal estimate is a weighted sum of the
nearby samples, with weights given by the sinc function. `resample` truncates
the sinc smoothly with a Kaiser window and precomputes the weights for every
fractional offset that the two rates can produce (a polyphase filter bank), so
that each output sample costs one short dot product.

When the rate is lowered, frequencies above half the new rate (the Nyquist
frequency, 12 kHz at 24 kHz) can no longer be represented, and if they were
kept they would fold back as false tones. The filter therefore removes them
first.

### The log-mel spectrogram

A spectrogram describes sound the way a musical score does: for each short
moment, which frequencies are present and how strongly. `MelAnalyzer::log_mel`
computes it in three steps.

1. **Short-time Fourier transform.** The signal is cut into overlapping
   windows of 1024 samples (42.7 ms), one starting every 256 samples (10.7 ms),
   and each window is decomposed into 513 frequencies spaced 23.4 Hz apart.
2. **Mel bands.** Hearing distinguishes low pitches finely and high pitches
   coarsely, so the 513 frequencies are pooled into 128 overlapping triangular
   bands that are narrow at low frequencies and wide at high ones (the Slaney
   mel scale, linear below 1 kHz and logarithmic above it).
3. **Logarithm.** Loudness is perceived roughly logarithmically, so each band
   energy is replaced by its natural logarithm, with a floor of 10⁻⁵ for
   silence.

The default settings, in `MelConfig::default()`, are those of the Vocos
vocoder except for the number of bands:

| Setting | Value | What it means |
|---|---|---|
| `sample_rate` | 24,000 Hz | frequencies up to 12 kHz can be represented |
| `fft_size` | 1024 samples | each frame looks at 42.7 ms of sound |
| `hop_size` | 256 samples | a new frame every 10.7 ms, 93.75 per second |
| `mel_bands` | 128 | the frequency axis has 128 perceptual steps |
| `min_frequency`, `max_frequency` | 0 Hz to 12 kHz | the full range at this rate |

### Griffin-Lim reconstruction

A spectrogram records the strength of each frequency but not its phase, that
is, where in its cycle each wave is at each moment. `griffin_lim` starts from
random phases and repeats two steps: build the waveform that the current
spectrum describes, then analyse that waveform again and keep its phases while
restoring the target strengths. The fast variant of Perraudin et al. adds
momentum (0.99), which reaches a good estimate in far fewer iterations. The
result is recognisable but slightly "phasey"; a neural vocoder sounds cleaner.

## More examples

### A higher note lights up a higher band

Below 1 kHz the mel bands are evenly spaced, so doubling the pitch roughly
doubles the band index. The example finds the loudest band of a note and the
frequency at which that band is centred.

```rust
use staleynet_audio::{MelAnalyzer, MelConfig};
use std::f32::consts::TAU;

/// The loudest band in frame 40 of a one-second note, and its centre in Hz.
fn loudest_band(analyzer: &MelAnalyzer, frequency: f32) -> (usize, f32) {
    let config = analyzer.config;
    let note: Vec<f32> = (0..config.sample_rate)
        .map(|n| 0.5 * (TAU * frequency * n as f32 / config.sample_rate as f32).sin())
        .collect();

    let bands = config.mel_bands;
    let log_mel = analyzer.log_mel(&note);
    let frame = &log_mel[40 * bands..41 * bands];
    let band = (0..bands).max_by(|&a, &b| frame[a].total_cmp(&frame[b])).unwrap();

    // The filterbank is stored as [frequency bin][band]; a band's centre is the
    // bin where its triangle peaks.
    let weight = |bin: usize| analyzer.filterbank[bin * bands + band];
    let peak_bin = (0..config.frequency_bins()).max_by(|&a, &b| weight(a).total_cmp(&weight(b))).unwrap();
    let hz_per_bin = config.sample_rate as f32 / config.fft_size as f32;
    (band, peak_bin as f32 * hz_per_bin)
}

let analyzer = MelAnalyzer::new(MelConfig::default());

let (a4_band, a4_centre) = loudest_band(&analyzer, 440.0);
let (a5_band, _) = loudest_band(&analyzer, 880.0);
assert_eq!((a4_band, a5_band), (16, 32)); // one octave up, twice the index
assert!((a4_centre - 440.0).abs() < 10.0); // band 16 is centred at 445.3 Hz
```

### The spectrogram as a matrix product

Each frequency of a frame is a weighted sum of the frame's 1024 samples, so a
whole frame is one row of samples multiplied by a fixed matrix.
`windowed_dft_basis` returns that matrix (one for the real part and one for
the imaginary part). StaleyNET's GPU path computes spectrograms this way,
because GPUs are fastest at matrix products; the example checks that the two
methods agree.

```rust
use staleynet_audio::{MelAnalyzer, MelConfig};

let analyzer = MelAnalyzer::new(MelConfig::default());
let signal: Vec<f32> = (0..24_000).map(|n| (n as f32 * 0.05).sin()).collect();

let spectrum = analyzer.stft(&signal);
let basis = analyzer.windowed_dft_basis();
let padded = analyzer.reflect_pad(&signal);

let (frame, fft_size, bins) = (10, 1024, analyzer.config.frequency_bins());
let window = &padded[frame * 256..frame * 256 + fft_size];

for bin in 0..bins {
    let real: f32 = (0..fft_size).map(|n| window[n] * basis.real[n * bins + bin]).sum();
    let imaginary: f32 = (0..fft_size).map(|n| window[n] * basis.imaginary[n * bins + bin]).sum();
    assert!((real - spectrum[frame][bin].re).abs() < 1e-3);
    assert!((imaginary - spectrum[frame][bin].im).abs() < 1e-3);
}
```

### Measuring how much reconstruction loses

A practical way to judge a reconstruction is to analyse it again and compare
the two spectrograms. For a two-second chord the mean difference after 64
iterations is about 0.44 in natural-log units, an energy ratio of about
1.9 dB. The waveform is 128 samples shorter than the input, because the last
frame describes only the first half of its window.

```rust
use staleynet_audio::griffin_lim::{GriffinLimConfig, griffin_lim};
use staleynet_audio::{MelAnalyzer, MelConfig};
use std::f32::consts::TAU;

let analyzer = MelAnalyzer::new(MelConfig::default());

// An A major chord (A3, C♯4, E4) for two seconds.
let chord: Vec<f32> = (0..48_000)
    .map(|n| {
        let time = n as f32 / 24_000.0;
        0.3 * (TAU * 220.0 * time).sin() + 0.2 * (TAU * 277.2 * time).sin() + 0.2 * (TAU * 329.6 * time).sin()
    })
    .collect();

let original = analyzer.log_mel(&chord);
let waveform = griffin_lim(&analyzer, &original, GriffinLimConfig { iterations: 64, seed: 7 });
let again = analyzer.log_mel(&waveform);

let compared = original.len().min(again.len());
let mean_error = (0..compared).map(|i| (original[i] - again[i]).abs()).sum::<f32>() / compared as f32;
assert!(mean_error < 0.5);
assert_eq!(waveform.len(), 47_872);
```

### Analysing a folder of songs

For real recordings, decode each file and resample it before analysis.

```rust,no_run
use staleynet_audio::decode::is_audio_file;
use staleynet_audio::resample::resample;
use staleynet_audio::{MelAnalyzer, MelConfig, decode_mono};

fn main() -> anyhow::Result<()> {
    let analyzer = MelAnalyzer::new(MelConfig::default());
    let rate = analyzer.config.sample_rate;

    for entry in std::fs::read_dir("songs")? {
        let path = entry?.path();
        if !is_audio_file(&path) {
            continue;
        }

        let audio = decode_mono(&path)?;
        let samples = resample(&audio.samples, audio.sample_rate, rate);
        let frames = analyzer.log_mel(&samples).len() / analyzer.config.mel_bands;
        println!("{}: {:.1} s, {frames} frames", path.display(), samples.len() as f32 / rate as f32);
    }
    Ok(())
}
```

## Modules

| Module | What it does |
|---|---|
| `decode` | opens audio files with Symphonia and mixes them to mono |
| `resample` | changes the sample rate with a Kaiser-windowed polyphase filter |
| `mel` | short-time Fourier transform, Slaney mel filterbank, log-mel, inverse transform, DFT matrix |
| `griffin_lim` | fast Griffin-Lim phase reconstruction from a log-mel spectrogram |
| `wav` | 16-bit mono WAV output, scaled down only when the peak would clip |

## Quality

- 40 unit tests, with 99.1 percent line coverage, and every example on this
  page runs as a documentation test.
- Griffin-Lim output is deterministic for a fixed seed.
- Every constant is named and documented with its source.
- The CPU spectrogram is the reference for StaleyNET's GPU kernels, which are
  tested against it.

## How StaleyNET uses this crate

StaleyNET learns music from about 23 hours of Creative Commons songs (the Song
Describer Dataset) and from JamendoLyrics, whose songs come with timed lyrics.
Every song passes through this crate once: decoded, resampled to 24 kHz and
analysed into a log-mel spectrogram. A residual k-means tokenizer then turns
each spectrogram frame into a short list of integer codes, and a decoder-only
Transformer, written in Rust with GPU kernels compiled by NVIDIA's cuda-oxide,
learns to predict those codes from a text description and optional lyrics.
At generation time the predicted codes become a spectrogram again, and
`griffin_lim` turns it into sound. All training ran on two NVIDIA Jetson
computers, an Orin NX with 16 GB of memory and an AGX Thor with 128 GB.

## Resources

StaleyNET:

- Source code: <https://github.com/arpanpathak/staleynet>
- Paper, single column: <https://github.com/arpanpathak/staleynet/blob/main/paper/preprint.pdf>
- Paper, IEEE two-column: <https://github.com/arpanpathak/staleynet/blob/main/paper/ieee.pdf>
- Book chapter on audio representation: <https://github.com/arpanpathak/staleynet/blob/main/book/src/audio.md>
- Book chapter on tokenization: <https://github.com/arpanpathak/staleynet/blob/main/book/src/tokenizer.md>
- Book chapter on GPU implementation: <https://github.com/arpanpathak/staleynet/blob/main/book/src/gpu.md>
- API documentation: <https://docs.rs/staleynet-audio>
- Checkpoints and derived data are kept in Hugging Face repositories
  (`arpanpathak/staleynet-checkpoints` and `arpanpathak/staleynet-data`),
  which are private and available on request.

Background reading:

- J. O. Smith, *Digital Audio Resampling Home Page*, CCRMA, Stanford University:
  <https://ccrma.stanford.edu/~jos/resample/>
- J. O. Smith, *Spectral Audio Signal Processing*, CCRMA, Stanford University:
  <https://ccrma.stanford.edu/~jos/sasp/>
- D. Griffin and J. Lim, "Signal estimation from modified short-time Fourier
  transform", IEEE Transactions on Acoustics, Speech, and Signal Processing, 1984:
  <https://doi.org/10.1109/TASSP.1984.1164317>
- N. Perraudin, P. Balazs and P. L. Søndergaard, "A fast Griffin-Lim
  algorithm", WASPAA 2013: <https://doi.org/10.1109/WASPAA.2013.6701851>
- H. Siuzdak, "Vocos: Closing the gap between time-domain and Fourier-based
  neural vocoders", 2023: <https://arxiv.org/abs/2306.00814>
- The mel filterbank convention reproduced here, from librosa:
  <https://librosa.org/doc/latest/generated/librosa.filters.mel.html>

## License

Apache-2.0. See [LICENSE](https://github.com/arpanpathak/staleynet/blob/main/LICENSE).
