# staleynet-audio

Audio input, analysis, and waveform reconstruction for [StaleyNET](https://github.com/arpanpathak/staleynet).

This crate is the CPU audio layer used by StaleyNET. It decodes compressed and uncompressed audio, resamples it to the model rate, computes log-mel spectrograms, and reconstructs waveforms with Griffin-Lim. It has no GPU dependency.

## Install

```bash
cargo add staleynet-audio
```

## Quick example

```rust
use staleynet_audio::griffin_lim::{griffin_lim, GriffinLimConfig};
use staleynet_audio::resample::resample;
use staleynet_audio::{decode_mono, MelAnalyzer, MelConfig};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let audio = decode_mono(Path::new("song.mp3"))?;
    let samples = resample(&audio.samples, audio.sample_rate, 24_000);
    let analyzer = MelAnalyzer::new(MelConfig::default());
    let log_mel = analyzer.log_mel(&samples);
    let waveform = griffin_lim(&analyzer, &log_mel, GriffinLimConfig { iterations: 64, seed: 7 });

    staleynet_audio::wav::write_mono_pcm16(Path::new("out.wav"), &waveform, 24_000)?;
    Ok(())
}
```

The default [`MelConfig`] uses 24 kHz audio, a 1024-point FFT, hop 256, and 128 mel bands. Resample first when the input rate differs.

## Modules

| Module | Responsibility |
|---|---|
| `decode` | Symphonia decoding and mono downmix |
| `resample` | Kaiser-windowed polyphase resampling |
| `mel` | STFT, Slaney mel filterbank, log-mel, inverse STFT |
| `griffin_lim` | Fast Griffin-Lim phase reconstruction |
| `wav` | 16-bit mono PCM output |

## Quality

- 40 unit tests, with 99.1 percent line coverage.
- Deterministic Griffin-Lim output for a fixed seed.
- CPU only, no CUDA or model weights required.
- Apache-2.0.

## Links

- Repository: <https://github.com/arpanpathak/staleynet>
- Book: <https://github.com/arpanpathak/staleynet/tree/main/book>
- Paper: <https://github.com/arpanpathak/staleynet/tree/main/paper>
- API documentation: <https://docs.rs/staleynet-audio>

## License

Apache-2.0. See [LICENSE](https://github.com/arpanpathak/staleynet/blob/main/LICENSE).
