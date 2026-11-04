//! WAV output (RIFF/WAVE, 16-bit linear PCM).
//!
//! Format reference: <https://en.wikipedia.org/wiki/WAV>

use anyhow::Result;
use std::path::Path;

/// Peak level output is normalised to when it would otherwise clip:
/// 10^(−1/20) ≈ 0.891, i.e. −1 dBFS of headroom against inter-sample peaks.
/// See <https://en.wikipedia.org/wiki/DBFS>
const OUTPUT_PEAK_LIMIT: f32 = 0.891;

/// Full-scale value of a signed 16-bit sample, 2¹⁵ − 1.
/// See <https://en.wikipedia.org/wiki/Audio_bit_depth>
const PCM16_FULL_SCALE: f32 = 32767.0;

/// Writes `samples` as 16-bit mono PCM, scaling down only if the peak exceeds
/// [`OUTPUT_PEAK_LIMIT`].
pub fn write_mono_pcm16(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec { channels: 1, sample_rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };

    let peak = samples.iter().fold(0.0f32, |peak, sample| peak.max(sample.abs()));
    let gain = if peak > OUTPUT_PEAK_LIMIT { OUTPUT_PEAK_LIMIT / peak } else { 1.0 };

    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample((sample * gain * PCM16_FULL_SCALE) as i16)?;
    }
    writer.finalize()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::WavReader;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn temp_path(tag: &str) -> std::path::PathBuf {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);

        std::env::temp_dir().join(format!("staleynet-wav-{tag}-{}-{id}.wav", std::process::id()))
    }

    #[test]
    fn writes_mono_pcm16_at_the_requested_rate() {
        let path = temp_path("mono");
        let input = [0.0, 0.25, -0.5];

        write_mono_pcm16(&path, &input, 24_000).unwrap();

        let mut reader = WavReader::open(&path).unwrap();
        let spec = reader.spec();

        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 24_000);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);

        let samples: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();

        assert_eq!(samples.len(), 3);
        assert!(samples[0].abs() <= 1);
        assert!((samples[1] as f32 / PCM16_FULL_SCALE - 0.25).abs() < 1e-3);
        assert!((samples[2] as f32 / PCM16_FULL_SCALE + 0.5).abs() < 1e-3);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn normalises_only_when_the_peak_exceeds_the_limit() {
        let clipped = temp_path("clipped");
        write_mono_pcm16(&clipped, &[2.0, -1.0], 24_000).unwrap();

        let samples: Vec<i16> = WavReader::open(&clipped).unwrap().samples::<i16>().map(Result::unwrap).collect();

        assert!(samples[0] > 0);
        assert!(samples[1] < 0);
        assert!((samples[0] as f32 / PCM16_FULL_SCALE) <= OUTPUT_PEAK_LIMIT + 1e-3);

        let ratio = samples[0] as f32 / samples[1].abs() as f32;

        assert!((ratio - 2.0).abs() < 0.01, "ratio {ratio}");

        std::fs::remove_file(clipped).unwrap();

        let quiet = temp_path("quiet");
        write_mono_pcm16(&quiet, &[0.5, -0.5], 24_000).unwrap();

        let samples: Vec<i16> = WavReader::open(&quiet).unwrap().samples::<i16>().map(Result::unwrap).collect();

        assert!((samples[0] as f32 / PCM16_FULL_SCALE - 0.5).abs() < 1e-3);
        assert!((samples[1] as f32 / PCM16_FULL_SCALE + 0.5).abs() < 1e-3);

        std::fs::remove_file(quiet).unwrap();
    }

    #[test]
    fn writes_an_empty_file_without_panicking() {
        let path = temp_path("empty");

        write_mono_pcm16(&path, &[], 16_000).unwrap();

        let mut reader = WavReader::open(&path).unwrap();

        assert_eq!(reader.spec().sample_rate, 16_000);
        assert_eq!(reader.samples::<i16>().count(), 0);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reports_an_error_when_the_parent_directory_is_missing() {
        let path =
            std::env::temp_dir().join(format!("staleynet-wav-missing-{}", std::process::id())).join("does-not-exist").join("out.wav");

        assert!(write_mono_pcm16(&path, &[0.0], 24_000).is_err());
    }
}
