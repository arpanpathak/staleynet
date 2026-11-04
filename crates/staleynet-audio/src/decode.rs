//! Decoding of compressed and uncompressed audio files into mono samples,
//! using the pure-Rust Symphonia demuxers and codecs
//! (<https://github.com/pdeljanov/Symphonia>).

use anyhow::{Context, Result, anyhow};
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as DecodeError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// File extensions the decoder is expected to handle.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["wav", "mp3", "flac", "m4a", "aac", "ogg", "mp4"];

/// A single-channel waveform.
pub struct MonoAudio {
    /// Samples in `[-1, 1]`.
    pub samples: Vec<f32>,

    /// Sampling rate of `samples` in Hz.
    pub sample_rate: u32,
}

/// Returns true when `path` has one of the [`SUPPORTED_EXTENSIONS`].
pub fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| SUPPORTED_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()))
}

/// Decodes `path` and averages all channels into one (equal-weight
/// downmix; see <https://en.wikipedia.org/wiki/Downmixing>).
///
/// Corrupt packets are skipped rather than aborting the whole file, which is
/// the common failure mode of scraped MP3s.
pub fn decode_mono(path: &Path) -> Result<MonoAudio> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, stream, &FormatOptions::default(), &MetadataOptions::default())
        .with_context(|| format!("probe {}", path.display()))?;
    let mut container = probed.format;

    let track = container.default_track().ok_or_else(|| anyhow!("no audio track in {}", path.display()))?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.ok_or_else(|| anyhow!("unknown sample rate in {}", path.display()))?;
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut samples = Vec::new();
    loop {
        let packet = match container.next_packet() {
            Ok(packet) => packet,
            Err(DecodeError::IoError(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(DecodeError::ResetRequired) => break,
            Err(error) => return Err(error.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(DecodeError::DecodeError(_)) => continue,
            Err(error) => return Err(error.into()),
        };

        let spec = *decoded.spec();
        let channel_count = spec.channels.count().max(1);
        let mut interleaved = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        interleaved.copy_interleaved_ref(decoded);

        let channel_weight = 1.0 / channel_count as f32;
        samples.extend(interleaved.samples().chunks_exact(channel_count).map(|frame| frame.iter().sum::<f32>() * channel_weight));
    }

    Ok(MonoAudio { samples, sample_rate })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{WavSpec, WavWriter};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn temp_path(tag: &str, extension: &str) -> std::path::PathBuf {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);

        std::env::temp_dir().join(format!("staleynet-decode-{tag}-{}-{id}.{extension}", std::process::id()))
    }

    fn write_wav(path: &std::path::Path, channels: u16, sample_rate: u32, samples: &[i16]) {
        let spec = WavSpec { channels, sample_rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut writer = WavWriter::create(path, spec).unwrap();

        for sample in samples {
            writer.write_sample(*sample).unwrap();
        }

        writer.finalize().unwrap();
    }

    #[test]
    fn recognises_supported_extensions_case_insensitively() {
        for extension in SUPPORTED_EXTENSIONS {
            let lower = format!("song.{extension}");
            let upper = format!("song.{}", extension.to_uppercase());

            assert!(is_audio_file(std::path::Path::new(&lower)));
            assert!(is_audio_file(std::path::Path::new(&upper)));
        }

        assert!(!is_audio_file(std::path::Path::new("song.txt")));
        assert!(!is_audio_file(std::path::Path::new("song")));
    }

    #[test]
    fn decodes_a_mono_wav() {
        let path = temp_path("mono", "wav");

        write_wav(&path, 1, 8_000, &[0, 8_192, -8_192]);

        let audio = decode_mono(&path).unwrap();

        assert_eq!(audio.sample_rate, 8_000);
        assert_eq!(audio.samples.len(), 3);
        assert!(audio.samples[0].abs() < 1e-3);
        assert!((audio.samples[1] - 0.25).abs() < 1e-3);
        assert!((audio.samples[2] + 0.25).abs() < 1e-3);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn downmixes_stereo_by_averaging_channels() {
        let path = temp_path("stereo", "wav");

        write_wav(&path, 2, 8_000, &[8_192, -8_192, 4_096, 4_096]);

        let audio = decode_mono(&path).unwrap();

        assert_eq!(audio.sample_rate, 8_000);
        assert_eq!(audio.samples.len(), 2);
        assert!(audio.samples[0].abs() < 1e-3);
        assert!((audio.samples[1] - 0.125).abs() < 1e-3);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn accepts_an_empty_wav() {
        let path = temp_path("empty", "wav");

        write_wav(&path, 1, 8_000, &[]);

        let audio = decode_mono(&path).unwrap();

        assert_eq!(audio.sample_rate, 8_000);
        assert!(audio.samples.is_empty());

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn chained_streams_stop_at_the_reset_boundary() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chained.ogg");

        let audio = decode_mono(&path).unwrap();

        assert_eq!(audio.sample_rate, 44_100);
        assert!(audio.samples.is_empty());
    }

    #[test]
    fn reports_missing_and_corrupt_files() {
        let missing = temp_path("missing", "wav");

        assert!(decode_mono(&missing).is_err());

        let corrupt = temp_path("corrupt", "wav");
        std::fs::write(&corrupt, b"this is not a wav file").unwrap();

        assert!(decode_mono(&corrupt).is_err());

        std::fs::remove_file(corrupt).unwrap();
    }
}
