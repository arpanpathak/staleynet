//! The machine the runs train on: GPU load, memory, CPU load and temperatures,
//! now and over the last hour.
//!
//! Everything is read from the kernel's own files, except the GPU load on
//! machines whose driver reports it only through `nvidia-smi` (the AGX Thor).
//! On Jetson modules with an integrated GPU (the Orin NX), the load is the
//! sysfs file the Jetson tools read. A reading that is not available on a
//! machine is left out rather than guessed.

use serde::Serialize;
use std::collections::VecDeque;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Seconds between two samples of the history.
pub const SAMPLE_SECONDS: u64 = 10;

/// Samples kept: one hour.
const HISTORY_SAMPLES: usize = 360;

/// Jetson sysfs files holding the integrated GPU's load in tenths of a percent.
const JETSON_GPU_LOAD_FILES: [&str; 2] = ["/sys/devices/platform/gpu.0/load", "/sys/devices/gpu.0/load"];

/// Tenths of a percent per percent, the unit of the Jetson load file.
const PER_MILLE_PER_PERCENT: f64 = 10.0;

/// Thermal zones report millidegrees Celsius.
const MILLIDEGREES_PER_DEGREE: f64 = 1000.0;

/// Kibibytes per gibibyte, for `/proc/meminfo`.
const KIB_PER_GIB: f64 = 1024.0 * 1024.0;

/// One reading of the machine.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Sample {
    /// Seconds since the Unix epoch.
    pub time: u64,

    /// GPU busy time, in percent, where the machine reports it.
    pub gpu_percent: Option<f64>,

    /// Memory in use, in GiB (the GPU shares it on Jetson modules).
    pub memory_used_gib: Option<f64>,

    /// Total memory, in GiB.
    pub memory_total_gib: Option<f64>,

    /// Runnable processes averaged over the last minute.
    pub load_one_minute: Option<f64>,

    /// Logical CPUs, to read the load against.
    pub cpus: usize,

    /// Every thermal zone the kernel reports.
    pub temperatures: Vec<Temperature>,
}

/// One thermal zone's reading.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Temperature {
    /// The zone's name, such as `gpu-thermal`.
    pub zone: String,

    /// Degrees Celsius.
    pub celsius: f64,
}

/// The latest samples, oldest first, shared between the sampler and the server.
#[derive(Clone, Default)]
pub struct Recorder {
    /// The samples.
    samples: Arc<Mutex<VecDeque<Sample>>>,
}

impl Recorder {
    /// Starts sampling every [`SAMPLE_SECONDS`] on a background thread.
    pub fn start() -> Self {
        let recorder = Recorder::default();
        let shared = recorder.clone();

        std::thread::spawn(move || {
            loop {
                shared.record(sample());
                std::thread::sleep(Duration::from_secs(SAMPLE_SECONDS));
            }
        });

        recorder
    }

    /// Adds a sample, dropping the oldest beyond one hour.
    fn record(&self, sample: Sample) {
        let mut samples = self
            .samples
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        samples.push_back(sample);

        while samples.len() > HISTORY_SAMPLES {
            samples.pop_front();
        }
    }

    /// Every sample kept, oldest first.
    pub fn history(&self) -> Vec<Sample> {
        let samples = self
            .samples
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        samples.iter().cloned().collect()
    }
}

/// The machine now.
pub fn sample() -> Sample {
    let memory = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let total = meminfo_kib(&memory, "MemTotal");
    let available = meminfo_kib(&memory, "MemAvailable");

    Sample {
        time: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs()),
        gpu_percent: gpu_percent(),
        memory_used_gib: total.and_then(|total| available.map(|available| (total - available) / KIB_PER_GIB)),
        memory_total_gib: total.map(|total| total / KIB_PER_GIB),
        load_one_minute: std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|text| text.split_whitespace().next()?.parse().ok()),
        cpus: std::thread::available_parallelism().map_or(1, |cpus| cpus.get()),
        temperatures: temperatures(),
    }
}

/// A field of `/proc/meminfo`, in KiB.
fn meminfo_kib(meminfo: &str, field: &str) -> Option<f64> {
    let line = meminfo
        .lines()
        .find(|line| line.starts_with(&format!("{field}:")))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// The GPU's load in percent, from the Jetson sysfs file or else `nvidia-smi`.
fn gpu_percent() -> Option<f64> {
    for file in JETSON_GPU_LOAD_FILES {
        if let Some(per_mille) = std::fs::read_to_string(file)
            .ok()
            .and_then(|text| text.trim().parse::<f64>().ok())
        {
            return Some(per_mille / PER_MILLE_PER_PERCENT);
        }
    }

    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

/// Every thermal zone with a readable name and temperature.
fn temperatures() -> Vec<Temperature> {
    let Ok(zones) = std::fs::read_dir("/sys/class/thermal") else { return Vec::new() };

    let mut readings: Vec<Temperature> = zones
        .filter_map(|zone| zone.ok())
        .filter(|zone| {
            zone.file_name()
                .to_string_lossy()
                .starts_with("thermal_zone")
        })
        .filter_map(|zone| read_zone(&zone.path()))
        .collect();

    readings.sort_by(|left, right| left.zone.cmp(&right.zone));
    readings
}

/// One thermal zone, when both its name and a plausible temperature can be read.
fn read_zone(zone: &Path) -> Option<Temperature> {
    let name = std::fs::read_to_string(zone.join("type")).ok()?;
    let millidegrees: f64 = std::fs::read_to_string(zone.join("temp"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    let celsius = millidegrees / MILLIDEGREES_PER_DEGREE;

    // Disabled zones report placeholder values such as -40 or 0.
    (celsius > 0.0).then(|| Temperature { zone: name.trim().to_string(), celsius })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meminfo_fields_are_read_in_kib() {
        let meminfo = "MemTotal:       16384000 kB\nMemFree:          100 kB\nMemAvailable:    8192000 kB\n";

        assert_eq!(meminfo_kib(meminfo, "MemTotal"), Some(16_384_000.0));
        assert_eq!(meminfo_kib(meminfo, "MemAvailable"), Some(8_192_000.0));
        assert_eq!(meminfo_kib(meminfo, "Missing"), None);
    }

    #[test]
    fn the_history_keeps_one_hour() {
        let recorder = Recorder::default();
        for time in 0..(HISTORY_SAMPLES as u64 + 5) {
            recorder.record(Sample {
                time,
                gpu_percent: None,
                memory_used_gib: None,
                memory_total_gib: None,
                load_one_minute: None,
                cpus: 1,
                temperatures: Vec::new(),
            });
        }

        let history = recorder.history();
        assert_eq!(history.len(), HISTORY_SAMPLES);
        assert_eq!(history[0].time, 5);
    }
}
