//! StaleyNET Dashboard: live observability for training runs.
//!
//! ```text
//! cargo run --release -- [--address 0.0.0.0:8790] --runs thor=~/Projects/stayleNET/runs [--runs orin=/path/to/runs]
//! ```
//!
//! The dashboard reads the `metrics.jsonl` and `config.json` that every
//! StaleyNET trainer writes, so it needs no change to training and no GPU
//! libraries. Each `--runs` folder is given a label, shown with its runs. It
//! serves one page that refreshes itself: losses, the generalization gap,
//! progress and the machine's load.

mod events;
mod history;
mod runs;
mod server;
mod system;

use runs::Source;
use server::Dashboard;
use std::path::PathBuf;
use std::sync::Arc;

/// Where the dashboard listens when no address is given.
const DEFAULT_ADDRESS: &str = "127.0.0.1:8790";

/// Requests answered at once; each one reads a few files.
const WORKERS: usize = 4;

/// The command line, parsed.
struct Settings {
    /// `host:port` to listen on.
    address: String,

    /// The runs folders.
    sources: Vec<Source>,
}

/// Parses `--address` and every `--runs label=folder`.
fn parse(arguments: impl Iterator<Item = String>) -> Result<Settings, String> {
    let mut settings = Settings { address: DEFAULT_ADDRESS.into(), sources: Vec::new() };
    let mut arguments = arguments.peekable();

    while let Some(flag) = arguments.next() {
        let value = arguments.next().ok_or(format!("{flag} needs a value"))?;

        match flag.as_str() {
            "--address" => settings.address = value,
            "--runs" => settings.sources.push(source(&value)?),
            other => return Err(format!("unknown option {other}")),
        }
    }

    if settings.sources.is_empty() {
        return Err("give at least one --runs label=folder".into());
    }

    Ok(settings)
}

/// One `label=folder` argument; a bare folder is labelled by its parent's name.
fn source(value: &str) -> Result<Source, String> {
    let Some((label, folder)) = value.split_once('=') else {
        let folder = PathBuf::from(value);
        let label = folder
            .parent()
            .and_then(|parent| parent.file_name())
            .map_or("runs".into(), |name| name.to_string_lossy().into_owned());

        return Ok(Source { label, folder });
    };

    if label.is_empty() || label.contains('/') {
        return Err(format!("a label must be a plain name, not {label:?}"));
    }

    Ok(Source { label: label.into(), folder: PathBuf::from(folder) })
}

fn main() {
    let settings = match parse(std::env::args().skip(1)) {
        Ok(settings) => settings,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

    let server = match tiny_http::Server::http(&settings.address) {
        Ok(server) => Arc::new(server),
        Err(error) => {
            eprintln!("cannot listen on {}: {error}", settings.address);
            std::process::exit(1);
        }
    };

    for source in &settings.sources {
        eprintln!("runs from {} ({})", source.folder.display(), source.label);
    }
    eprintln!("StaleyNET Dashboard on http://{}", settings.address);

    let dashboard = Arc::new(Dashboard { sources: settings.sources, recorder: system::Recorder::start() });
    let workers: Vec<_> = (0..WORKERS)
        .map(|_| {
            let server = Arc::clone(&server);
            let dashboard = Arc::clone(&dashboard);

            std::thread::spawn(move || {
                for request in server.incoming_requests() {
                    dashboard.handle(request);
                }
            })
        })
        .collect();

    for worker in workers {
        let _ = worker.join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> impl Iterator<Item = String> {
        values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn labelled_and_bare_folders_are_accepted() {
        let settings = parse(arguments(&["--runs", "thor=/data/runs", "--runs", "/home/me/stayleNET/runs"])).unwrap();

        assert_eq!(settings.sources[0], Source { label: "thor".into(), folder: "/data/runs".into() });
        assert_eq!(settings.sources[1].label, "stayleNET");
        assert_eq!(settings.address, DEFAULT_ADDRESS);
    }

    #[test]
    fn missing_runs_and_bad_labels_are_refused() {
        assert!(parse(arguments(&["--address", "0.0.0.0:1"])).is_err());
        assert!(parse(arguments(&["--runs", "a/b=/runs"])).is_err());
        assert!(parse(arguments(&["--runs"])).is_err());
    }
}
