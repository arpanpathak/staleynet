//! StaleyNET Studio: a browser interface for generating and listening to
//! samples from trained StaleyNET runs.
//!
//! ```text
//! cargo run --release -- [--address 127.0.0.1:8080] [--project ..] [--binary ../target/release/staleynet]
//!                         [--tls-cert certs/studio.crt --tls-key certs/studio.key]
//! ```
//!
//! Browsers allow microphone access only on `https://` pages or on
//! `localhost` (<https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts>).
//! To record from another machine, serve over HTTPS with a self-signed
//! certificate (`./make-certificate.sh`) and accept the browser's warning once.
//!
//! The studio does not link the GPU code; it invokes the `staleynet` binary,
//! so it builds with plain `cargo` in seconds.

mod generation;
mod library;
mod recordings;
mod runs;
mod server;

use server::Studio;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Where the studio listens unless `--address` is given. Loopback only; use
/// `--address 0.0.0.0:8080` to reach it from another machine on the network.
const DEFAULT_ADDRESS: &str = "127.0.0.1:8080";

/// Threads answering requests. One generation runs at a time; the others keep
/// the clip list and audio streaming responsive meanwhile.
const WORKER_THREADS: usize = 4;

/// Command-line settings.
struct Settings {
    /// Socket address to bind.
    address: String,

    /// StaleyNET checkout root.
    project: PathBuf,

    /// Path to the `staleynet` binary.
    binary: PathBuf,

    /// PEM certificate and private key; serve HTTPS when present.
    tls: Option<TlsFiles>,
}

/// Locations of the PEM files for HTTPS.
struct TlsFiles {
    /// Certificate chain.
    certificate: PathBuf,

    /// Private key (PKCS#8).
    private_key: PathBuf,
}

/// Parses the command line; defaults assume the studio is run from `studio/`
/// inside the checkout.
fn parse_settings() -> Result<Settings, String> {
    let mut settings = Settings { address: DEFAULT_ADDRESS.into(), project: PathBuf::from(".."), binary: PathBuf::new(), tls: None };

    let mut binary = None;
    let mut certificate = None;
    let mut private_key = None;

    let mut arguments = std::env::args().skip(1);

    while let Some(flag) = arguments.next() {
        let value = arguments.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--address" => settings.address = value,
            "--project" => settings.project = PathBuf::from(value),
            "--binary" => binary = Some(PathBuf::from(value)),
            "--tls-cert" => certificate = Some(PathBuf::from(value)),
            "--tls-key" => private_key = Some(PathBuf::from(value)),
            other => return Err(format!("unknown flag {other}")),
        }
    }

    settings.project = settings.project.canonicalize().map_err(|error| format!("project directory: {error}"))?;
    settings.binary = binary.unwrap_or_else(|| settings.project.join("target/release/staleynet"));
    if !settings.binary.is_file() {
        return Err(format!("{} not found; build it with `cargo oxide build -- --release`", settings.binary.display()));
    }

    settings.tls = match (certificate, private_key) {
        (Some(certificate), Some(private_key)) => Some(TlsFiles { certificate, private_key }),
        (None, None) => None,
        _ => return Err("--tls-cert and --tls-key go together".into()),
    };

    Ok(settings)
}

/// Binds the server, over HTTPS when certificate files were given; returns it
/// with the URL scheme it speaks.
fn listen(settings: &Settings) -> Result<(tiny_http::Server, &'static str), String> {
    let Some(tls) = &settings.tls else {
        return tiny_http::Server::http(&settings.address).map(|server| (server, "http")).map_err(|error| error.to_string());
    };

    let read = |path: &PathBuf| std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()));
    let ssl = tiny_http::SslConfig { certificate: read(&tls.certificate)?, private_key: read(&tls.private_key)? };
    tiny_http::Server::https(&settings.address, ssl).map(|server| (server, "https")).map_err(|error| error.to_string())
}

fn main() {
    let settings = match parse_settings() {
        Ok(settings) => settings,
        Err(message) => {
            eprintln!("staleynet-studio: {message}");
            std::process::exit(2);
        }
    };

    let (server, scheme) = match listen(&settings) {
        Ok(listening) => listening,
        Err(message) => {
            eprintln!("staleynet-studio: cannot listen on {}: {message}", settings.address);
            std::process::exit(1);
        }
    };

    let server = Arc::new(server);
    let studio = Arc::new(Studio { project: settings.project, binary: settings.binary, generation_lock: Mutex::new(()) });

    eprintln!("StaleyNET Studio on {scheme}://{}  (project {})", settings.address, studio.project.display());

    let workers: Vec<_> = (0..WORKER_THREADS)
        .map(|_| {
            let server = Arc::clone(&server);
            let studio = Arc::clone(&studio);
            std::thread::spawn(move || {
                while let Ok(request) = server.recv() {
                    studio.handle(request);
                }
            })
        })
        .collect();

    for worker in workers {
        let _ = worker.join();
    }
}
