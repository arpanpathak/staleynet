//! HTTP routing.
//!
//! | Method | Path               | Purpose                              |
//! |--------|--------------------|--------------------------------------|
//! | GET    | `/`                | the single-page interface            |
//! | GET    | `/api/runs`        | runs with checkpoints                |
//! | GET    | `/api/clips`       | every WAV under `samples/`           |
//! | POST   | `/api/generate`    | generate a clip (JSON body)          |
//! | GET    | `/audio/<path>`    | stream one WAV                       |
//! | GET    | `/api/recordings`  | your saved takes                     |
//! | POST   | `/api/recordings`  | save a take (JSON with base64 WAV)   |
//! | DELETE | `/api/recordings/<id>` | delete a take                    |
//! | GET    | `/recordings/<id>.wav` | stream one take                  |

use crate::generation::{GenerationError, GenerationRequest, generate};
use crate::{library, recordings, runs};
use serde::Serialize;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Mutex;
use tiny_http::{Header, Method, Request, Response, StatusCode};

/// The page, compiled into the binary so the studio is a single file.
const INDEX_HTML: &str = include_str!("../static/index.html");

/// Largest generation form accepted (one is a few hundred bytes).
const MAX_FORM_BYTES: u64 = 16 * 1024;

/// Largest take upload accepted: [`recordings::MAX_SECONDS`] of 48 kHz 16-bit
/// mono is 28.8 MB, and base64 adds a third.
const MAX_UPLOAD_BYTES: u64 = 64 * 1024 * 1024;

/// HTTP status codes used by the studio.
mod status {
    pub const OK: u16 = 200;
    pub const BAD_REQUEST: u16 = 400;
    pub const NOT_FOUND: u16 = 404;
    pub const PAYLOAD_TOO_LARGE: u16 = 413;
    pub const INTERNAL_ERROR: u16 = 500;
}

/// Shared state of the running studio.
pub struct Studio {
    /// Root of the StaleyNET checkout (contains `runs/`, `samples/`, `data/`).
    pub project: PathBuf,

    /// The `staleynet` binary used for generation.
    pub binary: PathBuf,

    /// Held while a generation runs: the GPU serves one request at a time.
    pub generation_lock: Mutex<()>,
}

/// A response body plus its content type and status.
struct Reply {
    /// HTTP status code.
    status: u16,

    /// `Content-Type` header value.
    content_type: &'static str,

    /// Body bytes.
    body: Vec<u8>,
}

impl Reply {
    /// `value` serialised as JSON with status `status`.
    fn json(status: u16, value: &impl Serialize) -> Self {
        Self { status, content_type: "application/json", body: serde_json::to_vec(value).unwrap_or_default() }
    }

    /// `{"error": message}` with status `status`.
    fn error(status: u16, message: impl Into<String>) -> Self {
        Self::json(status, &serde_json::json!({ "error": message.into() }))
    }
}

impl Studio {
    /// Answers one request.
    pub fn handle(&self, mut request: Request) {
        let url = request.url().to_string();
        let path = url.split('?').next().unwrap_or_default();

        let reply = match (request.method(), path) {
            (Method::Get, "/") => {
                Reply { status: status::OK, content_type: "text/html; charset=utf-8", body: INDEX_HTML.as_bytes().to_vec() }
            }
            (Method::Get, "/api/runs") => Reply::json(status::OK, &runs::discover(&self.project)),
            (Method::Get, "/api/clips") => Reply::json(status::OK, &library::list(&self.project.join("samples"))),
            (Method::Post, "/api/generate") => self.generate(&mut request),
            (Method::Get, audio) if audio.starts_with("/audio/") => self.audio(&self.project.join("samples"), &audio["/audio/".len()..]),
            (Method::Get, "/api/recordings") => Reply::json(status::OK, &recordings::list(&self.project)),
            (Method::Post, "/api/recordings") => self.save_recording(&mut request),
            (Method::Delete, take) if take.starts_with("/api/recordings/") => self.delete_recording(&take["/api/recordings/".len()..]),
            (Method::Get, take) if take.starts_with("/recordings/") => {
                self.audio(&recordings::directory(&self.project), &take["/recordings/".len()..])
            }
            _ => Reply::error(status::NOT_FOUND, "not found"),
        };

        let header = Header::from_bytes("Content-Type", reply.content_type).expect("static header is valid");
        let response = Response::from_data(reply.body).with_status_code(StatusCode(reply.status)).with_header(header);
        if let Err(error) = request.respond(response) {
            eprintln!("could not send response for {url}: {error}");
        }
    }

    /// `POST /api/generate`.
    fn generate(&self, request: &mut Request) -> Reply {
        let form: GenerationRequest = match read_json(request, MAX_FORM_BYTES) {
            Ok(form) => form,
            Err(reply) => return reply,
        };

        let _exclusive = self.generation_lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        eprintln!("generating {:.0}s with {}: {:?}", form.seconds, form.run, form.prompt);

        match generate(&self.project, &self.binary, &form) {
            Ok(path) => Reply::json(status::OK, &serde_json::json!({ "path": path })),
            Err(GenerationError::Invalid(message)) => Reply::error(status::BAD_REQUEST, message),
            Err(GenerationError::Failed(message)) => Reply::error(status::INTERNAL_ERROR, message),
        }
    }

    /// `POST /api/recordings`.
    fn save_recording(&self, request: &mut Request) -> Reply {
        let upload: recordings::Upload = match read_json(request, MAX_UPLOAD_BYTES) {
            Ok(upload) => upload,
            Err(reply) => return reply,
        };

        match recordings::save(&self.project, upload) {
            Ok(recording) => {
                eprintln!("saved take {} ({:.1} s)", recording.id, recording.seconds);
                Reply::json(status::OK, &recording)
            }
            Err(message) => Reply::error(status::BAD_REQUEST, message),
        }
    }

    /// `DELETE /api/recordings/<id>`.
    fn delete_recording(&self, id: &str) -> Reply {
        if !recordings::delete(&self.project, &percent_decode(id)) {
            return Reply::error(status::NOT_FOUND, "no such take");
        }
        Reply::json(status::OK, &serde_json::json!({ "deleted": id }))
    }

    /// Streams one WAV from `root`, refusing paths that leave it.
    fn audio(&self, root: &std::path::Path, encoded_path: &str) -> Reply {
        let relative = percent_decode(encoded_path);
        let Some(file) = library::resolve(root, &relative) else {
            return Reply::error(status::NOT_FOUND, "no such clip");
        };

        match std::fs::read(file) {
            Ok(bytes) => Reply { status: status::OK, content_type: "audio/wav", body: bytes },
            Err(error) => Reply::error(status::INTERNAL_ERROR, error.to_string()),
        }
    }
}

/// Reads at most `limit` bytes of body and parses them as JSON, or returns
/// the error reply to send.
fn read_json<T: serde::de::DeserializeOwned>(request: &mut Request, limit: u64) -> Result<T, Reply> {
    let too_large = request.body_length().is_some_and(|length| length as u64 > limit);
    if too_large {
        return Err(Reply::error(status::PAYLOAD_TOO_LARGE, format!("request body over {limit} bytes")));
    }

    let mut body = String::new();
    if request.as_reader().take(limit).read_to_string(&mut body).is_err() {
        return Err(Reply::error(status::BAD_REQUEST, "unreadable request body"));
    }
    serde_json::from_str(&body).map_err(|error| Reply::error(status::BAD_REQUEST, format!("malformed request: {error}")))
}

/// Decodes `%XX` escapes in a URL path segment.
/// See RFC 3986 §2.1: <https://www.rfc-editor.org/rfc/rfc3986#section-2.1>
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        let escape = bytes.get(index + 1..index + 3).filter(|_| bytes[index] == b'%');
        let value = escape.and_then(|hex| std::str::from_utf8(hex).ok()).and_then(|hex| u8::from_str_radix(hex, 16).ok());

        match value {
            Some(byte) => {
                decoded.push(byte);
                index += 3;
            }
            None => {
                decoded.push(bytes[index]);
                index += 1;
            }
        }
    }

    String::from_utf8_lossy(&decoded).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_decoding_handles_escapes_and_plain_text() {
        assert_eq!(percent_decode("studio/a%20b.wav"), "studio/a b.wav");
        assert_eq!(percent_decode("100%"), "100%");
    }
}
