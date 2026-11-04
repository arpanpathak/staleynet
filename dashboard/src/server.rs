//! HTTP routing.
//!
//! | Method | Path               | Purpose                                        |
//! |--------|--------------------|------------------------------------------------|
//! | GET    | `/`                | the single-page dashboard                      |
//! | GET    | `/api/runs`        | every run at a glance                          |
//! | GET    | `/api/runs/<id>`   | one run's summary, configuration and series    |
//! | GET    | `/api/system`      | the machine now and over the last hour         |
//!
//! Every response is marked `no-store`: the page and the numbers change while
//! runs train, and a browser must never show a stored copy.

use crate::runs::{self, Source};
use crate::system::{self, Recorder, Sample};
use serde::Serialize;
use tiny_http::{Header, Method, Request, Response, StatusCode};

/// The page, compiled into the binary so the dashboard is a single file.
const INDEX_HTML: &str = include_str!("../static/index.html");

/// HTTP status codes the dashboard answers with.
mod status {
    pub const OK: u16 = 200;
    pub const NOT_FOUND: u16 = 404;
    pub const INTERNAL_ERROR: u16 = 500;
}

/// What the dashboard serves from.
pub struct Dashboard {
    /// The runs folders, each with its label.
    pub sources: Vec<Source>,

    /// The machine's recent history.
    pub recorder: Recorder,
}

/// The machine now, with its last hour.
#[derive(Serialize)]
struct SystemReply {
    /// The latest reading.
    now: Sample,

    /// Readings every [`system::SAMPLE_SECONDS`] over the last hour, oldest first.
    history: Vec<Sample>,
}

/// A response body with its content type and status.
struct Reply {
    /// HTTP status code.
    status: u16,

    /// `Content-Type` header value.
    content_type: &'static str,

    /// The body.
    body: Vec<u8>,
}

impl Reply {
    /// `value` as JSON.
    fn json(value: &impl Serialize) -> Reply {
        match serde_json::to_vec(value) {
            Ok(body) => Reply { status: status::OK, content_type: "application/json", body },
            Err(error) => Reply::error(status::INTERNAL_ERROR, &error.to_string()),
        }
    }

    /// A plain-text error.
    fn error(status: u16, message: &str) -> Reply {
        Reply { status, content_type: "text/plain; charset=utf-8", body: message.as_bytes().to_vec() }
    }
}

impl Dashboard {
    /// Answers one request.
    pub fn handle(&self, request: Request) {
        let url = request.url().to_string();
        let path = url.split('?').next().unwrap_or("");

        let reply = match (request.method(), path) {
            (Method::Get, "/") => {
                Reply { status: status::OK, content_type: "text/html; charset=utf-8", body: INDEX_HTML.as_bytes().to_vec() }
            }
            (Method::Get, "/api/runs") => Reply::json(&runs::discover(&self.sources)),
            (Method::Get, "/api/system") => Reply::json(&SystemReply { now: system::sample(), history: self.recorder.history() }),
            (Method::Get, run) if run.starts_with("/api/runs/") => self.run(&run["/api/runs/".len()..]),
            _ => Reply::error(status::NOT_FOUND, "not found"),
        };

        let content_type = Header::from_bytes("Content-Type", reply.content_type).expect("static header is valid");
        let no_store = Header::from_bytes("Cache-Control", "no-store").expect("static header is valid");
        let response = Response::from_data(reply.body)
            .with_status_code(StatusCode(reply.status))
            .with_header(content_type)
            .with_header(no_store);

        if let Err(error) = request.respond(response) {
            eprintln!("could not answer {url}: {error}");
        }
    }

    /// One run's detail, by id.
    fn run(&self, id: &str) -> Reply {
        let Some(detail) = runs::find(&self.sources, id) else { return Reply::error(status::NOT_FOUND, "no such run") };

        Reply::json(&detail)
    }
}
