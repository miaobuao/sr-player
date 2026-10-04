//! Error type shared by the whole engine.
//!
//! The distinction that matters is [`Error::retryable`]: the pipeline runner
//! uses it to decide between "walk the degrade ladder and try again" and
//! "mark the stage permanently failed".

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("required tool not found: {0}. Install FFmpeg or set SR_FFMPEG / SR_FFPROBE")]
    ToolMissing(String),

    #[error("ffprobe failed for {path}: {detail}")]
    Probe { path: PathBuf, detail: String },

    #[error("ffmpeg exited with code {code}{}{}", if *.oom { " (out of memory)" } else { "" }, if stderr.is_empty() { String::new() } else { format!(": {stderr}") })]
    Ffmpeg {
        code: i32,
        stderr: String,
        oom: bool,
    },

    #[error("stage {stage} failed: {detail}")]
    Stage { stage: String, detail: String },

    #[error("no usable video encoder in this FFmpeg build (tried: {tried})")]
    NoEncoder { tried: String },

    #[error("cancelled by user")]
    Cancelled,

    #[error("unsupported input: {0}")]
    Unsupported(String),

    #[error("job state error: {0}")]
    State(String),

    #[error("plugin {plugin} error: {detail}")]
    Plugin { plugin: String, detail: String },

    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("io error: {0}")]
    BareIo(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("{0}")]
    Other(String),
}

impl Error {
    /// May the caller retry the same stage, possibly with a weaker profile?
    pub fn retryable(&self) -> bool {
        match self {
            // Any FFmpeg failure is worth one retry: the runner answers an OOM by
            // walking the degrade ladder and an unusable encoder by falling back
            // to the next one, and both are cheap next to losing a job.
            Error::Ffmpeg { .. } => true,
            Error::Plugin { .. } => true, // fall back to the FFmpeg executor
            Error::Stage { .. } => true,
            Error::Cancelled => false,
            Error::Unsupported(_) => false,
            Error::ToolMissing(_) => false,
            Error::Probe { .. } => false,
            Error::NoEncoder { .. } => false,
            Error::State(_) => false,
            Error::Io { .. } | Error::BareIo(_) => false,
            Error::Json(_) => false,
            Error::Db(_) => false,
            Error::Other(_) => false,
        }
    }

    /// True when the failure looks like memory pressure (the degrade signal).
    pub fn is_oom(&self) -> bool {
        match self {
            Error::Ffmpeg { oom, .. } => *oom,
            other => looks_like_oom(&other.to_string()),
        }
    }

    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}

/// Heuristic used on FFmpeg/plugin stderr text.
pub fn looks_like_oom(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    const NEEDLES: [&str; 8] = [
        "out of memory",
        "cuda out of memory",
        "cuda_error_out_of_memory",
        "cannot allocate memory",
        "failed to allocate",
        "insufficient memory",
        "vk_error_out_of_device_memory",
        "not enough memory",
    ];
    NEEDLES.iter().any(|n| t.contains(n))
}
