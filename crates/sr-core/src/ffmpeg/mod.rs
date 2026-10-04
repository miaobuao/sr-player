//! FFmpeg discovery, capability probing and vendor-neutral encoder selection.
//!
//! FFmpeg is the only hard external dependency of the engine. Everything that
//! follows from it — which hardware encoder exists, whether 10-bit output is
//! possible, which filters are present — is *probed*, never assumed. The engine
//! does not know what "NVENC" is; it asks for the best available encoder and
//! records what it got.

pub mod process;
pub mod progress;

use crate::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub use process::{
    capture, format_command, read_exact_or_eof, read_lines, run_tool, write_all, RunOutcome,
    RunSpec, StreamingChild,
};
pub use progress::{ProgressParser, ProgressTick};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoCodec {
    Av1,
    Hevc,
    H264,
}

impl VideoCodec {
    pub fn as_str(self) -> &'static str {
        match self {
            VideoCodec::Av1 => "av1",
            VideoCodec::Hevc => "hevc",
            VideoCodec::H264 => "h264",
        }
    }
}

/// What the engine would *like* to encode with. The answer may be different.
#[derive(Clone, Debug)]
pub struct EncoderPreference {
    pub prefer_hardware: bool,
    pub allow_software: bool,
    pub prefer_10bit: bool,
    pub codec_order: Vec<VideoCodec>,
    /// `SR_VIDEO_ENCODER` / profile override.
    pub forced: Option<String>,
    /// CQ/CRF-style quality target.
    pub quality: i32,
}

impl Default for EncoderPreference {
    fn default() -> Self {
        EncoderPreference {
            prefer_hardware: true,
            allow_software: true,
            prefer_10bit: true,
            codec_order: vec![VideoCodec::Av1, VideoCodec::Hevc, VideoCodec::H264],
            forced: std::env::var("SR_VIDEO_ENCODER").ok().filter(|s| !s.is_empty()),
            quality: 24,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectedVideoEncoder {
    pub name: String,
    pub codec: VideoCodec,
    pub vendor: Option<String>,
    pub hardware: bool,
    pub pix_fmt: String,
    pub quality_args: Vec<String>,
    /// Args that must precede `-i` (device selection).
    pub input_args: Vec<String>,
    /// True when frames must be uploaded to the device in the filter graph.
    pub requires_hwupload: bool,
    pub note: String,
}

impl SelectedVideoEncoder {
    /// One line for the UI: `av1_nvenc (nvidia, hardware, p010le)`.
    pub fn describe(&self) -> String {
        let vendor = self.vendor.as_deref().unwrap_or("generic");
        let kind = if self.hardware { "hardware" } else { "software" };
        format!("{} ({vendor}, {kind}, {})", self.name, self.pix_fmt)
    }
}

struct Candidate {
    name: &'static str,
    codec: VideoCodec,
    vendor: Option<&'static str>,
    hardware: bool,
    /// Preferred output pixel formats, best first.
    pix_fmts: &'static [&'static str],
    quality: fn(i32) -> Vec<String>,
    input_args: &'static [&'static str],
    requires_hwupload: bool,
}

fn nvenc_quality(q: i32) -> Vec<String> {
    vec![
        "-rc".into(),
        "vbr".into(),
        "-cq".into(),
        q.to_string(),
        "-b:v".into(),
        "0".into(),
        "-preset".into(),
        "p5".into(),
        "-tune".into(),
        "hq".into(),
    ]
}

fn amf_quality(q: i32) -> Vec<String> {
    vec![
        "-rc".into(),
        "cqp".into(),
        "-qp_i".into(),
        q.to_string(),
        "-qp_p".into(),
        q.to_string(),
        "-quality".into(),
        "quality".into(),
    ]
}

fn qsv_quality(q: i32) -> Vec<String> {
    vec![
        "-global_quality".into(),
        q.to_string(),
        "-preset".into(),
        "medium".into(),
    ]
}

fn vaapi_quality(q: i32) -> Vec<String> {
    vec!["-qp".into(), q.to_string()]
}

fn svtav1_quality(q: i32) -> Vec<String> {
    vec![
        "-preset".into(),
        "6".into(),
        "-crf".into(),
        q.to_string(),
    ]
}

fn x265_quality(q: i32) -> Vec<String> {
    vec![
        "-preset".into(),
        "medium".into(),
        "-crf".into(),
        q.to_string(),
        "-x265-params".into(),
        "log-level=error".into(),
    ]
}

fn x264_quality(q: i32) -> Vec<String> {
    vec![
        "-preset".into(),
        "medium".into(),
        "-crf".into(),
        q.to_string(),
    ]
}

const CANDIDATES: &[Candidate] = &[
    Candidate {
        name: "av1_nvenc",
        codec: VideoCodec::Av1,
        vendor: Some("nvidia"),
        hardware: true,
        pix_fmts: &["p010le", "nv12"],
        quality: nvenc_quality,
        input_args: &[],
        requires_hwupload: false,
    },
    Candidate {
        name: "hevc_nvenc",
        codec: VideoCodec::Hevc,
        vendor: Some("nvidia"),
        hardware: true,
        pix_fmts: &["p010le", "nv12"],
        quality: nvenc_quality,
        input_args: &[],
        requires_hwupload: false,
    },
    Candidate {
        name: "h264_nvenc",
        codec: VideoCodec::H264,
        vendor: Some("nvidia"),
        hardware: true,
        pix_fmts: &["nv12"],
        quality: nvenc_quality,
        input_args: &[],
        requires_hwupload: false,
    },
    Candidate {
        name: "av1_amf",
        codec: VideoCodec::Av1,
        vendor: Some("amd"),
        hardware: true,
        pix_fmts: &["p010le", "nv12"],
        quality: amf_quality,
        input_args: &[],
        requires_hwupload: false,
    },
    Candidate {
        name: "hevc_amf",
        codec: VideoCodec::Hevc,
        vendor: Some("amd"),
        hardware: true,
        pix_fmts: &["p010le", "nv12"],
        quality: amf_quality,
        input_args: &[],
        requires_hwupload: false,
    },
    Candidate {
        name: "h264_amf",
        codec: VideoCodec::H264,
        vendor: Some("amd"),
        hardware: true,
        pix_fmts: &["nv12"],
        quality: amf_quality,
        input_args: &[],
        requires_hwupload: false,
    },
    Candidate {
        name: "av1_qsv",
        codec: VideoCodec::Av1,
        vendor: Some("intel"),
        hardware: true,
        pix_fmts: &["p010le", "nv12"],
        quality: qsv_quality,
        input_args: &[],
        requires_hwupload: false,
    },
    Candidate {
        name: "hevc_qsv",
        codec: VideoCodec::Hevc,
        vendor: Some("intel"),
        hardware: true,
        pix_fmts: &["p010le", "nv12"],
        quality: qsv_quality,
        input_args: &[],
        requires_hwupload: false,
    },
    Candidate {
        name: "h264_qsv",
        codec: VideoCodec::H264,
        vendor: Some("intel"),
        hardware: true,
        pix_fmts: &["nv12"],
        quality: qsv_quality,
        input_args: &[],
        requires_hwupload: false,
    },
    Candidate {
        name: "av1_vaapi",
        codec: VideoCodec::Av1,
        vendor: None,
        hardware: true,
        pix_fmts: &["p010le", "nv12"],
        quality: vaapi_quality,
        input_args: &["-vaapi_device", "/dev/dri/renderD128"],
        requires_hwupload: true,
    },
    Candidate {
        name: "hevc_vaapi",
        codec: VideoCodec::Hevc,
        vendor: None,
        hardware: true,
        pix_fmts: &["p010le", "nv12"],
        quality: vaapi_quality,
        input_args: &["-vaapi_device", "/dev/dri/renderD128"],
        requires_hwupload: true,
    },
    Candidate {
        name: "libsvtav1",
        codec: VideoCodec::Av1,
        vendor: None,
        hardware: false,
        pix_fmts: &["yuv420p10le", "yuv420p"],
        quality: svtav1_quality,
        input_args: &[],
        requires_hwupload: false,
    },
    Candidate {
        name: "libx265",
        codec: VideoCodec::Hevc,
        vendor: None,
        hardware: false,
        pix_fmts: &["yuv420p10le", "yuv420p"],
        quality: x265_quality,
        input_args: &[],
        requires_hwupload: false,
    },
    Candidate {
        name: "libx264",
        codec: VideoCodec::H264,
        vendor: None,
        hardware: false,
        pix_fmts: &["yuv420p"],
        quality: x264_quality,
        input_args: &[],
        requires_hwupload: false,
    },
];

/// A probed FFmpeg installation.
pub struct Ffmpeg {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub version: String,
    pub config: String,
    encoders: Vec<String>,
    muxers: Vec<String>,
    hwaccels: Vec<String>,
    filters: Vec<String>,
    pix_fmt_cache: Mutex<HashMap<String, Vec<String>>>,
}

impl std::fmt::Debug for Ffmpeg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ffmpeg")
            .field("ffmpeg", &self.ffmpeg)
            .field("ffprobe", &self.ffprobe)
            .field("version", &self.version)
            .field("encoders", &self.encoders.len())
            .finish()
    }
}

impl Ffmpeg {
    /// `SR_FFMPEG` / `SR_PROBE` first, then `PATH` (with ffprobe looked up next
    /// to the ffmpeg binary before falling back to bare `ffprobe`).
    pub fn discover() -> Result<Self> {
        let ffmpeg = std::env::var_os("SR_FFMPEG")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" }));
        let ffprobe = match std::env::var_os("SR_FFPROBE") {
            Some(p) => PathBuf::from(p),
            None => {
                let sibling_name = if cfg!(windows) {
                    "ffprobe.exe"
                } else {
                    "ffprobe"
                };
                match ffmpeg.parent() {
                    Some(dir) if !dir.as_os_str().is_empty() => {
                        let candidate = dir.join(sibling_name);
                        if candidate.exists() {
                            candidate
                        } else {
                            PathBuf::from(sibling_name)
                        }
                    }
                    _ => PathBuf::from(sibling_name),
                }
            }
        };
        Ffmpeg::from_paths(ffmpeg, ffprobe)
    }

    pub fn from_paths(ffmpeg: PathBuf, ffprobe: PathBuf) -> Result<Self> {
        let version_text = capture(&ffmpeg, &args(&["-hide_banner", "-version"]))
            .map_err(|_| Error::ToolMissing(ffmpeg.display().to_string()))?;
        let (version, config) = parse_version(&version_text);
        // ffprobe must exist too: probing is not optional for this engine.
        capture(&ffprobe, &args(&["-hide_banner", "-version"]))
            .map_err(|_| Error::ToolMissing(ffprobe.display().to_string()))?;

        let encoders = parse_capability_list(&capture(&ffmpeg, &args(&["-hide_banner", "-encoders"]))?);
        let muxers = parse_capability_list(&capture(&ffmpeg, &args(&["-hide_banner", "-muxers"]))?);
        let hwaccels = parse_hwaccels(&capture(&ffmpeg, &args(&["-hide_banner", "-hwaccels"]))?);
        let filters = parse_capability_list(&capture(&ffmpeg, &args(&["-hide_banner", "-filters"]))?);

        Ok(Ffmpeg {
            ffmpeg,
            ffprobe,
            version,
            config,
            encoders,
            muxers,
            hwaccels,
            filters,
            pix_fmt_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn has_encoder(&self, name: &str) -> bool {
        self.encoders.iter().any(|e| e == name)
    }

    /// A compact version for UI chrome: `8.0.1-full_build` rather than
    /// `ffmpeg version 8.0.1-full_build-www.gyan.dev Copyright (c) 2000-2025 the
    /// FFmpeg developers`. The full line still lives in [`Self::version`] and in
    /// the job log, which is where a durable record belongs.
    pub fn version_short(&self) -> String {
        let mut tokens = self.version.split_whitespace();
        let token = tokens
            .nth(2)
            // Tolerate a shorter line (`ffmpeg version n8.0`) by taking whatever
            // follows, and a single-token line by using it as-is.
            .or_else(|| self.version.split_whitespace().nth(1))
            .unwrap_or(self.version.as_str());
        // Drop the packager's URL, which is the longest and least useful part.
        let trimmed = token
            .split("www.")
            .next()
            .unwrap_or(token)
            .trim_end_matches('-');
        if trimmed.is_empty() {
            token.to_string()
        } else {
            trimmed.to_string()
        }
    }

    pub fn has_muxer(&self, name: &str) -> bool {
        self.muxers.iter().any(|m| m == name)
    }

    pub fn has_filter(&self, name: &str) -> bool {
        self.filters.iter().any(|f| f == name)
    }

    pub fn has_hwaccel(&self, name: &str) -> bool {
        self.hwaccels.iter().any(|h| h == name)
    }

    pub fn encoders(&self) -> &[String] {
        &self.encoders
    }

    pub fn hwaccels(&self) -> &[String] {
        &self.hwaccels
    }

    /// `ffmpeg -h encoder=X` -> supported pixel formats (cached).
    pub fn encoder_pixel_formats(&self, encoder: &str) -> Vec<String> {
        if let Some(hit) = self.pix_fmt_cache.lock().get(encoder) {
            return hit.clone();
        }
        let formats = capture(
            &self.ffmpeg,
            &args(&["-hide_banner", "-h", &format!("encoder={encoder}")]),
        )
        .ok()
        .map(|text| parse_supported_pixel_formats(&text))
        .unwrap_or_default();
        self.pix_fmt_cache
            .lock()
            .insert(encoder.to_string(), formats.clone());
        formats
    }

    /// Every encoder this build can use, best first.
    ///
    /// The runner walks this list when an encoder turns out to be advertised but
    /// unusable — an AMD AMF encoder on an NVIDIA machine, for instance, is
    /// present in `-encoders` and still fails to open. Ordering is
    /// hardware-before-software, then the caller's codec preference, then the
    /// order of the candidate table (never alphabetical: `av1_amf` must not win
    /// over `av1_nvenc` because of its name).
    pub fn video_encoder_chain(&self, pref: &EncoderPreference) -> Vec<SelectedVideoEncoder> {
        if let Some(forced) = pref.forced.as_deref() {
            if !self.has_encoder(forced) {
                return Vec::new();
            }
            if let Some(cand) = CANDIDATES.iter().find(|c| c.name == forced) {
                let supported = self.encoder_pixel_formats(forced);
                let pix_fmt = pick_pix_fmt(cand.pix_fmts, &supported, pref.prefer_10bit)
                    .unwrap_or_else(|| "yuv420p".to_string());
                return vec![self.materialize(cand, pix_fmt, pref)];
            }
            let supported = self.encoder_pixel_formats(forced);
            let pix_fmt = pick_pix_fmt(
                &["p010le", "nv12", "yuv420p10le", "yuv420p"],
                &supported,
                pref.prefer_10bit,
            )
            .unwrap_or_else(|| "yuv420p".to_string());
            return vec![SelectedVideoEncoder {
                name: forced.to_string(),
                codec: VideoCodec::H264,
                vendor: None,
                hardware: false,
                pix_fmt,
                quality_args: Vec::new(),
                input_args: Vec::new(),
                requires_hwupload: false,
                note: "forced by SR_VIDEO_ENCODER; no quality tuning applied".to_string(),
            }];
        }

        let codec_weight = |c: VideoCodec| pref.codec_order.iter().position(|x| *x == c).unwrap_or(99);
        let mut ordered: Vec<(usize, &Candidate)> = CANDIDATES
            .iter()
            .enumerate()
            .filter(|(_, c)| pref.allow_software || c.hardware)
            .collect();
        ordered.sort_by_key(|(index, c)| {
            let hardware_rank = if c.hardware == pref.prefer_hardware { 0 } else { 1 };
            (hardware_rank, codec_weight(c.codec), *index)
        });

        let mut chain = Vec::new();
        for (_, cand) in ordered {
            if !self.has_encoder(cand.name) {
                continue;
            }
            let supported = self.encoder_pixel_formats(cand.name);
            if let Some(pix_fmt) = pick_pix_fmt(cand.pix_fmts, &supported, pref.prefer_10bit) {
                chain.push(self.materialize(cand, pix_fmt, pref));
            }
        }
        chain
    }

    /// Picks the best encoder this build can actually run.
    pub fn pick_video_encoder(&self, pref: &EncoderPreference) -> Result<SelectedVideoEncoder> {
        match self.video_encoder_chain(pref).into_iter().next() {
            Some(encoder) => Ok(encoder),
            None => Err(Error::NoEncoder {
                tried: CANDIDATES
                    .iter()
                    .map(|c| c.name)
                    .collect::<Vec<_>>()
                    .join(", "),
            }),
        }
    }

    fn materialize(
        &self,
        cand: &Candidate,
        pix_fmt: String,
        pref: &EncoderPreference,
    ) -> SelectedVideoEncoder {
        let ten_bit = pix_fmt.contains("10");
        let note = if ten_bit {
            format!("10-bit output via {pix_fmt}")
        } else {
            format!("8-bit output via {pix_fmt} (10-bit not available for this encoder)")
        };
        SelectedVideoEncoder {
            name: cand.name.to_string(),
            codec: cand.codec,
            vendor: cand.vendor.map(|v| v.to_string()),
            hardware: cand.hardware,
            pix_fmt,
            quality_args: (cand.quality)(pref.quality),
            input_args: args(cand.input_args),
            requires_hwupload: cand.requires_hwupload,
            note,
        }
    }

    /// Lossless first, then high-quality lossy. Used for the enhanced track.
    pub fn pick_audio_encoder(&self, want_lossless: bool) -> SelectedAudioEncoder {
        let lossless: &[&str] = &["flac", "alac", "tta"];
        let lossy: &[&str] = &["libopus", "aac", "libvorbis"];
        let order: Vec<&str> = if want_lossless {
            lossless.iter().chain(lossy.iter()).copied().collect()
        } else {
            lossy.iter().chain(lossless.iter()).copied().collect()
        };
        for name in order {
            if self.has_encoder(name) {
                let (args, lossless_hit) = match name {
                    "flac" => (vec!["-compression_level".to_string(), "8".to_string()], true),
                    "alac" => (vec![], true),
                    "libopus" => (
                        vec![
                            "-b:a".to_string(),
                            "256k".to_string(),
                            "-vbr".to_string(),
                            "on".to_string(),
                        ],
                        false,
                    ),
                    "aac" => (vec!["-b:a".to_string(), "320k".to_string()], false),
                    _ => (vec![], false),
                };
                return SelectedAudioEncoder {
                    name: name.to_string(),
                    lossless: lossless_hit,
                    args,
                };
            }
        }
        // `pcm_s16le` ships with every FFmpeg build; it is a safe last resort.
        SelectedAudioEncoder {
            name: "pcm_s16le".to_string(),
            lossless: true,
            args: vec![],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectedAudioEncoder {
    pub name: String,
    pub lossless: bool,
    pub args: Vec<String>,
}

impl SelectedAudioEncoder {
    pub fn describe(&self) -> String {
        format!(
            "{} ({})",
            self.name,
            if self.lossless { "lossless" } else { "lossy" }
        )
    }
}

fn pick_pix_fmt(
    preferred: &[&str],
    supported: &[String],
    prefer_10bit: bool,
) -> Option<String> {
    let mut order: Vec<&str> = preferred.to_vec();
    if !prefer_10bit {
        order.sort_by_key(|f| if f.contains("10") { 1 } else { 0 });
    }
    // An empty `supported` list means `-h encoder=` told us nothing; trust the
    // table rather than failing the job.
    if supported.is_empty() {
        return order.first().map(|s| s.to_string());
    }
    order
        .into_iter()
        .find(|f| supported.iter().any(|s| s == f))
        .map(|s| s.to_string())
}

pub fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

/// First line + configuration line of `ffmpeg -version`.
fn parse_version(text: &str) -> (String, String) {
    let mut version = String::new();
    let mut config = String::new();
    for line in text.lines() {
        if version.is_empty() {
            version = line.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("configuration:") {
            config = rest.trim().to_string();
        }
    }
    (version, config)
}

/// Parses `-encoders` / `-muxers` / `-filters` tables.
///
/// Every one of these tables is ` <flags> <name> <rest...>`, with a legend of
/// `  T.. = Timeline support` rows and a `------` / `--` rule above the data.
/// The legend rows are the only ones containing ` = `.
fn parse_capability_list(text: &str) -> Vec<String> {
    const FLAG_CHARS: &[char] = &[
        '.', 'V', 'A', 'S', 'D', 'E', 'F', 'X', 'B', 'T', 'C', 'N', '|',
    ];
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.ends_with(':') || trimmed.contains(" = ") {
            continue;
        }
        if trimmed.starts_with("--") {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let flags = match parts.next() {
            Some(f) => f,
            None => continue,
        };
        if !flags.chars().all(|c| FLAG_CHARS.contains(&c)) {
            continue;
        }
        if let Some(name) = parts.next() {
            if !name.is_empty() && !name.contains('=') {
                out.push(name.to_string());
            }
        }
    }
    out
}

fn parse_hwaccels(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.ends_with(':') && !l.contains("Hardware acceleration"))
        .map(|l| l.to_string())
        .collect()
}

fn parse_supported_pixel_formats(text: &str) -> Vec<String> {
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("Supported pixel formats:") {
            return rest.split_whitespace().map(|s| s.to_string()).collect();
        }
    }
    Vec::new()
}

/// True when this FFmpeg build advertises the given hardware acceleration.
pub fn hardware_accel_summary(ff: &Ffmpeg) -> String {
    if ff.hwaccels().is_empty() {
        "none".to_string()
    } else {
        ff.hwaccels().join(", ")
    }
}

/// Convenience: is this path an existing file we can read?
pub fn is_readable_file(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENCODERS: &str = "\
Encoders:
 V..... = Video
 A..... = Audio
 S..... = Subtitle
 .F.... = Frame-level multithreading
 ..S... = Slice-level multithreading
 ...X.. = Codec is experimental
 ....B. = Supports draw_horiz_band
 .....D = Supports direct rendering method 1
 ------
 V....D av1_nvenc            NVIDIA NVENC av1 encoder (codec av1)
 V....D hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)
 V....D libx264              libx264 H.264 / AVC (codec h264)
 A....D aac                  AAC (Advanced Audio Coding)
 A....D flac                 FLAC (Free Lossless Audio Codec)
";

    const FILTERS: &str = "\
Filters:
  T.. = Timeline support
  .S. = Slice threading
  ..C = Command support
  A = Audio input/output
  V = Video input/output
  N = Dynamic number and/or type of input/output
  | = Source or sink filter
 ... abench            A->A       Benchmark part of a filtergraph.
 ..C acopy             A->A       Copy an audio stream.
 ... bwdif             V->V       Deinterlace the input image.
 ... ebur128            N->N       EBU R128 scanner.
 ... idet              V->V       Interlace detect Filter.
";

    const MUXERS: &str = "\
File formats:
 D. = Demuxing supported
 .E = Muxing supported
 --
  E 3g2                 3GP2 (3GPP2 file format)
  E matroska            Matroska
  E mp4                 MP4 (MPEG-4 Part 14)
";

    #[test]
    fn parses_encoder_tables() {
        let encoders = parse_capability_list(ENCODERS);
        assert!(encoders.contains(&"av1_nvenc".to_string()));
        assert!(encoders.contains(&"flac".to_string()));
        assert!(!encoders.contains(&"Video".to_string()));
        assert_eq!(encoders.len(), 5);
    }

    #[test]
    fn parses_filter_tables_without_a_separator_rule() {
        let filters = parse_capability_list(FILTERS);
        assert!(filters.contains(&"bwdif".to_string()));
        assert!(filters.contains(&"idet".to_string()));
        assert!(filters.contains(&"ebur128".to_string()));
        assert!(!filters.contains(&"T..".to_string()));
        assert!(!filters.contains(&"Benchmark".to_string()));
    }

    #[test]
    fn parses_muxer_tables() {
        let muxers = parse_capability_list(MUXERS);
        assert!(muxers.contains(&"matroska".to_string()));
        assert!(muxers.contains(&"mp4".to_string()));
        assert!(!muxers.contains(&"Demuxing".to_string()));
    }

    #[test]
    fn parses_version_block() {
        let (v, c) = parse_version(
            "ffmpeg version 8.0.1-full_build Copyright (c) 2000-2025\nconfiguration: --enable-gpl --enable-nvenc\n",
        );
        assert!(v.starts_with("ffmpeg version 8.0.1"));
        assert!(c.contains("--enable-nvenc"));
    }

    #[test]
    fn short_version_drops_the_copyright_and_the_packager_url() {
        let ff = Ffmpeg {
            ffmpeg: PathBuf::from("ffmpeg"),
            ffprobe: PathBuf::from("ffprobe"),
            version: "ffmpeg version 8.0.1-full_build-www.gyan.dev Copyright (c) 2000-2025 the FFmpeg developers".into(),
            config: String::new(),
            encoders: vec![],
            muxers: vec![],
            hwaccels: vec![],
            filters: vec![],
            pix_fmt_cache: Mutex::new(HashMap::new()),
        };
        assert_eq!(ff.version_short(), "8.0.1-full_build");
    }

    #[test]
    fn short_version_survives_an_unexpected_version_line() {
        let ff = Ffmpeg {
            ffmpeg: PathBuf::from("ffmpeg"),
            ffprobe: PathBuf::from("ffprobe"),
            version: "something unusual".into(),
            config: String::new(),
            encoders: vec![],
            muxers: vec![],
            hwaccels: vec![],
            filters: vec![],
            pix_fmt_cache: Mutex::new(HashMap::new()),
        };
        assert_eq!(ff.version_short(), "unusual");
    }

    #[test]
    fn parses_hwaccels() {
        let hw = parse_hwaccels("Hardware acceleration methods:\ncuda\nd3d11va\ndxva2\n");
        assert_eq!(hw, vec!["cuda", "d3d11va", "dxva2"]);
    }

    #[test]
    fn parses_encoder_pixel_formats() {
        let text = "Encoder hevc_nvenc [NVIDIA NVENC hevc encoder]:\n    General capabilities: dr1 delay hardware\n    Supported pixel formats: yuv420p nv12 p010le yuv444p p016le\n";
        assert_eq!(
            parse_supported_pixel_formats(text),
            vec!["yuv420p", "nv12", "p010le", "yuv444p", "p016le"]
        );
    }

    #[test]
    fn pix_fmt_preference_respects_10bit_request() {
        let supported: Vec<String> = ["nv12", "p010le"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            pick_pix_fmt(&["p010le", "nv12"], &supported, true),
            Some("p010le".to_string())
        );
        assert_eq!(
            pick_pix_fmt(&["p010le", "nv12"], &supported, false),
            Some("nv12".to_string())
        );
    }

    #[test]
    fn unknown_pixel_format_support_falls_back_to_the_table() {
        // `-h encoder=` gave us nothing (stripped build): trust the table.
        assert_eq!(
            pick_pix_fmt(&["p010le", "nv12"], &[], true),
            Some("p010le".to_string())
        );
    }

    #[test]
    fn unsupported_encoder_reports_what_was_tried() {
        // A Ffmpeg struct with an empty encoder list: selection must fail loudly
        // rather than silently pick something unusable.
        let ff = Ffmpeg {
            ffmpeg: PathBuf::from("ffmpeg"),
            ffprobe: PathBuf::from("ffprobe"),
            version: "test".into(),
            config: String::new(),
            encoders: vec![],
            muxers: vec![],
            hwaccels: vec![],
            filters: vec![],
            pix_fmt_cache: Mutex::new(HashMap::new()),
        };
        let err = ff.pick_video_encoder(&EncoderPreference::default()).unwrap_err();
        assert!(matches!(err, Error::NoEncoder { .. }));
        assert!(err.to_string().contains("av1_nvenc"));
    }

    #[test]
    fn hardware_is_preferred_over_a_better_codec() {
        let ff = Ffmpeg {
            ffmpeg: PathBuf::from("ffmpeg"),
            ffprobe: PathBuf::from("ffprobe"),
            version: "test".into(),
            config: String::new(),
            // software AV1 and hardware HEVC both exist
            encoders: vec!["libsvtav1".into(), "hevc_nvenc".into()],
            muxers: vec![],
            hwaccels: vec![],
            filters: vec![],
            pix_fmt_cache: Mutex::new(HashMap::new()),
        };
        let picked = ff.pick_video_encoder(&EncoderPreference::default()).unwrap();
        assert_eq!(picked.name, "hevc_nvenc");
        assert!(picked.hardware);
    }

    #[test]
    fn the_candidate_table_order_decides_between_equal_encoders() {
        // `av1_amf` sorts before `av1_nvenc` alphabetically, and both are
        // hardware AV1. The table order must win, because an AMD encoder on an
        // NVIDIA machine is advertised and still refuses to open.
        let ff = Ffmpeg {
            ffmpeg: PathBuf::from("ffmpeg"),
            ffprobe: PathBuf::from("ffprobe"),
            version: "test".into(),
            config: String::new(),
            encoders: vec!["av1_amf".into(), "av1_nvenc".into()],
            muxers: vec![],
            hwaccels: vec![],
            filters: vec![],
            pix_fmt_cache: Mutex::new(HashMap::new()),
        };
        let chain = ff.video_encoder_chain(&EncoderPreference::default());
        let names: Vec<&str> = chain.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["av1_nvenc", "av1_amf"]);
    }

    #[test]
    fn the_chain_contains_every_usable_encoder_as_a_fallback() {
        let ff = Ffmpeg {
            ffmpeg: PathBuf::from("ffmpeg"),
            ffprobe: PathBuf::from("ffprobe"),
            version: "test".into(),
            config: String::new(),
            encoders: vec![
                "av1_nvenc".into(),
                "hevc_nvenc".into(),
                "libsvtav1".into(),
                "libx264".into(),
            ],
            muxers: vec![],
            hwaccels: vec![],
            filters: vec![],
            pix_fmt_cache: Mutex::new(HashMap::new()),
        };
        let chain = ff.video_encoder_chain(&EncoderPreference::default());
        // hardware first (AV1 then HEVC), then software (AV1 then H.264)
        assert_eq!(
            chain.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
            vec!["av1_nvenc", "hevc_nvenc", "libsvtav1", "libx264"]
        );
        assert!(chain.iter().any(|e| !e.hardware), "software must be a fallback");
    }
}
