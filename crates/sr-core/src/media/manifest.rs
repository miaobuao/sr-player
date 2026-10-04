//! Typed view of an ffprobe result.
//!
//! ffprobe is inconsistent about JSON types (`nb_frames` is a string, `width` is
//! a number, `bit_rate` is a string, absent fields appear as `"N/A"`), so the
//! raw shape is deserialized tolerantly and then normalised into this model.

use crate::time::{parse_frame_rate, Rational, Timestamp};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// ffprobe mixes strings and numbers; accept either, reject `"N/A"`.
mod de {
    use super::*;

    fn to_string_value(v: Value) -> Option<String> {
        match v {
            Value::String(s) => {
                let t = s.trim().to_string();
                if t.is_empty() || t.eq_ignore_ascii_case("n/a") {
                    None
                } else {
                    Some(t)
                }
            }
            Value::Number(n) => Some(n.to_string()),
            Value::Bool(b) => Some(b.to_string()),
            _ => None,
        }
    }

    pub fn string_opt<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
        Ok(Option::<Value>::deserialize(d)?.and_then(to_string_value))
    }

    pub fn u64_opt<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
        Ok(Option::<Value>::deserialize(d)?
            .and_then(to_string_value)
            .and_then(|s| s.parse::<u64>().ok()))
    }

    pub fn i64_opt<'de, D: Deserializer<'de>>(d: D) -> Result<Option<i64>, D::Error> {
        Ok(Option::<Value>::deserialize(d)?
            .and_then(to_string_value)
            .and_then(|s| s.parse::<i64>().ok()))
    }

    pub fn f64_opt<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
        Ok(Option::<Value>::deserialize(d)?
            .and_then(to_string_value)
            .and_then(|s| s.parse::<f64>().ok()))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Disposition {
    #[serde(default)]
    pub default: i64,
    #[serde(default)]
    pub forced: i64,
    #[serde(default)]
    pub attached_pic: i64,
    #[serde(default)]
    pub comment: i64,
    #[serde(default)]
    pub lyrics: i64,
}

impl Disposition {
    pub fn is_default(&self) -> bool {
        self.default != 0
    }
    pub fn is_forced(&self) -> bool {
        self.forced != 0
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamBase {
    #[serde(default, deserialize_with = "de::i64_opt")]
    pub index: Option<i64>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub codec_name: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub codec_long_name: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub profile: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub codec_tag_string: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub time_base: Option<String>,
    #[serde(default, deserialize_with = "de::i64_opt")]
    pub start_pts: Option<i64>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub start_time: Option<String>,
    #[serde(default, deserialize_with = "de::i64_opt")]
    pub duration_ts: Option<i64>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub duration: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub bit_rate: Option<String>,
    #[serde(default, deserialize_with = "de::u64_opt")]
    pub nb_frames: Option<u64>,
    #[serde(default)]
    pub disposition: Disposition,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
}

impl StreamBase {
    pub fn index(&self) -> i64 {
        self.index.unwrap_or(-1)
    }

    /// The stream's first timestamp, in seconds. A stream can carry an offset the
    /// container does not report.
    pub fn start_seconds(&self) -> Option<f64> {
        self.start_time.as_deref().and_then(|s| s.parse().ok())
    }

    pub fn time_base(&self) -> Option<Rational> {
        self.time_base.as_deref().and_then(|s| Rational::parse(s).ok())
    }

    /// Duration as an exact timestamp in the stream's own timebase.
    pub fn duration_ts_value(&self) -> Option<Timestamp> {
        match (self.duration_ts, self.time_base()) {
            (Some(ts), Some(tb)) => Some(Timestamp::new(ts, tb)),
            _ => None,
        }
    }

    pub fn language(&self) -> Option<&str> {
        self.tags.get("language").map(|s| s.as_str())
    }

    pub fn title(&self) -> Option<&str> {
        self.tags.get("title").map(|s| s.as_str())
    }

    pub fn bit_rate_bps(&self) -> Option<u64> {
        self.bit_rate.as_deref().and_then(|s| s.parse().ok())
    }

    /// `"eng"`/`"jpn"` or the title, whichever is more useful to a human.
    pub fn describe(&self) -> String {
        let codec = self.codec_name.as_deref().unwrap_or("?");
        let lang = self.language().unwrap_or("und");
        match self.title() {
            Some(t) => format!("{codec} ({lang}) \"{t}\""),
            None => format!("{codec} ({lang})"),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VideoStream {
    #[serde(flatten)]
    pub base: StreamBase,
    #[serde(default, deserialize_with = "de::u64_opt")]
    pub width: Option<u64>,
    #[serde(default, deserialize_with = "de::u64_opt")]
    pub height: Option<u64>,
    #[serde(default, deserialize_with = "de::u64_opt")]
    pub coded_width: Option<u64>,
    #[serde(default, deserialize_with = "de::u64_opt")]
    pub coded_height: Option<u64>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub sample_aspect_ratio: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub display_aspect_ratio: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub pix_fmt: Option<String>,
    #[serde(default, deserialize_with = "de::u64_opt")]
    pub bits_per_raw_sample: Option<u64>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub color_range: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub color_space: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub color_transfer: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub color_primaries: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub chroma_location: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub field_order: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub r_frame_rate: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub avg_frame_rate: Option<String>,
    #[serde(default, deserialize_with = "de::u64_opt")]
    pub has_b_frames: Option<u64>,
    /// Populated by [`crate::media::classify`], not by ffprobe.
    #[serde(default)]
    pub temporal: Option<crate::media::classify::TemporalReport>,
}

impl VideoStream {
    /// `avg_frame_rate` is the honest one for CFR content; `r_frame_rate` is the
    /// container's guess and is often the field rate of telecined material.
    pub fn fps(&self) -> Option<Rational> {
        self.avg_frame_rate
            .as_deref()
            .and_then(parse_frame_rate)
            .or_else(|| self.r_frame_rate.as_deref().and_then(parse_frame_rate))
    }

    pub fn r_fps(&self) -> Option<Rational> {
        self.r_frame_rate.as_deref().and_then(parse_frame_rate)
    }

    pub fn size(&self) -> Option<(u64, u64)> {
        match (self.width, self.height) {
            (Some(w), Some(h)) if w > 0 && h > 0 => Some((w, h)),
            _ => None,
        }
    }

    /// Sample aspect ratio, `"1:1"` -> `1/1`. Absent means square pixels.
    pub fn sar(&self) -> Rational {
        self.sample_aspect_ratio
            .as_deref()
            .and_then(parse_ratio_colon)
            .filter(|r| !r.is_zero())
            .unwrap_or(Rational::ONE)
    }

    pub fn dar(&self) -> Option<Rational> {
        self.display_aspect_ratio
            .as_deref()
            .and_then(parse_ratio_colon)
    }

    /// The raster the *image* actually occupies once non-square pixels are
    /// undone. AI stages must run on this, never on the stored raster.
    pub fn square_pixel_size(&self) -> Option<(u64, u64)> {
        let (w, h) = self.size()?;
        let sar = self.sar();
        if sar == Rational::ONE {
            return Some((w, h));
        }
        let scaled = (w as i128 * sar.num() as i128 / sar.den() as i128) as u64;
        Some((scaled.max(2), h))
    }

    pub fn is_hdr(&self) -> bool {
        let transfer = self.color_transfer.as_deref().unwrap_or("");
        let primaries = self.color_primaries.as_deref().unwrap_or("");
        matches!(transfer, "smpte2084" | "arib-std-b67") || primaries.starts_with("bt2020")
    }

    pub fn is_limited_range(&self) -> bool {
        matches!(self.color_range.as_deref(), Some("tv") | Some("limited"))
    }

    /// The container's field-order flag. Treat as a hint: [`super::classify`]
    /// checks the actual frames, because this flag lies regularly.
    pub fn declared_interlaced(&self) -> bool {
        matches!(
            self.field_order.as_deref(),
            Some("tt") | Some("bb") | Some("tb") | Some("bt")
        )
    }

    /// Frame count implied by `duration * fps`, for VFR/CFR comparison.
    pub fn expected_frames(&self) -> Option<u64> {
        let fps = self.fps()?;
        let dur = self.base.duration_ts_value()?.seconds_f64();
        Some((dur * fps.to_f64()).round() as u64)
    }

    pub fn describe(&self) -> String {
        let size = self
            .size()
            .map(|(w, h)| format!("{w}x{h}"))
            .unwrap_or_else(|| "?x?".into());
        let fps = self
            .fps()
            .map(|f| format!("{:.3} fps", f.to_f64()))
            .unwrap_or_else(|| "? fps".into());
        format!(
            "{} {size} {fps} {}",
            self.base.codec_name.as_deref().unwrap_or("?"),
            self.pix_fmt.as_deref().unwrap_or("?")
        )
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioStream {
    #[serde(flatten)]
    pub base: StreamBase,
    #[serde(default, deserialize_with = "de::u64_opt")]
    pub sample_rate: Option<u64>,
    #[serde(default, deserialize_with = "de::u64_opt")]
    pub channels: Option<u64>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub channel_layout: Option<String>,
    #[serde(default, deserialize_with = "de::u64_opt")]
    pub bits_per_raw_sample: Option<u64>,
    /// True when the source codec carries dynamic-range-control metadata
    /// (AC-3 / E-AC-3). Decoding such a stream with DRC applied would silently
    /// change the very dynamics we are about to measure.
    #[serde(default)]
    pub carries_drc_metadata: bool,
}

impl AudioStream {
    pub fn is_multichannel(&self) -> bool {
        self.channels.unwrap_or(0) > 2
    }

    pub fn describe(&self) -> String {
        let layout = self
            .channel_layout
            .clone()
            .unwrap_or_else(|| format!("{} ch", self.channels.unwrap_or(0)));
        format!(
            "{} {} {} Hz",
            self.base.codec_name.as_deref().unwrap_or("?"),
            layout,
            self.sample_rate.unwrap_or(0)
        )
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubtitleStream {
    #[serde(flatten)]
    pub base: StreamBase,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttachmentStream {
    #[serde(flatten)]
    pub base: StreamBase,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub filename: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub mimetype: Option<String>,
}

impl AttachmentStream {
    /// ffprobe reports attachment metadata inside `tags`, so fall back to it when
    /// the field was never lifted out.
    pub fn filename(&self) -> Option<&str> {
        self.filename
            .as_deref()
            .or_else(|| self.base.tags.get("filename").map(|s| s.as_str()))
    }

    pub fn mimetype(&self) -> Option<&str> {
        self.mimetype
            .as_deref()
            .or_else(|| self.base.tags.get("mimetype").map(|s| s.as_str()))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Chapter {
    #[serde(default, deserialize_with = "de::i64_opt")]
    pub id: Option<i64>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub time_base: Option<String>,
    #[serde(default, deserialize_with = "de::i64_opt")]
    pub start: Option<i64>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub start_time: Option<String>,
    #[serde(default, deserialize_with = "de::i64_opt")]
    pub end: Option<i64>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub end_time: Option<String>,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
}

impl Chapter {
    pub fn title(&self) -> Option<&str> {
        self.tags.get("title").map(|s| s.as_str())
    }

    pub fn start_seconds(&self) -> Option<f64> {
        self.start_time.as_deref().and_then(|s| s.parse().ok())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FormatInfo {
    #[serde(default, deserialize_with = "de::string_opt")]
    pub filename: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub format_name: Option<String>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub format_long_name: Option<String>,
    #[serde(default, deserialize_with = "de::u64_opt")]
    pub nb_streams: Option<u64>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub duration: Option<String>,
    /// Where the container's timestamps begin. Not always zero, and when it is not,
    /// the duration above counts the offset as content.
    #[serde(default, deserialize_with = "de::string_opt")]
    pub start_time: Option<String>,
    #[serde(default, deserialize_with = "de::u64_opt")]
    pub size: Option<u64>,
    #[serde(default, deserialize_with = "de::string_opt")]
    pub bit_rate: Option<String>,
    /// `-show_format` reports the container's own duration probe.
    #[serde(default, deserialize_with = "de::f64_opt")]
    pub probe_score: Option<f64>,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
}

impl FormatInfo {
    /// The container's first timestamp, in seconds.
    pub fn start_seconds(&self) -> Option<f64> {
        self.start_time.as_deref().and_then(|s| s.parse().ok())
    }

    pub fn duration_seconds(&self) -> Option<f64> {
        self.duration.as_deref().and_then(|s| s.parse().ok())
    }

    /// Duration as an exact timestamp in microseconds (ffprobe prints 6 dp).
    pub fn duration_ts(&self) -> Option<Timestamp> {
        let tb = Rational::new(1, 1_000_000).ok()?;
        let text = self.duration.as_deref()?;
        let rational = Rational::from_decimal_str(text).ok()?;
        let micros = rational
            .checked_mul(&Rational::from_i64(1_000_000))?
            .round_i64();
        Some(Timestamp::new(micros, tb))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MediaManifest {
    pub path: PathBuf,
    pub format: FormatInfo,
    #[serde(default)]
    pub video: Vec<VideoStream>,
    #[serde(default)]
    pub audio: Vec<AudioStream>,
    #[serde(default)]
    pub subtitles: Vec<SubtitleStream>,
    #[serde(default)]
    pub attachments: Vec<AttachmentStream>,
    #[serde(default)]
    pub chapters: Vec<Chapter>,
    /// ffprobe version string, recorded for reproducibility.
    #[serde(default)]
    pub probed_by: String,
}

impl MediaManifest {
    pub fn primary_video(&self) -> Option<&VideoStream> {
        self.video
            .iter()
            .find(|v| v.base.disposition.attached_pic == 0 && !is_cover_art(v))
            .or_else(|| {
                self.video
                    .iter()
                    .find(|v| v.base.disposition.attached_pic == 0)
            })
            .or_else(|| self.video.first())
    }

    pub fn primary_audio(&self) -> Option<&AudioStream> {
        self.audio
            .iter()
            .find(|a| a.base.disposition.is_default())
            .or_else(|| self.audio.first())
    }

    pub fn duration(&self) -> Option<Timestamp> {
        self.format
            .duration_ts()
            .or_else(|| self.primary_video().and_then(|v| v.base.duration_ts_value()))
    }

    pub fn duration_seconds(&self) -> f64 {
        self.duration().map(|t| t.seconds_f64()).unwrap_or(0.0)
    }

    /// How long the content actually runs.
    ///
    /// The container's duration counts a leading timestamp offset as content. A
    /// file whose first frame is at ten seconds reports eleven seconds of duration
    /// for one second of pictures, and every comparison against that number is
    /// wrong: a correct transcode looks ten seconds short, progress bars run to
    /// the wrong place, and an unattended run reports a good file as a failure.
    ///
    /// Measured from the format's start time when it has one, and from the primary
    /// video stream's otherwise, because a stream can carry the offset when the
    /// container does not.
    pub fn content_duration_seconds(&self) -> f64 {
        let raw = self.duration_seconds();
        let start = self
            .format
            .start_seconds()
            .or_else(|| {
                self.primary_video()
                    .and_then(|video| video.base.start_seconds())
            })
            .unwrap_or(0.0);
        if start <= 0.0 {
            raw
        } else {
            (raw - start).max(0.0)
        }
    }

    pub fn has_hdr_video(&self) -> bool {
        self.primary_video().map(|v| v.is_hdr()).unwrap_or(false)
    }

    /// A frame count that disagrees with `duration * fps` means the file is VFR
    /// (or the timestamps are broken). Either way, timestamps trump the rate.
    pub fn vfr_suspected(&self) -> bool {
        match self.primary_video() {
            Some(v) => match (v.base.nb_frames, v.expected_frames()) {
                (Some(actual), Some(expected)) if expected > 0 => {
                    let diff = (actual as i64 - expected as i64).unsigned_abs();
                    diff > (expected / 100).max(2)
                }
                _ => false,
            },
            None => false,
        }
    }

    /// Rows for the GUI's info panel.
    pub fn summary_lines(&self) -> Vec<(String, String)> {
        let mut rows = Vec::new();
        rows.push((
            "container".into(),
            self.format
                .format_long_name
                .clone()
                .or_else(|| self.format.format_name.clone())
                .unwrap_or_else(|| "?".into()),
        ));
        if let Some(d) = self.duration() {
            let secs = d.seconds_f64();
            rows.push((
                "duration".into(),
                format!("{} ({secs:.3} s)", d.format_hms()),
            ));
        }
        if let Some(size) = self.format.size {
            rows.push(("size".into(), format!("{:.2} GiB", size as f64 / 1_073_741_824.0)));
        }
        for (i, v) in self.video.iter().enumerate() {
            rows.push((format!("video #{i}"), v.describe()));
            if let Some((w, h)) = v.square_pixel_size() {
                let stored = v.size().map(|(a, b)| format!("{a}x{b}")).unwrap_or_default();
                if stored != format!("{w}x{h}") {
                    rows.push((
                        format!("video #{i} square px"),
                        format!("{w}x{h} (SAR {} from {stored})", v.sar()),
                    ));
                }
            }
            rows.push((
                format!("video #{i} colour"),
                format!(
                    "primaries={} transfer={} matrix={} range={}",
                    v.color_primaries.as_deref().unwrap_or("?"),
                    v.color_transfer.as_deref().unwrap_or("?"),
                    v.color_space.as_deref().unwrap_or("?"),
                    v.color_range.as_deref().unwrap_or("?")
                ),
            ));
            if let Some(t) = &v.temporal {
                rows.push((
                    format!("video #{i} cadence"),
                    format!(
                        "{} ({:.0}% confidence)",
                        t.mode.as_str(),
                        t.confidence * 100.0
                    ),
                ));
            }
            if v.declared_interlaced() {
                rows.push((
                    format!("video #{i} field order"),
                    format!("{} (container flag)", v.field_order.clone().unwrap_or_default()),
                ));
            }
        }
        for (i, a) in self.audio.iter().enumerate() {
            rows.push((format!("audio #{i}"), a.describe()));
            if a.carries_drc_metadata {
                rows.push((
                    format!("audio #{i} note"),
                    "carries DRC metadata; decoded with drc_scale=0".into(),
                ));
            }
        }
        for (i, s) in self.subtitles.iter().enumerate() {
            rows.push((format!("subtitle #{i}"), s.base.describe()));
        }
        if !self.attachments.is_empty() {
            rows.push(("attachments".into(), self.attachments.len().to_string()));
        }
        if !self.chapters.is_empty() {
            rows.push(("chapters".into(), self.chapters.len().to_string()));
        }
        rows
    }

    /// Sanity checks that must pass before we spend hours encoding.
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if self.video.is_empty() {
            problems.push("no video stream found".to_string());
        }
        if let Some(v) = self.primary_video() {
            if v.fps().is_none() {
                problems.push("video frame rate is unknown or zero".to_string());
            }
            if v.size().is_none() {
                problems.push("video dimensions are missing".to_string());
            }
            if v.is_hdr() {
                problems.push(
                    "HDR source: the SDR restoration path is bypassed by default".to_string(),
                );
            }
        }
        if self.duration_seconds() <= 0.0 {
            problems.push("duration is zero or unreadable".to_string());
        }
        if self.vfr_suspected() {
            problems.push(
                "frame count disagrees with duration x fps: treating as VFR, timestamps win"
                    .to_string(),
            );
        }
        problems
    }
}

/// Cover art rides in the video stream list on many containers. It must never
/// be mistaken for the feature.
fn is_cover_art(v: &VideoStream) -> bool {
    if v.base.disposition.attached_pic != 0 {
        return true;
    }
    let still_image = v
        .base
        .codec_name
        .as_deref()
        .map(|c| matches!(c, "mjpeg" | "png" | "bmp" | "gif" | "webp"))
        .unwrap_or(false);
    let small = v
        .size()
        .map(|(w, h)| w <= 2048 && h <= 2048)
        .unwrap_or(false);
    still_image && small
}

/// `"40:33"` / `"1:1"` -> `40/33`.
pub fn parse_ratio_colon(s: &str) -> Option<Rational> {
    let (a, b) = s.split_once(':')?;
    let num = a.trim().parse::<i64>().ok()?;
    let den = b.trim().parse::<i64>().ok()?;
    if den == 0 {
        return None;
    }
    Rational::new(num, den).ok()
}

/// Streams whose presence must survive a remux, for QC.
pub fn preservation_inventory(m: &MediaManifest) -> PreservationInventory {
    PreservationInventory {
        video_streams: m.video.len(),
        audio_streams: m.audio.len(),
        subtitle_streams: m.subtitles.len(),
        attachments: m.attachments.len(),
        chapters: m.chapters.len(),
        attachment_names: m
            .attachments
            .iter()
            .filter_map(|a| a.filename().map(|s| s.to_string()))
            .collect(),
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreservationInventory {
    pub video_streams: usize,
    pub audio_streams: usize,
    pub subtitle_streams: usize,
    pub attachments: usize,
    pub chapters: usize,
    pub attachment_names: Vec<String>,
}

/// Convenience for the CLI: does this path look like a media file at all?
pub fn looks_like_media(path: &Path) -> bool {
    const EXT: &[&str] = &[
        "mkv", "mp4", "m4v", "avi", "mov", "wmv", "mpg", "mpeg", "m2ts", "ts", "vob", "webm",
        "flv", "ogv", "rmvb", "rm", "divx", "mts", "iso", "mxf",
    ];
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| EXT.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A container duration counts a leading timestamp offset as content, and
    /// comparing against it declares a correct transcode short. The corpus found
    /// this on a real file; this is the rule in isolation.
    #[test]
    fn the_content_duration_excludes_a_leading_offset() {
        let mut manifest = MediaManifest::default();
        manifest.format.duration = Some("11.000000".into());
        manifest.format.start_time = Some("10.000000".into());
        // Eleven seconds of container for one second of pictures.
        assert_eq!(manifest.duration_seconds(), 11.0);
        assert_eq!(manifest.content_duration_seconds(), 1.0);

        // A stream can carry the offset when the container does not.
        manifest.format.start_time = None;
        let mut video = VideoStream::default();
        video.base.start_time = Some("10.000000".into());
        manifest.video.push(video);
        assert_eq!(manifest.content_duration_seconds(), 1.0);

        // And a normal file is unchanged: no offset, no subtraction, and a
        // negative offset is not allowed to invent runtime.
        manifest.format.start_time = Some("0.000000".into());
        manifest.video.clear();
        assert_eq!(manifest.content_duration_seconds(), 11.0);
        manifest.format.start_time = Some("-2.000000".into());
        assert_eq!(manifest.content_duration_seconds(), 11.0);
    }

    const SAMPLE: &str = r#"{
      "streams": [
        {
          "index": 0,
          "codec_name": "mpeg2video",
          "codec_long_name": "MPEG-2 video",
          "profile": "Main",
          "codec_type": "video",
          "width": 720,
          "height": 480,
          "pix_fmt": "yuv420p",
          "field_order": "tt",
          "color_range": "tv",
          "color_space": "smpte170m",
          "r_frame_rate": "30000/1001",
          "avg_frame_rate": "30000/1001",
          "time_base": "1/90000",
          "start_pts": 0,
          "start_time": "0.000000",
          "duration_ts": 6486480,
          "duration": "72.072000",
          "nb_frames": "2162",
          "bit_rate": "6000000",
          "sample_aspect_ratio": "8:9",
          "display_aspect_ratio": "4:3",
          "disposition": {"default": 1, "forced": 0, "attached_pic": 0},
          "tags": {"language": "eng"}
        },
        {
          "index": 1,
          "codec_name": "ac3",
          "codec_type": "audio",
          "sample_rate": "48000",
          "channels": 6,
          "channel_layout": "5.1(side)",
          "time_base": "1/48000",
          "duration_ts": 3459456,
          "nb_frames": "N/A",
          "disposition": {"default": 1},
          "tags": {"language": "eng", "title": "Main"}
        },
        {
          "index": 2,
          "codec_name": "subrip",
          "codec_type": "subtitle",
          "disposition": {"default": 0, "forced": 1},
          "tags": {"language": "chi"}
        },
        {
          "index": 3,
          "codec_name": "ttf",
          "codec_type": "attachment",
          "disposition": {"default": 0},
          "tags": {"filename": "Arial.ttf", "mimetype": "application/x-truetype-font"}
        }
      ],
      "chapters": [
        {"id": 1, "time_base": "1/1000", "start": 0, "start_time": "0.000000", "end": 60000, "end_time": "60.000000", "tags": {"title": "Opening"}},
        {"id": 2, "time_base": "1/1000", "start": 60000, "start_time": "60.000000", "end": 72072, "end_time": "72.072000", "tags": {"title": "End"}}
      ],
      "format": {
        "filename": "movie.vob",
        "nb_streams": 4,
        "format_name": "mpeg",
        "format_long_name": "MPEG-PS (MPEG-2 Program Stream)",
        "start_time": "0.000000",
        "duration": "72.072000",
        "size": "55000000",
        "bit_rate": "6100000",
        "probe_score": 100,
        "tags": {"title": "Test Film"}
      }
    }"#;

    /// Builds a manifest from the raw ffprobe `streams/chapters/format` shape.
    fn manifest_from_sample() -> MediaManifest {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default)]
            streams: Vec<Value>,
            #[serde(default)]
            chapters: Vec<Chapter>,
            format: FormatInfo,
        }
        let raw: Raw = serde_json::from_str(SAMPLE).unwrap();
        let mut video = Vec::new();
        let mut audio = Vec::new();
        let mut subtitles = Vec::new();
        let mut attachments = Vec::new();
        for stream in raw.streams {
            let kind = stream
                .get("codec_type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            match kind.as_str() {
                "video" => video.push(serde_json::from_value(stream).unwrap()),
                "audio" => audio.push(serde_json::from_value(stream).unwrap()),
                "subtitle" => subtitles.push(serde_json::from_value(stream).unwrap()),
                "attachment" => attachments.push(serde_json::from_value(stream).unwrap()),
                _ => {}
            }
        }
        MediaManifest {
            path: PathBuf::from("movie.vob"),
            format: raw.format,
            video,
            audio,
            subtitles,
            attachments,
            chapters: raw.chapters,
            probed_by: "test".into(),
        }
    }

    #[test]
    fn parses_a_dvd_style_manifest() {
        let m = manifest_from_sample();
        let v = m.primary_video().unwrap();
        assert_eq!(v.size(), Some((720, 480)));
        assert_eq!(v.fps().unwrap(), Rational::new(30000, 1001).unwrap());
        assert_eq!(v.sar(), Rational::new(8, 9).unwrap());
        assert!(v.declared_interlaced());
        assert!(v.is_limited_range());
        assert!(!v.is_hdr());
        assert_eq!(v.base.nb_frames, Some(2162));
        assert_eq!(m.audio.len(), 1);
        assert!(m.audio[0].is_multichannel());
        assert_eq!(m.subtitles.len(), 1);
        assert!(m.subtitles[0].base.disposition.is_forced());
        assert_eq!(m.attachments.len(), 1);
        assert_eq!(m.chapters.len(), 2);
        assert_eq!(m.chapters[0].title(), Some("Opening"));
        assert_eq!(m.format.duration_seconds(), Some(72.072));
    }

    #[test]
    fn non_square_pixels_are_undone_for_the_ai_working_raster() {
        let m = manifest_from_sample();
        let v = m.primary_video().unwrap();
        // 720x480 with 8:9 SAR is a 640x480 image stretched into 720 columns
        assert_eq!(v.square_pixel_size(), Some((640, 480)));
    }

    #[test]
    fn duration_parsing_is_exact_not_floating_point() {
        let m = manifest_from_sample();
        let d = m.duration().unwrap();
        assert_eq!(d.format_hms(), "00:01:12.072");
        // 72.072 s is exactly 72072000 microseconds
        assert_eq!(d.pts, 72_072_000);
        assert_eq!(d.tb, Rational::new(1, 1_000_000).unwrap());
    }

    #[test]
    fn nan_values_become_none_instead_of_panicking() {
        let m = manifest_from_sample();
        assert_eq!(m.audio[0].base.nb_frames, None);
        assert_eq!(m.audio[0].base.bit_rate, None);
    }

    #[test]
    fn summary_rows_cover_every_stream_class() {
        let m = manifest_from_sample();
        let rows = m.summary_lines();
        let keys: Vec<&str> = rows.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"container"));
        assert!(keys.contains(&"video #0"));
        assert!(keys.contains(&"video #0 square px"));
        assert!(keys.contains(&"video #0 colour"));
        assert!(keys.contains(&"audio #0"));
        assert!(keys.contains(&"subtitle #0"));
        assert!(keys.contains(&"chapters"));
    }

    #[test]
    fn validation_flags_interlaced_dvd_hints_without_blocking_it() {
        let m = manifest_from_sample();
        let problems = m.validate();
        // A 29.97 DVD whose frame count matches duration*fps is not VFR.
        assert!(
            !problems.iter().any(|p| p.contains("VFR")),
            "unexpected VFR flag: {problems:?}"
        );
        assert!(!problems.iter().any(|p| p.contains("no video stream")));
    }

    #[test]
    fn vfr_is_detected_when_frame_count_disagrees_with_duration() {
        let mut m = manifest_from_sample();
        let v = m.video.first_mut().unwrap();
        v.base.nb_frames = Some(1000); // wildly different from 2162
        assert!(m.vfr_suspected());
        assert!(m.validate().iter().any(|p| p.contains("VFR")));
    }

    #[test]
    fn hdr_is_detected_and_flagged() {
        let mut m = manifest_from_sample();
        let v = m.video.first_mut().unwrap();
        v.color_transfer = Some("smpte2084".into());
        v.color_primaries = Some("bt2020".into());
        assert!(m.has_hdr_video());
        assert!(m.validate().iter().any(|p| p.contains("HDR")));
    }

    #[test]
    fn preservation_inventory_counts_what_must_survive() {
        let m = manifest_from_sample();
        let inv = preservation_inventory(&m);
        assert_eq!(inv.audio_streams, 1);
        assert_eq!(inv.subtitle_streams, 1);
        assert_eq!(inv.attachments, 1);
        assert_eq!(inv.chapters, 2);
        assert_eq!(inv.attachment_names, vec!["Arial.ttf".to_string()]);
    }

    #[test]
    fn ratio_parsing_rejects_junk() {
        assert_eq!(parse_ratio_colon("40:33"), Rational::new(40, 33).ok());
        assert_eq!(parse_ratio_colon("0:1"), Some(Rational::ZERO));
        assert_eq!(parse_ratio_colon("1:0"), None);
        assert_eq!(parse_ratio_colon("N/A"), None);
    }

    #[test]
    fn media_extension_check() {
        assert!(looks_like_media(Path::new("a/b.MKV")));
        assert!(!looks_like_media(Path::new("notes.txt")));
    }
}
