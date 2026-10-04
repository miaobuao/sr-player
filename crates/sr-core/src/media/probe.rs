//! ffprobe invocation and normalisation into [`MediaManifest`].

use crate::error::{Error, Result};
use crate::ffmpeg::{args, capture, Ffmpeg};
use crate::media::manifest::{
    AttachmentStream, AudioStream, Chapter, FormatInfo, MediaManifest, SubtitleStream, VideoStream,
};
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct RawProbe {
    #[serde(default)]
    streams: Vec<Value>,
    #[serde(default)]
    chapters: Vec<Chapter>,
    format: FormatInfo,
}

/// Runs ffprobe and keeps the raw JSON alongside the parsed manifest, so a job
/// record never has to re-probe the source to explain what it saw.
pub fn probe_with_raw(ff: &Ffmpeg, path: &Path) -> Result<(MediaManifest, String)> {
    let text = probe_json(ff, path)?;
    let manifest = parse_manifest(&ff.version, path, &text)?;
    Ok((manifest, text))
}

pub fn probe(ff: &Ffmpeg, path: &Path) -> Result<MediaManifest> {
    probe_with_raw(ff, path).map(|(m, _)| m)
}

pub fn probe_json(ff: &Ffmpeg, path: &Path) -> Result<String> {
    if !path.exists() {
        return Err(Error::Probe {
            path: path.to_path_buf(),
            detail: "file does not exist".into(),
        });
    }
    let mut argv = args(&[
        "-hide_banner",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
    ]);
    argv.push(path.display().to_string());
    capture(&ff.ffprobe, &argv).map_err(|e| Error::Probe {
        path: path.to_path_buf(),
        detail: e.to_string(),
    })
}

/// Pure function: ffprobe JSON in, manifest out. `probed_by` is recorded for
/// reproducibility (`ffmpeg version 8.0.1-...`).
pub fn parse_manifest(probed_by: &str, path: &Path, json: &str) -> Result<MediaManifest> {
    let raw: RawProbe = serde_json::from_str(json).map_err(|e| Error::Probe {
        path: path.to_path_buf(),
        detail: format!("cannot parse ffprobe JSON: {e}"),
    })?;

    let mut manifest = MediaManifest {
        path: path.to_path_buf(),
        format: raw.format,
        probed_by: probed_by.to_string(),
        chapters: raw.chapters,
        ..Default::default()
    };

    for stream in raw.streams {
        let kind = stream
            .get("codec_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let unknown_stream = || {
            // A stream we cannot model must not fail the job, but it must be
            // visible: the mux stage maps by index and drops nothing silently.
            format!("unmodelled {kind} stream ignored")
        };
        match kind.as_str() {
            "video" => match serde_json::from_value::<VideoStream>(stream) {
                Ok(v) => manifest.video.push(v),
                Err(e) => tracing::warn!("video stream skipped: {e}"),
            },
            "audio" => match serde_json::from_value::<AudioStream>(stream) {
                Ok(mut a) => {
                    // AC-3 / E-AC-3 can carry dynamic range control metadata.
                    // Recorded here so the decode path pins `drc_scale=0`
                    // instead of applying the source's dynamics twice.
                    a.carries_drc_metadata = matches!(
                        a.base.codec_name.as_deref(),
                        Some("ac3") | Some("eac3")
                    );
                    manifest.audio.push(a);
                }
                Err(e) => tracing::warn!("audio stream skipped: {e}"),
            },
            "subtitle" => match serde_json::from_value::<SubtitleStream>(stream) {
                Ok(s) => manifest.subtitles.push(s),
                Err(e) => tracing::warn!("subtitle stream skipped: {e}"),
            },
            "attachment" => match serde_json::from_value::<AttachmentStream>(stream) {
                Ok(mut a) => {
                    if a.filename.is_none() {
                        a.filename = a.base.tags.get("filename").cloned();
                    }
                    if a.mimetype.is_none() {
                        a.mimetype = a.base.tags.get("mimetype").cloned();
                    }
                    manifest.attachments.push(a);
                }
                Err(e) => tracing::warn!("attachment stream skipped: {e}"),
            },
            _ => tracing::debug!("{}", unknown_stream()),
        }
    }

    Ok(manifest)
}

/// The ffprobe version string, for the reproducibility manifest.
pub fn ffprobe_version(ff: &Ffmpeg) -> String {
    capture(&ff.ffprobe, &args(&["-hide_banner", "-version"]))
        .ok()
        .and_then(|t| t.lines().next().map(|l| l.trim().to_string()))
        .unwrap_or_else(|| "unknown".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Rational;

    const MKV_SHAPE: &str = r#"{
      "streams": [
        {"index":0,"codec_name":"h264","codec_type":"video","width":1920,"height":1080,
         "pix_fmt":"yuv420p","r_frame_rate":"24000/1001","avg_frame_rate":"24000/1001",
         "time_base":"1/24000","duration_ts":144000,"duration":"6.006000","nb_frames":"144",
         "disposition":{"default":1,"forced":0,"attached_pic":0},"tags":{}},
        {"index":1,"codec_name":"eac3","codec_type":"audio","sample_rate":"48000","channels":6,
         "channel_layout":"5.1(side)","disposition":{"default":1},"tags":{"language":"eng"}},
        {"index":2,"codec_name":"ass","codec_type":"subtitle","disposition":{"default":1},
         "tags":{"language":"chi"}},
        {"index":3,"codec_name":"ttf","codec_type":"attachment","disposition":{"default":0},
         "tags":{"filename":"Font.ttf","mimetype":"application/x-truetype-font"}}
      ],
      "chapters": [
        {"id":1,"time_base":"1/1000","start":0,"start_time":"0.000000","end":6006,"end_time":"6.006000",
         "tags":{"title":"Chapter 1"}}
      ],
      "format": {"filename":"x.mkv","nb_streams":4,"format_name":"matroska,webm",
                 "duration":"6.006000","size":"1234567","bit_rate":"1600000","probe_score":100,"tags":{}}
    }"#;

    #[test]
    fn classifies_streams_by_type() {
        let m = parse_manifest("ffmpeg version 8.0.1", Path::new("x.mkv"), MKV_SHAPE).unwrap();
        assert_eq!(m.video.len(), 1);
        assert_eq!(m.audio.len(), 1);
        assert_eq!(m.subtitles.len(), 1);
        assert_eq!(m.attachments.len(), 1);
        assert_eq!(m.chapters.len(), 1);
        assert_eq!(m.probed_by, "ffmpeg version 8.0.1");
        assert!((m.duration_seconds() - 6.006).abs() < 1e-9);
    }

    #[test]
    fn attachment_metadata_is_lifted_out_of_tags() {
        let m = parse_manifest("x", Path::new("x.mkv"), MKV_SHAPE).unwrap();
        assert_eq!(m.attachments[0].filename.as_deref(), Some("Font.ttf"));
        assert_eq!(
            m.attachments[0].mimetype.as_deref(),
            Some("application/x-truetype-font")
        );
    }

    #[test]
    fn eac3_is_flagged_as_carrying_drc_metadata() {
        let m = parse_manifest("x", Path::new("x.mkv"), MKV_SHAPE).unwrap();
        assert!(m.audio[0].carries_drc_metadata);
    }

    #[test]
    fn video_stream_keeps_exact_rational_cadence() {
        let m = parse_manifest("x", Path::new("x.mkv"), MKV_SHAPE).unwrap();
        let v = m.primary_video().unwrap();
        assert_eq!(v.fps(), Some(Rational::new(24000, 1001).unwrap()));
        assert_eq!(v.size(), Some((1920, 1080)));
        assert_eq!(v.square_pixel_size(), Some((1920, 1080)));
        assert!(!v.declared_interlaced());
    }

    #[test]
    fn broken_json_is_a_probe_error() {
        let err = parse_manifest("x", Path::new("x.mkv"), "{not json").unwrap_err();
        assert!(matches!(err, Error::Probe { .. }));
        assert!(err.to_string().contains("cannot parse ffprobe JSON"));
    }

    #[test]
    fn missing_file_is_a_probe_error_not_a_panic() {
        let ff = match Ffmpeg::from_paths("ffmpeg".into(), "ffprobe".into()) {
            Ok(ff) => ff,
            Err(_) => return, // no FFmpeg in this environment
        };
        let err = probe(&ff, Path::new("definitely/not/here.mkv")).unwrap_err();
        assert!(matches!(err, Error::Probe { .. }));
    }
}
