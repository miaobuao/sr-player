//! PCM transport: streaming decode, streaming WAV write, and small in-memory
//! buffers for tests and short excerpts.
//!
//! A two hour 5.1 feature is roughly 8 GB as float32. Nothing in this module
//! keeps a whole film in memory: analysis consumes blocks as they arrive and the
//! remaster pass streams into a temporary 32-bit float WAV on disk.

use crate::error::{Error, Result};
use crate::events::{Reporter, Stage, StageProgress};
use crate::ffmpeg::{args, read_exact_or_eof, Ffmpeg, RunSpec, StreamingChild};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Interleaved float32 audio.
#[derive(Clone, Debug, Default)]
pub struct PcmBuffer {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl PcmBuffer {
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        PcmBuffer {
            sample_rate,
            channels,
            samples: Vec::new(),
        }
    }

    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }

    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.frames() as f64 / self.sample_rate as f64
        }
    }

    /// Mono downmix (average of all channels). Used for analysis only.
    pub fn mono(&self) -> Vec<f32> {
        let ch = self.channels as usize;
        if ch == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(self.frames());
        for frame in self.samples.chunks_exact(ch) {
            out.push(frame.iter().sum::<f32>() / ch as f32);
        }
        out
    }

    pub fn channel(&self, index: usize) -> Vec<f32> {
        let ch = self.channels as usize;
        if ch == 0 || index >= ch {
            return Vec::new();
        }
        self.samples.iter().skip(index).step_by(ch).copied().collect()
    }

    pub fn sample_peak(&self) -> f32 {
        self.samples
            .iter()
            .fold(0.0f32, |acc, s| acc.max(s.abs()))
    }

    pub fn is_finite(&self) -> bool {
        self.samples.iter().all(|s| s.is_finite())
    }

    /// Writes a 32-bit float WAV. FFmpeg reads these natively, so the remaster
    /// pass can hand a processed file straight to the encoder.
    pub fn write_wav(&self, path: &Path) -> Result<()> {
        let mut writer = WavWriter::create(path, self.sample_rate, self.channels)?;
        writer.write(&self.samples)?;
        writer.finish()
    }

    pub fn read_wav(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        parse_wav(&bytes)
    }
}

/// Streaming 32-bit float WAV writer.
pub struct WavWriter {
    inner: BufWriter<std::fs::File>,
    sample_rate: u32,
    channels: u16,
    frames_written: u64,
    bytes_written: u32,
    path: std::path::PathBuf,
    finalized: bool,
}

impl WavWriter {
    pub fn create(path: &Path, sample_rate: u32, channels: u16) -> Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
            }
        }
        let file = std::fs::File::create(path).map_err(|e| Error::io(path, e))?;
        let mut writer = WavWriter {
            inner: BufWriter::new(file),
            sample_rate,
            channels,
            frames_written: 0,
            bytes_written: 0,
            path: path.to_path_buf(),
            finalized: false,
        };
        writer.write_header(0)?;
        Ok(writer)
    }

    fn write_header(&mut self, data_bytes: u32) -> Result<()> {
        let block_align = (self.channels as u32) * 4;
        let byte_rate = self.sample_rate * block_align;
        let mut header = Vec::with_capacity(44);
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&(36u32.saturating_add(data_bytes)).to_le_bytes());
        header.extend_from_slice(b"WAVE");
        header.extend_from_slice(b"fmt ");
        header.extend_from_slice(&16u32.to_le_bytes());
        header.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        header.extend_from_slice(&self.channels.to_le_bytes());
        header.extend_from_slice(&self.sample_rate.to_le_bytes());
        header.extend_from_slice(&byte_rate.to_le_bytes());
        header.extend_from_slice(&(block_align as u16).to_le_bytes());
        header.extend_from_slice(&32u16.to_le_bytes()); // bits per sample
        header.extend_from_slice(b"data");
        header.extend_from_slice(&data_bytes.to_le_bytes());
        self.inner.write_all(&header).map_err(Error::BareIo)?;
        Ok(())
    }

    /// Appends interleaved samples.
    pub fn write(&mut self, samples: &[f32]) -> Result<()> {
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        self.inner.write_all(&bytes).map_err(Error::BareIo)?;
        self.bytes_written = self
            .bytes_written
            .saturating_add(bytes.len().min(u32::MAX as usize) as u32);
        if self.channels > 0 {
            self.frames_written += samples.len() as u64 / self.channels as u64;
        }
        Ok(())
    }

    pub fn frames_written(&self) -> u64 {
        self.frames_written
    }

    /// Patches the RIFF/data sizes and flushes.
    pub fn finish(mut self) -> Result<()> {
        self.finalize_inner()
    }

    fn finalize_inner(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.inner.flush().map_err(Error::BareIo)?;
        let data_bytes = self.bytes_written;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&self.path)
            .map_err(|e| Error::io(&self.path, e))?;
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(4)).map_err(Error::BareIo)?;
        file.write_all(&(36u32.saturating_add(data_bytes)).to_le_bytes())
            .map_err(Error::BareIo)?;
        file.seek(SeekFrom::Start(40)).map_err(Error::BareIo)?;
        file.write_all(&data_bytes.to_le_bytes())
            .map_err(Error::BareIo)?;
        file.sync_all().map_err(Error::BareIo)?;
        self.finalized = true;
        Ok(())
    }
}

impl Drop for WavWriter {
    fn drop(&mut self) {
        let _ = self.finalize_inner();
    }
}

fn parse_wav(bytes: &[u8]) -> Result<PcmBuffer> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(Error::Other("not a RIFF/WAVE file".into()));
    }
    let mut pos = 12usize;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits = 0u16;
    let mut format = 0u16;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + size).min(bytes.len());
        match id {
            b"fmt " => {
                if size >= 16 {
                    format = u16::from_le_bytes([bytes[body_start], bytes[body_start + 1]]);
                    channels =
                        u16::from_le_bytes([bytes[body_start + 2], bytes[body_start + 3]]);
                    sample_rate = u32::from_le_bytes([
                        bytes[body_start + 4],
                        bytes[body_start + 5],
                        bytes[body_start + 6],
                        bytes[body_start + 7],
                    ]);
                    bits = u16::from_le_bytes([bytes[body_start + 14], bytes[body_start + 15]]);
                }
            }
            b"data" => data = Some(&bytes[body_start..body_end]),
            _ => {}
        }
        pos = body_end + (size & 1);
    }
    let data = data.ok_or_else(|| Error::Other("WAV has no data chunk".into()))?;
    if format != 3 || bits != 32 {
        return Err(Error::Other(format!(
            "expected 32-bit float WAV, found format={format} bits={bits}"
        )));
    }
    let samples: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(PcmBuffer {
        sample_rate,
        channels,
        samples,
    })
}

/// Options for a streaming decode.
pub struct DecodeOptions {
    pub stream_index: usize,
    pub sample_rate: u32,
    /// `None` keeps the source layout, which is what a remaster must do.
    pub channels: Option<u16>,
    /// Set `drc_scale=0` explicitly (AC-3 / E-AC-3 would otherwise apply the
    /// source's own dynamic range control on top of ours).
    pub disable_drc: bool,
    pub expected_duration: Option<std::time::Duration>,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        DecodeOptions {
            stream_index: 0,
            sample_rate: 48_000,
            channels: None,
            disable_drc: true,
            expected_duration: None,
        }
    }
}

/// Streaming decoder: yields interleaved f32 blocks.
pub struct PcmReader {
    child: StreamingChild,
    stdout: std::io::BufReader<std::process::ChildStdout>,
    pub channels: u16,
    pub sample_rate: u32,
    bytes_read: u64,
    pub expected_bytes: Option<u64>,
}

impl PcmReader {
    pub fn open(
        ff: &Ffmpeg,
        path: &Path,
        opts: &DecodeOptions,
        reporter: &Reporter,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        // Nothing to do if the job was cancelled while we were planning.
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let mut argv = args(&["-hide_banner", "-nostdin", "-progress", "pipe:1", "-v", "error"]);
        if opts.disable_drc {
            // FFmpeg's AC-3 decoder defaults to drc_scale=1.
            argv.extend(args(&["-drc_scale", "0"]));
        }
        argv.push("-i".into());
        argv.push(path.display().to_string());
        argv.push("-map".into());
        argv.push(format!("0:a:{}", opts.stream_index));
        argv.extend(args(&["-vn", "-sn", "-dn"]));
        argv.push("-ar".into());
        argv.push(opts.sample_rate.to_string());
        if let Some(ch) = opts.channels {
            argv.push("-ac".into());
            argv.push(ch.to_string());
        }
        argv.extend(args(&["-f", "f32le", "-acodec", "pcm_f32le", "-"]));

        let mut spec = RunSpec::new(Stage::AudioAnalysis, "audio-decode")
            .stderr_level(crate::events::Level::Debug);
        if let Some(d) = opts.expected_duration {
            spec = spec.with_duration(d);
        }
        let mut child = StreamingChild::spawn(&ff.ffmpeg, &argv, reporter, &spec, false)?;
        let stdout = child.stdout.take().ok_or_else(|| Error::Stage {
            stage: Stage::AudioAnalysis.id().into(),
            detail: "ffmpeg produced no audio pipe".into(),
        })?;

        // Ask ffprobe-free: the channel count is whatever FFmpeg negotiated.
        let channels = opts.channels.unwrap_or(2);
        Ok(PcmReader {
            child,
            stdout,
            channels,
            sample_rate: opts.sample_rate,
            bytes_read: 0,
            expected_bytes: opts
                .expected_duration
                .map(|d| (d.as_secs_f64() * opts.sample_rate as f64 * channels as f64 * 4.0) as u64),
        })
    }

    /// Reads up to `frames` frames. Returns an empty vec at end of stream.
    pub fn next_block(&mut self, frames: usize, cancel: &AtomicBool) -> Result<Vec<f32>> {
        if cancel.load(Ordering::Relaxed) {
            self.child.kill();
            return Err(Error::Cancelled);
        }
        let channels = self.channels as usize;
        let mut bytes = vec![0u8; frames * channels * 4];
        let read = match read_exact_or_eof(&mut self.stdout, &mut bytes) {
            Ok(0) => return Ok(Vec::new()),
            Ok(n) => n,
            Err(e) => {
                self.child.kill();
                return Err(e);
            }
        };
        bytes.truncate(read - (read % 4));
        // A PCM block must end on a whole frame: a 3-sample tail of a 6-channel
        // stream is not a frame, and writing it produces a WAV that FFmpeg
        // rejects with "Invalid PCM packet".
        let frame_bytes = channels * 4;
        let whole = bytes.len() - (bytes.len() % frame_bytes);
        bytes.truncate(whole);
        self.bytes_read += bytes.len() as u64;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn progress_fraction(&self) -> Option<f32> {
        self.expected_bytes
            .filter(|e| *e > 0)
            .map(|expected| (self.bytes_read as f32 / expected as f32).clamp(0.0, 1.0))
    }

    pub fn finish(self) -> Result<()> {
        let Self { child, .. } = self;
        child.wait().map(|_| ())
    }

    pub fn kill(&mut self) {
        self.child.kill();
    }
}

/// Decodes a whole audio stream into memory. Only for short excerpts and tests.
pub fn decode_all(
    ff: &Ffmpeg,
    path: &Path,
    opts: &DecodeOptions,
    reporter: &Reporter,
    cancel: &AtomicBool,
) -> Result<PcmBuffer> {
    let mut reader = PcmReader::open(ff, path, opts, reporter, cancel)?;
    let channels = reader.channels;
    let sample_rate = reader.sample_rate;
    let mut out = PcmBuffer::new(sample_rate, channels);
    loop {
        let block = reader.next_block(48_000, cancel)?;
        if block.is_empty() {
            break;
        }
        out.samples.extend_from_slice(&block);
        if out.duration_seconds() > 3600.0 {
            reader.kill();
            return Err(Error::Unsupported(
                "in-memory decode refused for streams longer than an hour".into(),
            ));
        }
        if let Some(f) = reader.progress_fraction() {
            reporter.progress(StageProgress {
                stage: Stage::AudioAnalysis,
                fraction: Some(f),
                detail: format!("decoded {:.1}s", out.duration_seconds()),
                frames: None,
                fps: None,
                speed: None,
                out_time: None,
                eta: None,
            });
        }
    }
    reader.finish()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trips_float_samples() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.wav");
        let mut buf = PcmBuffer::new(48_000, 2);
        buf.samples = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25];
        buf.write_wav(&path).unwrap();
        let read = PcmBuffer::read_wav(&path).unwrap();
        assert_eq!(read.sample_rate, 48_000);
        assert_eq!(read.channels, 2);
        assert_eq!(read.samples, buf.samples);
        assert_eq!(read.frames(), 3);
        assert!((read.duration_seconds() - 3.0 / 48_000.0).abs() < 1e-12);
    }

    #[test]
    fn wav_header_sizes_are_patched_on_finish() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.wav");
        {
            let mut w = WavWriter::create(&path, 8_000, 1).unwrap();
            w.write(&[0.1; 100]).unwrap();
            w.write(&[0.2; 50]).unwrap();
            w.finish().unwrap();
        }
        let bytes = std::fs::read(&path).unwrap();
        let riff = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let data = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        assert_eq!(data, 150 * 4);
        assert_eq!(riff, 36 + 150 * 4);
        assert_eq!(PcmBuffer::read_wav(&path).unwrap().frames(), 150);
    }

    #[test]
    fn mono_downmix_averages_channels() {
        let mut buf = PcmBuffer::new(48_000, 2);
        buf.samples = vec![1.0, 0.0, 0.0, 1.0];
        assert_eq!(buf.mono(), vec![0.5, 0.5]);
        assert_eq!(buf.channel(0), vec![1.0, 0.0]);
        assert_eq!(buf.channel(1), vec![0.0, 1.0]);
    }

    #[test]
    fn peak_and_finiteness_reports_are_useful_for_qc() {
        let mut buf = PcmBuffer::new(48_000, 1);
        buf.samples = vec![0.5, -1.5, f32::NAN];
        assert_eq!(buf.sample_peak(), 1.5);
        assert!(!buf.is_finite());
    }

    #[test]
    fn truncated_wav_is_an_error_not_a_panic() {
        assert!(parse_wav(b"RIFF").is_err());
        assert!(parse_wav(b"NOTAWAVE!!!!").is_err());
    }
}
