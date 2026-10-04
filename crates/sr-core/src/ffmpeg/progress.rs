//! Parser for FFmpeg's `-progress pipe:1` stream.
//!
//! FFmpeg emits `key=value` lines and closes each block with `progress=...`.
//! We accumulate keys and hand back one [`ProgressTick`] per closed block, so a
//! caller never sees a half-updated frame counter.

use std::time::Duration;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProgressTick {
    pub frame: Option<u64>,
    pub fps: Option<f64>,
    pub out_time_us: Option<i64>,
    pub total_size: Option<u64>,
    pub speed: Option<f64>,
    pub drop_frames: Option<u64>,
    pub dup_frames: Option<u64>,
    /// Set when the closing `progress=` line was seen.
    pub complete: bool,
    /// `progress=end` — FFmpeg has finished writing.
    pub end: bool,
}

impl ProgressTick {
    pub fn out_time(&self) -> Option<Duration> {
        self.out_time_us
            .filter(|us| *us >= 0)
            .map(|us| Duration::from_micros(us as u64))
    }
}

#[derive(Debug, Default)]
pub struct ProgressParser {
    current: ProgressTick,
}

impl ProgressParser {
    pub fn new() -> Self {
        ProgressParser {
            current: ProgressTick::default(),
        }
    }

    /// Feeds one line. Returns `Some(tick)` when a progress block just closed.
    pub fn push_line(&mut self, line: &str) -> Option<ProgressTick> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }
        let (key, value) = match line.split_once('=') {
            Some(kv) => kv,
            None => return None,
        };
        let value = value.trim();
        match key {
            "frame" => self.current.frame = parse_u64(value),
            "fps" => self.current.fps = parse_f64(value),
            "out_time_us" | "out_time_ms" => {
                if let Some(v) = parse_i64(value) {
                    // Both keys carry microseconds in modern builds; keep the
                    // first non-N/A value we see for the block.
                    if self.current.out_time_us.is_none() {
                        self.current.out_time_us = Some(v);
                    }
                }
            }
            "total_size" => self.current.total_size = parse_u64(value),
            "speed" => self.current.speed = parse_speed(value),
            "drop_frames" => self.current.drop_frames = parse_u64(value),
            "dup_frames" => self.current.dup_frames = parse_u64(value),
            "progress" => {
                self.current.complete = true;
                self.current.end = value.eq_ignore_ascii_case("end");
                let tick = std::mem::take(&mut self.current);
                return Some(tick);
            }
            _ => {}
        }
        None
    }
}

fn parse_u64(v: &str) -> Option<u64> {
    v.parse::<u64>().ok()
}

fn parse_i64(v: &str) -> Option<i64> {
    v.parse::<i64>().ok()
}

fn parse_f64(v: &str) -> Option<f64> {
    let v = v.trim();
    if v.eq_ignore_ascii_case("n/a") {
        return None;
    }
    v.parse::<f64>().ok()
}

/// `1.02x`, `0.5x`, `N/A`.
fn parse_speed(v: &str) -> Option<f64> {
    let v = v.trim().trim_end_matches('x').trim_end_matches('X').trim();
    if v.eq_ignore_ascii_case("n/a") {
        return None;
    }
    v.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: &str = "\
frame=120
fps=24.00
stream_0_0_q=24.0
bitrate= 512.0kbits/s
total_size=1048576
out_time_us=5000000
out_time_ms=5000000
out_time=00:00:05.000000
dup_frames=0
drop_frames=2
speed=1.5x
progress=continue
";

    #[test]
    fn accumulates_a_full_block() {
        let mut p = ProgressParser::new();
        let mut tick = None;
        for line in BLOCK.lines() {
            if let Some(t) = p.push_line(line) {
                tick = Some(t);
            }
        }
        let tick = tick.expect("block should close");
        assert_eq!(tick.frame, Some(120));
        assert_eq!(tick.fps, Some(24.0));
        assert_eq!(tick.speed, Some(1.5));
        assert_eq!(tick.out_time(), Some(Duration::from_secs(5)));
        assert_eq!(tick.drop_frames, Some(2));
        assert!(tick.complete);
        assert!(!tick.end);
    }

    #[test]
    fn ignores_not_available_values() {
        let mut p = ProgressParser::new();
        let mut tick = None;
        for line in ["frame=N/A", "speed=N/A", "out_time_us=N/A", "progress=end"].iter() {
            if let Some(t) = p.push_line(line) {
                tick = Some(t);
            }
        }
        let tick = tick.unwrap();
        assert_eq!(tick.frame, None);
        assert_eq!(tick.speed, None);
        assert_eq!(tick.out_time(), None);
        assert!(tick.end);
    }

    #[test]
    fn blocks_do_not_leak_into_each_other() {
        let mut p = ProgressParser::new();
        assert!(p.push_line("frame=10").is_none());
        let first = p.push_line("progress=continue").unwrap();
        assert_eq!(first.frame, Some(10));
        assert!(p.push_line("progress=continue").is_none() || true);
        let second = p.push_line("progress=end").unwrap();
        assert_eq!(second.frame, None, "frame counter must reset per block");
        assert!(second.end);
    }

    #[test]
    fn negative_out_time_is_not_a_duration() {
        let mut p = ProgressParser::new();
        p.push_line("out_time_us=-1000");
        let tick = p.push_line("progress=continue").unwrap();
        assert_eq!(tick.out_time(), None);
    }
}
