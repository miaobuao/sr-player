//! Exact rational time.
//!
//! Every timestamp in the engine is `pts` plus an explicit timebase. Nothing in
//! this crate is allowed to reconstruct a timestamp from a frame index and a
//! floating point frame rate: that is how a two hour film ends up 1.7 s out of
//! sync at the end. All the arithmetic here is integer arithmetic with `i128`
//! intermediates so that both `24000/1001` cadences and `1/1000000` timebases
//! survive round trips.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::time::Duration;

#[derive(Debug, thiserror::Error, PartialEq, Eq, Clone)]
pub enum TimeError {
    #[error("denominator must not be zero")]
    ZeroDenominator,
    #[error("cannot parse {0:?} as a rational value")]
    Parse(String),
    #[error("arithmetic overflow")]
    Overflow,
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// A reduced rational number with a positive denominator.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rational {
    num: i64,
    den: i64,
}

impl Rational {
    pub const ZERO: Rational = Rational { num: 0, den: 1 };
    pub const ONE: Rational = Rational { num: 1, den: 1 };

    pub fn new(num: i64, den: i64) -> Result<Self, TimeError> {
        if den == 0 {
            return Err(TimeError::ZeroDenominator);
        }
        if num == 0 {
            return Ok(Rational::ZERO);
        }
        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
        let g = gcd(num.unsigned_abs(), den.unsigned_abs()) as i64;
        let g = g.max(1);
        Ok(Rational {
            num: num / g,
            den: den / g,
        })
    }

    pub const fn from_i64(v: i64) -> Self {
        Rational { num: v, den: 1 }
    }

    pub fn num(&self) -> i64 {
        self.num
    }

    pub fn den(&self) -> i64 {
        self.den
    }

    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    pub fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// `"24000/1001"`, `"24"`, `"25/1"`. Also tolerates ffprobe's `"0/0"`.
    pub fn parse(s: &str) -> Result<Self, TimeError> {
        let s = s.trim();
        match s.split_once('/') {
            Some((n, d)) => {
                let num = n
                    .trim()
                    .parse::<i64>()
                    .map_err(|_| TimeError::Parse(s.to_string()))?;
                let den = d
                    .trim()
                    .parse::<i64>()
                    .map_err(|_| TimeError::Parse(s.to_string()))?;
                if den == 0 {
                    return Err(TimeError::ZeroDenominator);
                }
                Rational::new(num, den)
            }
            None => {
                let num = s
                    .parse::<i64>()
                    .map_err(|_| TimeError::Parse(s.to_string()))?;
                Ok(Rational::from_i64(num))
            }
        }
    }

    /// Parses an exact decimal such as ffprobe's `"7200.123456"`, optionally
    /// with an exponent. No precision is lost.
    pub fn from_decimal_str(s: &str) -> Result<Self, TimeError> {
        let s = s.trim();
        if s.is_empty() {
            return Err(TimeError::Parse(s.to_string()));
        }
        let (mantissa, exp) = match s.find(['e', 'E']) {
            Some(i) => (
                &s[..i],
                s[i + 1..]
                    .parse::<i32>()
                    .map_err(|_| TimeError::Parse(s.to_string()))?,
            ),
            None => (s, 0),
        };
        let (int_part, frac_part) = match mantissa.find('.') {
            Some(i) => (&mantissa[..i], &mantissa[i + 1..]),
            None => (mantissa, ""),
        };
        let neg = int_part.starts_with('-');
        let int_digits = int_part.trim_start_matches(['-', '+']);
        if int_digits.is_empty() && frac_part.is_empty() {
            return Err(TimeError::Parse(s.to_string()));
        }
        if !int_digits.chars().all(|c| c.is_ascii_digit())
            || !frac_part.chars().all(|c| c.is_ascii_digit())
        {
            return Err(TimeError::Parse(s.to_string()));
        }
        let digits = format!("{int_digits}{frac_part}");
        let mut num: i64 = if digits.is_empty() {
            0
        } else {
            digits
                .parse::<i64>()
                .map_err(|_| TimeError::Parse(s.to_string()))?
        };
        if neg {
            num = -num;
        }
        let mut den: i64 = 10i64
            .checked_pow(frac_part.len() as u32)
            .ok_or(TimeError::Overflow)?;
        if exp > 0 {
            num = num
                .checked_mul(10i64.checked_pow(exp as u32).ok_or(TimeError::Overflow)?)
                .ok_or(TimeError::Overflow)?;
        } else if exp < 0 {
            den = den
                .checked_mul(10i64.checked_pow((-exp) as u32).ok_or(TimeError::Overflow)?)
                .ok_or(TimeError::Overflow)?;
        }
        Rational::new(num, den)
    }

    pub fn checked_mul(&self, other: &Rational) -> Option<Rational> {
        let g1 = gcd(self.num.unsigned_abs(), other.den.unsigned_abs()).max(1) as i64;
        let g2 = gcd(other.num.unsigned_abs(), self.den.unsigned_abs()).max(1) as i64;
        let a = self.num / g1;
        let d = other.den / g1;
        let c = other.num / g2;
        let b = self.den / g2;
        let num = a.checked_mul(c)?;
        let den = b.checked_mul(d)?;
        Rational::new(num, den).ok()
    }

    pub fn checked_div(&self, other: &Rational) -> Option<Rational> {
        if other.num == 0 {
            return None;
        }
        self.checked_mul(&other.inverse().ok()?)
    }

    pub fn inverse(&self) -> Result<Rational, TimeError> {
        if self.num == 0 {
            return Err(TimeError::ZeroDenominator);
        }
        Rational::new(self.den, self.num)
    }

    pub fn checked_add(&self, other: &Rational) -> Option<Rational> {
        let n = self.num as i128 * other.den as i128 + other.num as i128 * self.den as i128;
        let d = self.den as i128 * other.den as i128;
        let n = i64::try_from(n).ok()?;
        let d = i64::try_from(d).ok()?;
        Rational::new(n, d).ok()
    }

    /// Rounds to the nearest integer, halves away from zero.
    pub fn round_i64(&self) -> i64 {
        let sign: i128 = if self.num < 0 { -1 } else { 1 };
        let n = (self.num as i128 * sign + self.den as i128 / 2) / self.den as i128;
        (n * sign) as i64
    }

    /// Best rational approximation with a denominator of at most `max_den`,
    /// via continued-fraction convergents. Used for display and for recovering
    /// a friendly cadence (`23.976 -> 24000/1001`).
    pub fn approx(v: f64, max_den: i64) -> Rational {
        if !v.is_finite() || max_den <= 0 {
            return Rational::ZERO;
        }
        let neg = v < 0.0;
        let mut frac = v.abs();
        // convergents h(n-2) = p0/q0, h(n-1) = p1/q1
        let (mut p0, mut q0) = (0i64, 1i64);
        let (mut p1, mut q1) = (1i64, 0i64);
        let mut best = Rational::ZERO;
        for _ in 0..64 {
            let a = frac.floor();
            if !a.is_finite() {
                break;
            }
            let a = a as i64;
            let p2 = match a.checked_mul(p1).and_then(|x| x.checked_add(p0)) {
                Some(x) => x,
                None => break,
            };
            let q2 = match a.checked_mul(q1).and_then(|x| x.checked_add(q0)) {
                Some(x) => x,
                None => break,
            };
            if q2 <= 0 || q2 > max_den {
                break;
            }
            best = match Rational::new(p2, q2) {
                Ok(r) => r,
                Err(_) => break,
            };
            let rem = frac - a as f64;
            if rem.abs() <= f64::EPSILON {
                break;
            }
            frac = 1.0 / rem;
            p0 = p1;
            q0 = q1;
            p1 = p2;
            q1 = q2;
        }
        if neg {
            Rational::new(-best.num(), best.den()).unwrap_or(Rational::ZERO)
        } else {
            best
        }
    }
}

impl fmt::Debug for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.num, self.den)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// `24000/1001` -> `1001/24000` (the frame duration) and back.
pub fn parse_frame_rate(s: &str) -> Option<Rational> {
    let r = Rational::parse(s).ok()?;
    if r.is_zero() {
        None
    } else {
        Some(r)
    }
}

/// A timestamp: an integer `pts` in a fixed timebase.
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct Timestamp {
    pub pts: i64,
    pub tb: Rational,
}

impl Timestamp {
    pub fn new(pts: i64, tb: Rational) -> Self {
        Timestamp { pts, tb }
    }

    pub fn zero(tb: Rational) -> Self {
        Timestamp { pts: 0, tb }
    }

    /// Exact value as a rational number of seconds.
    pub fn seconds_rational(&self) -> Rational {
        Rational::new(self.pts, 1)
            .ok()
            .and_then(|p| p.checked_mul(&self.tb))
            .unwrap_or(Rational::ZERO)
    }

    pub fn seconds_f64(&self) -> f64 {
        self.pts as f64 * self.tb.to_f64()
    }

    pub fn duration(&self) -> Duration {
        let secs = self.seconds_f64();
        if !secs.is_finite() || secs <= 0.0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(secs)
        }
    }

    /// Re-expresses the same instant in another timebase.
    pub fn convert_to(&self, tb: Rational) -> Timestamp {
        // pts' = pts * tb / tb'
        let ratio = self
            .tb
            .checked_div(&tb)
            .unwrap_or(Rational::ONE);
        let pts = (self.pts as i128 * ratio.num() as i128 / ratio.den() as i128) as i64;
        Timestamp { pts, tb }
    }

    pub fn from_seconds(secs: f64, tb: Rational) -> Timestamp {
        let pts = (secs * tb.den() as f64 / tb.num() as f64).round();
        let pts = if pts.is_finite() { pts as i64 } else { 0 };
        Timestamp { pts, tb }
    }

    /// Offset by an exact number of seconds, staying in the same timebase.
    pub fn offset(&self, delta: Rational) -> Timestamp {
        let d = delta
            .checked_div(&self.tb)
            .map(|r| r.round_i64())
            .unwrap_or(0);
        Timestamp {
            pts: self.pts.saturating_add(d),
            tb: self.tb,
        }
    }

    pub fn max(&self, other: Timestamp) -> Timestamp {
        if self.cmp_time(&other) == Ordering::Less {
            other
        } else {
            *self
        }
    }

    /// Exact comparison across arbitrary timebases.
    pub fn cmp_time(&self, other: &Timestamp) -> Ordering {
        let l = self.pts as i128 * self.tb.num() as i128 * other.tb.den() as i128;
        let r = other.pts as i128 * other.tb.num() as i128 * self.tb.den() as i128;
        l.cmp(&r)
    }

    /// `HH:MM:SS.mmm`, the format every human-facing surface uses.
    pub fn format_hms(&self) -> String {
        let total = self.seconds_f64();
        if !total.is_finite() || total < 0.0 {
            return "--:--:--.---".to_string();
        }
        let ms = (total * 1000.0).round() as u64;
        let (h, m, s, millis) = (
            ms / 3_600_000,
            (ms / 60_000) % 60,
            (ms / 1000) % 60,
            ms % 1000,
        );
        format!("{h:02}:{m:02}:{s:02}.{millis:03}")
    }

    /// The decimal-seconds form FFmpeg's CLI accepts (microsecond precision).
    pub fn ffmpeg_seconds(&self) -> String {
        format!("{:.6}", self.seconds_f64())
    }
}

impl PartialEq for Timestamp {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_time(other) == Ordering::Equal
    }
}

impl Eq for Timestamp {}

impl PartialOrd for Timestamp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp_time(other))
    }
}

impl Ord for Timestamp {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_time(other)
    }
}

impl fmt::Debug for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({}/{})", self.format_hms(), self.pts, self.tb)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_hms())
    }
}

/// Half-open time interval.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TimeRange {
    pub start: Timestamp,
    pub end: Timestamp,
}

impl TimeRange {
    pub fn new(start: Timestamp, end: Timestamp) -> Self {
        TimeRange { start, end }
    }

    pub fn duration_seconds(&self) -> f64 {
        (self.end.seconds_f64() - self.start.seconds_f64()).max(0.0)
    }

    pub fn contains(&self, t: &Timestamp) -> bool {
        self.start.cmp_time(t) != Ordering::Greater && self.end.cmp_time(t) == Ordering::Greater
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_normalizes_and_keeps_sign_in_numerator() {
        let r = Rational::new(24000, -1001).unwrap();
        assert_eq!(r.num(), -24000);
        assert_eq!(r.den(), 1001);
        assert_eq!(Rational::new(50, 100).unwrap(), Rational::new(1, 2).unwrap());
        assert!(Rational::new(1, 0).is_err());
    }

    #[test]
    fn parses_ffprobe_frame_rates() {
        assert_eq!(parse_frame_rate("24000/1001"), Some(Rational::new(24000, 1001).unwrap()));
        assert_eq!(parse_frame_rate("25/1"), Some(Rational::from_i64(25)));
        assert_eq!(parse_frame_rate("0/0"), None);
        assert_eq!(parse_frame_rate("N/A"), None);
    }

    #[test]
    fn decimal_parse_is_exact() {
        let d = Rational::from_decimal_str("7200.123456").unwrap();
        assert_eq!(d, Rational::new(7200123456, 1000000).unwrap());
        // 7200.123456 s at microsecond resolution is exactly 7200123456 us
        assert_eq!(d.checked_mul(&Rational::new(1_000_000, 1).unwrap()).unwrap().num(), 7200123456);
        assert_eq!(
            Rational::from_decimal_str("-1.5").unwrap(),
            Rational::new(-3, 2).unwrap()
        );
        assert!(Rational::from_decimal_str("N/A").is_err());
    }

    #[test]
    fn timestamps_compare_exactly_across_timebases() {
        let fps = Rational::new(24000, 1001).unwrap();
        let tb = fps.inverse().unwrap(); // 1001/24000
        let one_hour_frames = 86400i64; // 1h at 24fps
        let t = Timestamp::new(one_hour_frames, tb);
        // 86400 frames * 1001/24000 = 3603.6 s
        assert!((t.seconds_f64() - 3603.6).abs() < 1e-9);
        let other_tb = Rational::new(1, 1000).unwrap();
        let same = Timestamp::from_seconds(t.seconds_f64(), other_tb);
        assert_eq!(t.cmp_time(&same), Ordering::Equal);
    }

    #[test]
    fn conversion_between_timebases_round_trips() {
        let fps = Rational::new(30000, 1001).unwrap();
        let tb = fps.inverse().unwrap();
        let ts = Timestamp::new(12345, tb);
        let ms = ts.convert_to(Rational::new(1, 1000).unwrap());
        let back = ms.convert_to(tb);
        // within one frame at most (integer pts rounding)
        let frames = (ts.pts - back.pts).abs();
        assert!(frames <= 1, "drift of {frames} frames");
    }

    #[test]
    fn no_float_drift_over_a_feature_length_timeline() {
        // 2h feature at 23.976: naive frame_index/fps accumulates a visible error
        // when the fps is stored as f64 (23.976 != 24000/1001). Rational must not.
        let fps = Rational::new(24000, 1001).unwrap();
        let tb = fps.inverse().unwrap();
        let frames = 172_800i64; // 2 hours of frames
        let exact = Timestamp::new(frames, tb);
        let naive = frames as f64 / 23.976;
        let exact_secs = exact.seconds_f64();
        assert!(
            (exact_secs - naive).abs() > 0.005,
            "this test is only meaningful if naive math actually drifts"
        );
        // rational answer is exactly frames * 1001/24000
        assert_eq!(exact.seconds_rational(), Rational::new(frames * 1001, 24000).unwrap());
    }

    #[test]
    fn hms_formatting() {
        let tb = Rational::new(1, 1000).unwrap();
        let t = Timestamp::new(3_603_600, tb);
        assert_eq!(t.format_hms(), "01:00:03.600");
        assert_eq!(Timestamp::zero(tb).format_hms(), "00:00:00.000");
    }

    #[test]
    fn approx_finds_sane_denominators() {
        let r = Rational::approx(23.976023976, 1001);
        assert!(r.den() <= 1001);
        assert!((r.to_f64() - 23.976023976).abs() < 0.001);
    }

    #[test]
    fn range_contains_is_half_open() {
        let tb = Rational::new(1, 1000).unwrap();
        let r = TimeRange::new(Timestamp::new(1000, tb), Timestamp::new(2000, tb));
        assert!(r.contains(&Timestamp::new(1000, tb)));
        assert!(r.contains(&Timestamp::new(1999, tb)));
        assert!(!r.contains(&Timestamp::new(2000, tb)));
    }
}
