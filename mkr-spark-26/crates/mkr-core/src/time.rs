//! Minimal calendar helpers (no chrono): civil <-> days conversion, date parsing, formatting.
use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_unix() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm).
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub fn format_date(unix: i64) -> String {
    let (y, m, d) = civil_from_days(unix.div_euclid(86400));
    format!("{y:04}-{m:02}-{d:02}")
}

const MONTHS: [&str; 12] = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];

fn month_from_name(s: &str) -> Option<i64> {
    let s = s.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase();
    if s.len() < 3 {
        return None;
    }
    MONTHS.iter().position(|m| m.starts_with(&s[..3.min(s.len())]) && (s.len() >= 3) && m.starts_with(&s) || (s.len() == 3 && m.starts_with(&s))).map(|i| i as i64 + 1)
}

fn valid(y: i64, m: i64, d: i64) -> bool {
    (1970..=2200).contains(&y) && (1..=12).contains(&m) && (1..=31).contains(&d)
}

/// Parse common publication-date formats: ISO (2026-11-02, 2026-11-02T10:00:00Z, 2026/11/02),
/// "November 2, 2026", "2 November 2026", "Nov 2 2026". Returns unix seconds at 00:00 UTC.
pub fn parse_date(s: &str) -> Option<i64> {
    let s = s.trim();
    let b = s.as_bytes();
    // ISO-like: YYYY-MM-DD or YYYY/MM/DD
    if b.len() >= 10 && b[..4].iter().all(u8::is_ascii_digit) && (b[4] == b'-' || b[4] == b'/') && b[5..7].iter().all(u8::is_ascii_digit) && (b[7] == b'-' || b[7] == b'/') && b[8..10].iter().all(u8::is_ascii_digit) {
        let (y, m, d) = (s[..4].parse().ok()?, s[5..7].parse().ok()?, s[8..10].parse().ok()?);
        return valid(y, m, d).then(|| days_from_civil(y, m, d) * 86400);
    }
    let words: Vec<&str> = s.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()).collect();
    for i in 0..words.len().saturating_sub(2) {
        // "Month D YYYY"
        if let (Some(m), Ok(d), Ok(y)) = (month_from_name(words[i]), words[i + 1].trim_end_matches(|c: char| c.is_alphabetic()).parse::<i64>(), words[i + 2].parse::<i64>()) {
            if valid(y, m, d) {
                return Some(days_from_civil(y, m, d) * 86400);
            }
        }
        // "D Month YYYY"
        if let (Ok(d), Some(m), Ok(y)) = (words[i].trim_end_matches(|c: char| c.is_alphabetic()).parse::<i64>(), month_from_name(words[i + 1]), words[i + 2].parse::<i64>()) {
            if valid(y, m, d) {
                return Some(days_from_civil(y, m, d) * 86400);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn civil_roundtrip() {
        for z in [-1000i64, 0, 1, 365, 10957, 20000, 20406] {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z);
        }
        assert_eq!(format_date(0), "1970-01-01");
        assert_eq!(format_date(days_from_civil(2026, 11, 14) * 86400), "2026-11-14");
    }
    #[test]
    fn parses_formats() {
        let want = days_from_civil(2026, 11, 2) * 86400;
        for s in ["2026-11-02", "2026-11-02T10:00:00Z", "2026/11/02", "November 2, 2026", "2 November 2026", "Nov 2 2026", "Published on 2 Nov 2026 by X"] {
            assert_eq!(parse_date(s), Some(want), "{s}");
        }
        assert_eq!(parse_date("no date here"), None);
        assert_eq!(parse_date("2026-13-45"), None);
    }
}
