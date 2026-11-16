//! HTTP client: timeouts, bounded retries with backoff, manual redirect following with per-hop URL
//! validation, response size cap, content-type allow-list, charset decoding (UTF-8, Latin-1, windows-1256).
use crate::security::{validate_url, UrlPolicy};
use mkr_core::{MkrError, Result};
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct HttpConfig {
    pub timeout: Duration,
    pub retries: u32,
    pub backoff_ms: u64,
    pub max_bytes: usize,
    pub max_redirects: usize,
    pub user_agent: String,
    pub policy: UrlPolicy,
}

impl Default for HttpConfig {
    fn default() -> Self {
        HttpConfig { timeout: Duration::from_secs(12), retries: 2, backoff_ms: 400, max_bytes: 2 * 1024 * 1024, max_redirects: 5, user_agent: "MKR-SPARK-26/0.1".into(), policy: UrlPolicy::default() }
    }
}

#[derive(Debug)]
pub struct Response {
    pub url: String,
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
    pub truncated: bool,
}

pub struct HttpClient {
    agent: ureq::Agent,
    pub cfg: HttpConfig,
    pub requests: AtomicUsize,
}

fn type_allowed(ct: &str) -> bool {
    let ct = ct.split(';').next().unwrap_or("").trim().to_lowercase();
    ct.is_empty() || ct.starts_with("text/") || ct == "application/json" || ct.ends_with("+json") || ct == "application/xml" || ct.ends_with("+xml") || ct == "application/xhtml+xml"
}

impl HttpClient {
    pub fn new(cfg: HttpConfig) -> HttpClient {
        let b = ureq::AgentBuilder::new().redirects(0).timeout_connect(cfg.timeout).timeout(cfg.timeout).user_agent(&cfg.user_agent);
        #[cfg(windows)]
        let b = match native_tls::TlsConnector::new() {
            Ok(c) => b.tls_connector(std::sync::Arc::new(c)),
            Err(_) => b,
        };
        HttpClient { agent: b.build(), cfg, requests: AtomicUsize::new(0) }
    }

    pub fn request_count(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }

    /// GET with validation at every hop. `headers` are extra request headers.
    pub fn get(&self, url: &str, headers: &[(&str, &str)]) -> Result<Response> {
        let mut current = validate_url(url, &self.cfg.policy)?.to_string();
        for _hop in 0..=self.cfg.max_redirects {
            let r = self.get_once(&current, headers)?;
            if (300..400).contains(&r.0) {
                let loc = r.1.ok_or_else(|| MkrError::Network(format!("redirect {} without Location", r.0)))?;
                let base = url::Url::parse(&current).map_err(|e| MkrError::Invalid(e.to_string()))?;
                let next = base.join(&loc).map_err(|e| MkrError::Invalid(format!("bad redirect target: {e}")))?;
                current = validate_url(next.as_str(), &self.cfg.policy).map_err(|e| MkrError::Invalid(format!("redirect blocked: {e}")))?.to_string();
                continue;
            }
            let resp = r.2.unwrap();
            return Ok(Response { url: current, ..resp });
        }
        Err(MkrError::Network(format!("too many redirects (>{})", self.cfg.max_redirects)))
    }

    /// One hop with retries. Returns (status, location, response-if-final).
    fn get_once(&self, url: &str, headers: &[(&str, &str)]) -> Result<(u16, Option<String>, Option<Response>)> {
        let mut last_err = String::new();
        for attempt in 0..=self.cfg.retries {
            if attempt > 0 {
                let wait = (self.cfg.backoff_ms << (attempt - 1).min(5)).min(5000);
                std::thread::sleep(Duration::from_millis(wait));
            }
            self.requests.fetch_add(1, Ordering::SeqCst);
            let mut req = self.agent.get(url);
            for (k, v) in headers {
                req = req.set(k, v);
            }
            let resp = match req.call() {
                Ok(r) => r,
                Err(ureq::Error::Status(code, r)) => {
                    if matches!(code, 429 | 500 | 502 | 503 | 504) {
                        last_err = format!("HTTP {code}");
                        if code == 429 {
                            if let Some(s) = r.header("Retry-After").and_then(|v| v.trim().parse::<u64>().ok()) {
                                std::thread::sleep(Duration::from_millis((s * 1000).min(3000)));
                            }
                        }
                        continue;
                    }
                    return Err(MkrError::Network(format!("HTTP {code} from {url}")));
                }
                Err(ureq::Error::Transport(t)) => {
                    last_err = t.to_string();
                    continue;
                }
            };
            let status = resp.status();
            if (300..400).contains(&status) {
                return Ok((status, resp.header("Location").map(|s| s.to_string()), None));
            }
            let content_type = resp.header("Content-Type").unwrap_or("").to_string();
            if !type_allowed(&content_type) {
                return Err(MkrError::Network(format!("content type '{content_type}' not allowed")));
            }
            let mut body = Vec::new();
            let mut rd = resp.into_reader().take(self.cfg.max_bytes as u64 + 1);
            if let Err(e) = rd.read_to_end(&mut body) {
                last_err = format!("read error: {e}");
                continue;
            }
            let truncated = body.len() > self.cfg.max_bytes;
            body.truncate(self.cfg.max_bytes);
            return Ok((status, None, Some(Response { url: url.to_string(), status, content_type, body, truncated })));
        }
        Err(MkrError::Network(format!("{url}: failed after {} attempt(s): {last_err}", self.cfg.retries + 1)))
    }

    pub fn get_text(&self, url: &str, headers: &[(&str, &str)]) -> Result<(String, String)> {
        let r = self.get(url, headers)?;
        let text = decode_body(&r.body, &r.content_type);
        Ok((r.url, text))
    }
}

const CP1256_HIGH: &str = "€پ‚ƒ„…†‡ˆ‰ٹ‹Œچژڈگ‘’“”•–—ک™ڑ›œ\u{200c}\u{200d}ں\u{a0}،¢£¤¥¦§¨©ھ«¬\u{ad}®¯°±²³´µ¶·¸¹؛»¼½¾؟ہءآأؤإئابةتثجحخدذرزسشصض×طظعغـفقكàلâمنهوçèéêëىيîï\u{64b}\u{64c}\u{64d}\u{64e}ô\u{64f}\u{650}÷\u{651}ù\u{652}ûü\u{200e}\u{200f}ے";

fn charset_of(ct: &str, body: &[u8]) -> String {
    let lower = ct.to_lowercase();
    if let Some(i) = lower.find("charset=") {
        return lower[i + 8..].trim_matches(|c: char| c == '"' || c == '\'' || c == ' ').split(';').next().unwrap_or("").to_string();
    }
    let head = String::from_utf8_lossy(&body[..body.len().min(2048)]).to_lowercase();
    if let Some(i) = head.find("charset=") {
        return head[i + 8..].trim_start_matches(|c: char| c == '"' || c == '\'' || c == ' ').chars().take_while(|c| c.is_alphanumeric() || *c == '-' || *c == '_').collect();
    }
    "utf-8".into()
}

pub fn decode_body(body: &[u8], content_type: &str) -> String {
    match charset_of(content_type, body).as_str() {
        "windows-1256" | "cp1256" | "iso-8859-6-i" | "x-cp1256" => {
            let table: Vec<char> = CP1256_HIGH.chars().collect();
            debug_assert_eq!(table.len(), 128);
            body.iter().map(|&b| if b < 0x80 { b as char } else { table[(b - 0x80) as usize] }).collect()
        }
        "iso-8859-1" | "latin1" | "windows-1252" | "latin-1" => body.iter().map(|&b| b as char).collect(),
        _ => String::from_utf8_lossy(body).into_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decodes_windows_1256_arabic() {
        // "مرحبا" in windows-1256
        let b = [0xE3u8, 0xD1, 0xCD, 0xC8, 0xC7];
        assert_eq!(decode_body(&b, "text/html; charset=windows-1256"), "مرحبا");
        assert_eq!(decode_body(&[0x61, 0xE9], "text/html; charset=ISO-8859-1"), "aé");
        assert_eq!(decode_body("مرحبا".as_bytes(), "text/html"), "مرحبا");
        let meta = b"<html><head><meta charset=\"windows-1256\"></head>\xE3\xD1";
        assert!(decode_body(meta, "text/html").ends_with("مر"));
        assert_eq!(CP1256_HIGH.chars().count(), 128);
    }
}
