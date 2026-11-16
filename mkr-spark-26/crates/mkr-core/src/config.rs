//! Plain `key = value` configuration (no external parser). Unknown keys are rejected so typos surface.
use crate::error::{MkrError, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub data_dir: PathBuf,
    pub model_path: PathBuf,
    pub threads: usize,
    pub online: bool,
    pub providers: Vec<String>,
    pub searxng_url: String,
    pub brave_api_key: String,
    pub timeout_secs: u64,
    pub retries: u32,
    pub max_results: usize,
    pub max_pages: usize,
    pub cache_ttl_hours: u64,
    pub knowledge_ttl_days: u64,
    pub user_agent: String,
    /// internal model is trusted only when mean log-prob of its output exceeds this (nats/byte)
    pub model_confidence_threshold: f32,
    /// stored knowledge answers a question directly when its match score exceeds this (0..1)
    pub kb_min_score: f32,
    pub temperature: f32,
    pub max_new_tokens: usize,
    /// answer directly from the neural weights when it is confident (still labelled unverified)
    pub trust_model_answers: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            data_dir: PathBuf::from("data"),
            model_path: PathBuf::from("models/MKR-SPARK-26.model"),
            threads: 0,
            online: true,
            providers: vec!["wikipedia".into(), "duckduckgo".into()],
            searxng_url: String::new(),
            brave_api_key: String::new(),
            timeout_secs: 12,
            retries: 2,
            max_results: 8,
            max_pages: 4,
            cache_ttl_hours: 24,
            knowledge_ttl_days: 180,
            user_agent: "MKR-SPARK-26/0.1 (+local research prototype)".into(),
            model_confidence_threshold: -0.35,
            kb_min_score: 0.55,
            temperature: 0.7,
            max_new_tokens: 200,
            trust_model_answers: false,
        }
    }
}

fn parse_bool(v: &str) -> Result<bool> {
    match v.to_lowercase().as_str() {
        "true" | "yes" | "on" | "1" => Ok(true),
        "false" | "no" | "off" | "0" => Ok(false),
        _ => Err(MkrError::Config(format!("not a boolean: {v}"))),
    }
}
fn parse_num<T: std::str::FromStr>(k: &str, v: &str) -> Result<T> {
    v.parse::<T>().map_err(|_| MkrError::Config(format!("{k}: invalid number '{v}'")))
}

impl Config {
    pub const KEYS: &'static [&'static str] = &[
        "data_dir", "model_path", "threads", "online", "providers", "searxng_url", "brave_api_key", "timeout_secs", "retries", "max_results", "max_pages", "cache_ttl_hours", "knowledge_ttl_days", "user_agent", "model_confidence_threshold", "kb_min_score", "temperature", "max_new_tokens", "trust_model_answers",
    ];

    pub fn set(&mut self, k: &str, v: &str) -> Result<()> {
        let v = v.trim();
        match k {
            "data_dir" => self.data_dir = v.into(),
            "model_path" => self.model_path = v.into(),
            "threads" => self.threads = parse_num(k, v)?,
            "online" => self.online = parse_bool(v)?,
            "providers" => self.providers = v.split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect(),
            "searxng_url" => self.searxng_url = v.into(),
            "brave_api_key" => self.brave_api_key = v.into(),
            "timeout_secs" => {
                let n: u64 = parse_num(k, v)?;
                if !(1..=120).contains(&n) {
                    return Err(MkrError::Config("timeout_secs must be 1..120".into()));
                }
                self.timeout_secs = n
            }
            "retries" => self.retries = parse_num::<u32>(k, v)?.min(6),
            "max_results" => self.max_results = parse_num::<usize>(k, v)?.clamp(1, 30),
            "max_pages" => self.max_pages = parse_num::<usize>(k, v)?.clamp(1, 12),
            "cache_ttl_hours" => self.cache_ttl_hours = parse_num(k, v)?,
            "knowledge_ttl_days" => self.knowledge_ttl_days = parse_num(k, v)?,
            "user_agent" => self.user_agent = v.into(),
            "model_confidence_threshold" => self.model_confidence_threshold = parse_num(k, v)?,
            "kb_min_score" => self.kb_min_score = parse_num::<f32>(k, v)?.clamp(0.0, 1.0),
            "temperature" => self.temperature = parse_num::<f32>(k, v)?.clamp(0.0, 3.0),
            "max_new_tokens" => self.max_new_tokens = parse_num::<usize>(k, v)?.clamp(1, 4096),
            "trust_model_answers" => self.trust_model_answers = parse_bool(v)?,
            _ => return Err(MkrError::Config(format!("unknown key '{k}'. Valid keys: {}", Self::KEYS.join(", ")))),
        }
        Ok(())
    }

    pub fn get(&self, k: &str) -> Option<String> {
        Some(match k {
            "data_dir" => self.data_dir.display().to_string(),
            "model_path" => self.model_path.display().to_string(),
            "threads" => self.threads.to_string(),
            "online" => self.online.to_string(),
            "providers" => self.providers.join(","),
            "searxng_url" => self.searxng_url.clone(),
            // never echo secrets in full
            "brave_api_key" => if self.brave_api_key.is_empty() { String::new() } else { "********".into() },
            "timeout_secs" => self.timeout_secs.to_string(),
            "retries" => self.retries.to_string(),
            "max_results" => self.max_results.to_string(),
            "max_pages" => self.max_pages.to_string(),
            "cache_ttl_hours" => self.cache_ttl_hours.to_string(),
            "knowledge_ttl_days" => self.knowledge_ttl_days.to_string(),
            "user_agent" => self.user_agent.clone(),
            "model_confidence_threshold" => self.model_confidence_threshold.to_string(),
            "kb_min_score" => self.kb_min_score.to_string(),
            "temperature" => self.temperature.to_string(),
            "max_new_tokens" => self.max_new_tokens.to_string(),
            "trust_model_answers" => self.trust_model_answers.to_string(),
            _ => return None,
        })
    }

    pub fn parse(text: &str) -> Result<Config> {
        let mut c = Config::default();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (k, v) = line.split_once('=').ok_or_else(|| MkrError::Config(format!("line {}: expected key = value", n + 1)))?;
            c.set(k.trim(), v.trim()).map_err(|e| MkrError::Config(format!("line {}: {e}", n + 1)))?;
        }
        Ok(c)
    }

    pub fn to_file_text(&self) -> String {
        let mut s = String::from("# MKR - SPARK 26 configuration (key = value). Lines starting with # are comments.\n");
        let mut m = BTreeMap::new();
        for k in Self::KEYS {
            let v = if *k == "brave_api_key" { self.brave_api_key.clone() } else { self.get(k).unwrap_or_default() };
            m.insert(*k, v);
        }
        for k in Self::KEYS {
            s.push_str(&format!("{k} = {}\n", m[k]));
        }
        s
    }

    pub fn load(path: &Path) -> Result<Config> {
        if !path.exists() {
            return Ok(Config::default());
        }
        Config::parse(&std::fs::read_to_string(path)?)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(p) = path.parent() {
            if !p.as_os_str().is_empty() {
                std::fs::create_dir_all(p)?;
            }
        }
        std::fs::write(path, self.to_file_text())?;
        Ok(())
    }

    pub fn paths(&self) -> Paths {
        Paths { data: self.data_dir.clone() }
    }
}

pub struct Paths {
    pub data: PathBuf,
}
impl Paths {
    pub fn memory_db(&self) -> PathBuf {
        self.data.join("memory.db")
    }
    pub fn knowledge_db(&self) -> PathBuf {
        self.data.join("knowledge.db")
    }
    pub fn cache_db(&self) -> PathBuf {
        self.data.join("search_cache.db")
    }
    pub fn learning_dir(&self) -> PathBuf {
        self.data.join("learning")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_and_validation() {
        let mut c = Config::default();
        c.set("providers", "wikipedia, SearXNG").unwrap();
        c.set("timeout_secs", "20").unwrap();
        c.set("brave_api_key", "secret123").unwrap();
        let txt = c.to_file_text();
        let c2 = Config::parse(&txt).unwrap();
        assert_eq!(c, c2);
        assert_eq!(c.get("brave_api_key").unwrap(), "********");
        assert!(c.set("timeout_secs", "0").is_err());
        assert!(c.set("nonsense", "1").is_err());
        assert!(Config::parse("online = maybe").is_err());
        assert!(Config::parse("no equals sign").is_err());
    }
}
