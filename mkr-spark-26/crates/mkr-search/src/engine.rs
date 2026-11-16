use crate::html::{self, PageContent};
use crate::http::{HttpClient, HttpConfig};
use crate::providers::{Brave, DuckDuckGo, Hit, Provider, SearxNg, Wikipedia};
use crate::security::{canonical_url, registrable_domain, UrlPolicy};
use mkr_core::{text, time::now_unix, trust::domain_trust, MkrError, Result, SourceRef};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Page {
    pub url: String,
    pub title: String,
    pub domain: String,
    pub text: String,
    pub published: Option<i64>,
    pub fetched_at: i64,
    pub provider: String,
}

impl Page {
    pub fn source_ref(&self) -> SourceRef {
        SourceRef { url: self.url.clone(), title: self.title.clone(), domain: self.domain.clone(), published: self.published, fetched_at: self.fetched_at }
    }
    /// `[n] Title — domain (published YYYY-MM-DD; accessed YYYY-MM-DD) URL`
    pub fn citation(&self, n: usize) -> String {
        let published = self.published.map(mkr_core::time::format_date).unwrap_or_else(|| "date unknown".into());
        format!("[{n}] {} — {} (published {published}; accessed {}) {}", self.title, self.domain, mkr_core::time::format_date(self.fetched_at), self.url)
    }
}

pub struct SearchOutcome {
    pub hits: Vec<Hit>,
    /// (provider, error message) for providers that failed
    pub errors: Vec<(String, String)>,
    pub cached: Vec<String>,
}

pub struct SearchEngine {
    pub http: HttpClient,
    providers: Vec<Box<dyn Provider>>,
    db: Mutex<Connection>,
    ttl_secs: i64,
    max_results: usize,
}

const CACHE_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS search_cache(key TEXT PRIMARY KEY, ts INTEGER NOT NULL, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS page_cache(url TEXT PRIMARY KEY, ts INTEGER NOT NULL, json TEXT NOT NULL);";

impl SearchEngine {
    pub fn new(http: HttpClient, providers: Vec<Box<dyn Provider>>, cache_path: Option<&Path>, ttl_hours: u64, max_results: usize) -> Result<SearchEngine> {
        let conn = match cache_path {
            Some(p) => mkr_core::db::open_db(p)?,
            None => mkr_core::db::open_memory_db()?,
        };
        conn.execute_batch(CACHE_SCHEMA)?;
        Ok(SearchEngine { http, providers, db: Mutex::new(conn), ttl_secs: ttl_hours as i64 * 3600, max_results })
    }

    pub fn from_config(cfg: &mkr_core::Config) -> Result<SearchEngine> {
        let mut policy = UrlPolicy::default();
        let mut providers: Vec<Box<dyn Provider>> = Vec::new();
        for name in &cfg.providers {
            match name.as_str() {
                "wikipedia" => providers.push(Box::new(Wikipedia { base: None })),
                "duckduckgo" | "ddg" => providers.push(Box::new(DuckDuckGo { base: None })),
                "searxng" => {
                    if cfg.searxng_url.is_empty() {
                        continue; // optional: silently unavailable until configured
                    }
                    // the user's own instance may live on a private network; exempt exactly that host
                    if let Ok(u) = url::Url::parse(&cfg.searxng_url) {
                        if let Some(h) = u.host_str() {
                            policy.allow_hosts.insert(h.to_lowercase());
                        }
                    }
                    providers.push(Box::new(SearxNg { base: cfg.searxng_url.clone() }));
                }
                "brave" => {
                    if !cfg.brave_api_key.is_empty() {
                        providers.push(Box::new(Brave { key: cfg.brave_api_key.clone(), base: None }));
                    }
                }
                other => return Err(MkrError::Config(format!("unknown search provider '{other}' (valid: wikipedia, duckduckgo, searxng, brave)"))),
            }
        }
        if providers.is_empty() {
            return Err(MkrError::Config("no usable search provider configured".into()));
        }
        let http = HttpClient::new(HttpConfig { timeout: Duration::from_secs(cfg.timeout_secs), retries: cfg.retries, user_agent: cfg.user_agent.clone(), policy, ..Default::default() });
        std::fs::create_dir_all(&cfg.data_dir)?;
        SearchEngine::new(http, providers, Some(&cfg.paths().cache_db()), cfg.cache_ttl_hours, cfg.max_results)
    }

    pub fn provider_names(&self) -> Vec<String> {
        self.providers.iter().map(|p| p.name().to_string()).collect()
    }

    fn cache_get<T: for<'a> Deserialize<'a>>(&self, table: &str, key: &str) -> Option<T> {
        let db = self.db.lock().ok()?;
        let row: Option<(i64, String)> = db.query_row(&format!("SELECT ts,json FROM {table} WHERE {}=?", if table == "page_cache" { "url" } else { "key" }), [key], |r| Ok((r.get(0)?, r.get(1)?))).optional().ok()?;
        let (ts, js) = row?;
        if now_unix() - ts > self.ttl_secs {
            return None;
        }
        serde_json::from_str(&js).ok()
    }
    fn cache_put<T: Serialize>(&self, table: &str, key: &str, v: &T) {
        if let (Ok(db), Ok(js)) = (self.db.lock(), serde_json::to_string(v)) {
            let col = if table == "page_cache" { "url" } else { "key" };
            let _ = db.execute(&format!("INSERT INTO {table}({col},ts,json) VALUES(?,?,?) ON CONFLICT({col}) DO UPDATE SET ts=excluded.ts, json=excluded.json"), params![key, now_unix(), js]);
        }
    }
    pub fn clear_cache(&self) -> usize {
        self.db.lock().map(|d| d.execute("DELETE FROM search_cache", []).unwrap_or(0) + d.execute("DELETE FROM page_cache", []).unwrap_or(0)).unwrap_or(0)
    }

    /// Query all providers (one failing never blocks the rest), de-duplicate and rank.
    pub fn search(&self, query: &str) -> Result<SearchOutcome> {
        let q = query.trim();
        if q.is_empty() || q.chars().count() > 500 {
            return Err(MkrError::Invalid("query must be 1..500 characters".into()));
        }
        let qkey = text::tokens(q, false).join(" ");
        let mut lists: Vec<Vec<Hit>> = Vec::new();
        let (mut errors, mut cached) = (Vec::new(), Vec::new());
        for p in &self.providers {
            let key = format!("{}|{}", p.name(), qkey);
            if let Some(h) = self.cache_get::<Vec<Hit>>("search_cache", &key) {
                cached.push(p.name().to_string());
                lists.push(h);
                continue;
            }
            match p.search(&self.http, q, self.max_results) {
                Ok(h) => {
                    self.cache_put("search_cache", &key, &h);
                    lists.push(h);
                }
                Err(e) => errors.push((p.name().to_string(), e.to_string())),
            }
        }
        if lists.is_empty() {
            return Err(MkrError::Network(format!("all search providers failed: {}", errors.iter().map(|(p, e)| format!("{p}: {e}")).collect::<Vec<_>>().join("; "))));
        }
        let mut hits = rank_and_dedupe(lists, q);
        hits.truncate(self.max_results);
        Ok(SearchOutcome { hits, errors, cached })
    }

    /// Fetch and extract a page for a hit (provider API when available, otherwise HTTP + HTML extraction).
    pub fn fetch(&self, hit: &Hit) -> Result<Page> {
        if let Some(p) = self.cache_get::<Page>("page_cache", &hit.url) {
            return Ok(p);
        }
        let provider = self.providers.iter().find(|p| p.name() == hit.provider);
        let content: PageContent = match provider.and_then(|p| p.fetch_content(&self.http, hit)) {
            Some(r) => r?,
            None => {
                let (_, body) = self.http.get_text(&hit.url, &[("Accept", "text/html,application/xhtml+xml,text/plain")])?;
                html::parse_page(&body)
            }
        };
        if content.text.chars().count() < 40 {
            return Err(MkrError::Network(format!("{}: page has no extractable text", hit.url)));
        }
        let host = url::Url::parse(&hit.url).ok().and_then(|u| u.host_str().map(|h| h.to_string())).unwrap_or_default();
        let page = Page {
            url: hit.url.clone(),
            title: if content.title.is_empty() { hit.title.clone() } else { content.title },
            domain: registrable_domain(&host),
            text: content.text.chars().take(20_000).collect(),
            published: content.published.or(hit.published),
            fetched_at: now_unix(),
            provider: hit.provider.clone(),
        };
        self.cache_put("page_cache", &hit.url, &page);
        Ok(page)
    }

    /// Fetch up to `n` pages, skipping failures; returns pages and per-URL errors.
    pub fn fetch_top(&self, hits: &[Hit], n: usize) -> (Vec<Page>, Vec<(String, String)>) {
        let (mut pages, mut errs) = (Vec::new(), Vec::new());
        for h in hits {
            if pages.len() >= n {
                break;
            }
            match self.fetch(h) {
                Ok(p) => pages.push(p),
                Err(e) => errs.push((h.url.clone(), e.to_string())),
            }
        }
        (pages, errs)
    }
}

/// Reciprocal-rank fusion across providers + trust prior + query overlap; near-duplicates merged.
pub fn rank_and_dedupe(lists: Vec<Vec<Hit>>, query: &str) -> Vec<Hit> {
    let qtok: Vec<String> = text::tokens(query, true);
    let mut merged: Vec<(String, Hit, f32)> = Vec::new(); // canonical key, hit, rrf
    let mut by_key: HashMap<String, usize> = HashMap::new();
    for list in lists {
        for h in list {
            let key = canonical_url(&h.url);
            let rrf = 1.0 / (60.0 + h.rank as f32);
            if let Some(&i) = by_key.get(&key) {
                merged[i].2 += rrf;
                if merged[i].1.snippet.len() < h.snippet.len() {
                    merged[i].1.snippet = h.snippet.clone();
                }
                if merged[i].1.published.is_none() {
                    merged[i].1.published = h.published;
                }
                continue;
            }
            // same domain + near-identical title => same article under another URL
            let dom = registrable_domain(url::Url::parse(&h.url).ok().and_then(|u| u.host_str().map(String::from)).unwrap_or_default().as_str());
            let dup = merged.iter().position(|(_, m, _)| {
                let md = registrable_domain(url::Url::parse(&m.url).ok().and_then(|u| u.host_str().map(String::from)).unwrap_or_default().as_str());
                md == dom && text::similarity(&m.title, &h.title) > 0.85
            });
            if let Some(i) = dup {
                merged[i].2 += rrf;
                continue;
            }
            by_key.insert(key.clone(), merged.len());
            merged.push((key, h, rrf));
        }
    }
    let mut out: Vec<Hit> = merged
        .into_iter()
        .map(|(_, mut h, rrf)| {
            let host = url::Url::parse(&h.url).ok().and_then(|u| u.host_str().map(String::from)).unwrap_or_default();
            let trust = domain_trust(&registrable_domain(&host));
            let ht: std::collections::HashSet<String> = text::tokens(&format!("{} {}", h.title, h.snippet), true).into_iter().collect();
            let overlap = if qtok.is_empty() { 0.0 } else { qtok.iter().filter(|t| ht.contains(*t)).count() as f32 / qtok.len() as f32 };
            h.score = rrf + 0.04 * (trust - 0.5) + 0.03 * overlap;
            h
        })
        .collect();
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hit(title: &str, url: &str, prov: &str, rank: usize) -> Hit {
        Hit { title: title.into(), url: url.into(), snippet: String::new(), provider: prov.into(), rank, published: None, score: 0.0 }
    }
    #[test]
    fn dedupes_fuses_and_prefers_trusted_sources() {
        let a = vec![hit("Rust language overview", "https://www.example.com/rust?utm_source=x", "a", 0), hit("Rust (programming language)", "https://en.wikipedia.org/wiki/Rust_(programming_language)", "a", 1), hit("Pins about rust", "https://pinterest.com/rust", "a", 2)];
        let b = vec![hit("Rust language overview", "http://example.com/rust", "b", 0), hit("Rust (programming language)", "https://en.wikipedia.org/wiki/Rust_(programming_language)", "b", 1)];
        let r = rank_and_dedupe(vec![a, b], "rust programming language");
        assert_eq!(r.len(), 3, "{:?}", r.iter().map(|h| &h.url).collect::<Vec<_>>());
        assert!(r[0].url.contains("wikipedia") || r[0].url.contains("example.com"));
        assert!(r[2].url.contains("pinterest"), "low-trust source should rank last");
    }
    #[test]
    fn citation_format() {
        let p = Page { url: "https://x.org/a".into(), title: "T".into(), domain: "x.org".into(), text: "t".into(), published: Some(0), fetched_at: 86400, provider: "p".into() };
        assert_eq!(p.citation(2), "[2] T — x.org (published 1970-01-01; accessed 1970-01-02) https://x.org/a");
    }
}
