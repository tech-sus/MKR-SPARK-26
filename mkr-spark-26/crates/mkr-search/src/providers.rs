//! Search providers behind one trait. Each parses a real, documented response format.
//! Free & keyless: Wikipedia (MediaWiki API), DuckDuckGo HTML. Optional: SearXNG instance, Brave API (key).
use crate::html::{self, PageContent};
use crate::http::HttpClient;
use mkr_core::{text, time::parse_date, MkrError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Hit {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub provider: String,
    pub rank: usize,
    pub published: Option<i64>,
    pub score: f32,
}

pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn search(&self, http: &HttpClient, query: &str, limit: usize) -> Result<Vec<Hit>>;
    /// Providers with a structured content API can return page text directly (None = use generic fetch).
    fn fetch_content(&self, _http: &HttpClient, _hit: &Hit) -> Option<Result<PageContent>> {
        None
    }
}

fn with_params(base: &str, path: &str, params: &[(&str, &str)]) -> Result<String> {
    Url::parse_with_params(&format!("{}{}", base.trim_end_matches('/'), path), params).map(|u| u.to_string()).map_err(|e| MkrError::Invalid(format!("bad provider URL: {e}")))
}

fn json(body: &str, who: &str) -> Result<Value> {
    serde_json::from_str(body).map_err(|e| MkrError::Network(format!("{who}: response is not valid JSON ({e})")))
}

// ---------------------------------------------------------------- Wikipedia
pub struct Wikipedia {
    /// override for tests / mirrors; default is https://{lang}.wikipedia.org
    pub base: Option<String>,
}

impl Wikipedia {
    fn base_for(&self, lang: &str) -> String {
        self.base.clone().unwrap_or_else(|| format!("https://{lang}.wikipedia.org"))
    }
}

impl Provider for Wikipedia {
    fn name(&self) -> &str {
        "wikipedia"
    }
    fn search(&self, http: &HttpClient, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let lang = if text::detect_lang(query) == text::Lang::Arabic { "ar" } else { "en" };
        let base = self.base_for(lang);
        let kw = text::keywords(query);
        let url = with_params(&base, "/w/api.php", &[("action", "query"), ("list", "search"), ("srsearch", &kw), ("srlimit", &limit.to_string()), ("format", "json"), ("utf8", "1")])?;
        let (_, body) = http.get_text(&url, &[("Accept", "application/json")])?;
        let v = json(&body, "wikipedia")?;
        if let Some(err) = v.get("error") {
            return Err(MkrError::Network(format!("wikipedia API error: {}", err.get("info").and_then(Value::as_str).unwrap_or("unknown"))));
        }
        let items = v.pointer("/query/search").and_then(Value::as_array).ok_or_else(|| MkrError::Network("wikipedia: unexpected response shape".into()))?;
        Ok(items
            .iter()
            .enumerate()
            .filter_map(|(rank, it)| {
                let title = it.get("title")?.as_str()?.to_string();
                let path = format!("/wiki/{}", title.replace(' ', "_"));
                let url = Url::parse(&base).ok()?.join(&path).ok()?.to_string();
                Some(Hit {
                    snippet: html::strip_tags(it.get("snippet").and_then(Value::as_str).unwrap_or("")),
                    published: it.get("timestamp").and_then(Value::as_str).and_then(parse_date),
                    title,
                    url,
                    provider: "wikipedia".into(),
                    rank,
                    score: 0.0,
                })
            })
            .collect())
    }

    fn fetch_content(&self, http: &HttpClient, hit: &Hit) -> Option<Result<PageContent>> {
        let u = Url::parse(&hit.url).ok()?;
        let base = format!("{}://{}", u.scheme(), u.host_str()?) + &u.port().map(|p| format!(":{p}")).unwrap_or_default();
        let title = hit.url.rsplit("/wiki/").next().map(|t| Url::parse(&format!("http://x/{t}")).ok().map(|u| u.path()[1..].replace('_', " ")).unwrap_or_default())?;
        let title = percent_decode(&title);
        let run = || -> Result<PageContent> {
            let url = with_params(&base, "/w/api.php", &[("action", "query"), ("prop", "extracts|info"), ("exintro", "1"), ("explaintext", "1"), ("redirects", "1"), ("titles", &title), ("format", "json"), ("utf8", "1")])?;
            let (_, body) = http.get_text(&url, &[("Accept", "application/json")])?;
            let v = json(&body, "wikipedia")?;
            let pages = v.pointer("/query/pages").and_then(Value::as_object).ok_or_else(|| MkrError::Network("wikipedia: no pages in response".into()))?;
            let page = pages.values().next().ok_or_else(|| MkrError::Network("wikipedia: empty page set".into()))?;
            let extract = page.get("extract").and_then(Value::as_str).unwrap_or("").trim().to_string();
            if extract.is_empty() {
                return Err(MkrError::Network("wikipedia: page has no extract".into()));
            }
            Ok(PageContent { title: page.get("title").and_then(Value::as_str).unwrap_or(&title).to_string(), description: String::new(), published: page.get("touched").and_then(Value::as_str).and_then(parse_date), text: extract })
        };
        Some(run())
    }
}

fn percent_decode(s: &str) -> String {
    Url::parse(&format!("http://x/?q={}", s.replace('+', "%2B"))).ok().and_then(|u| u.query_pairs().next().map(|(_, v)| v.into_owned())).unwrap_or_else(|| s.to_string())
}

// ---------------------------------------------------------------- DuckDuckGo (HTML endpoint)
pub struct DuckDuckGo {
    pub base: Option<String>,
}

/// Parse DuckDuckGo's HTML results page.
pub fn parse_ddg(page: &str) -> Result<Vec<Hit>> {
    // Entries in page order; ads are kept as `None` placeholders so that snippets stay attached
    // to the result block they belong to.
    let mut entries: Vec<(Option<String>, String, String)> = Vec::new();
    for (href, text, class) in html::anchors(page) {
        let is = |c: &str| class.split_whitespace().any(|x| x == c);
        if is("result__a") {
            entries.push((ddg_target(&href), text, String::new()));
        } else if is("result__snippet") {
            if let Some(last) = entries.last_mut() {
                if last.2.is_empty() {
                    last.2 = text;
                }
            }
        }
    }
    let results: Vec<(String, String, String)> = entries.into_iter().filter_map(|(u, t, s)| u.map(|u| (u, t, s))).collect();
    if results.is_empty() {
        let low = page.to_lowercase();
        if low.contains("anomaly") || low.contains("captcha") || low.contains("unusual traffic") || low.contains("challenge-form") {
            return Err(MkrError::Network("duckduckgo returned a bot-check page (rate limited); try again later or use another provider".into()));
        }
        if low.contains("no results") || low.contains("no more results") {
            return Ok(vec![]);
        }
        return Err(MkrError::Network("duckduckgo: no parsable results (page markup may have changed)".into()));
    }
    Ok(results.into_iter().enumerate().map(|(i, (url, title, snippet))| Hit { title, url, snippet, provider: "duckduckgo".into(), rank: i, published: None, score: 0.0 }).collect())
}

fn ddg_target(href: &str) -> Option<String> {
    let full = if href.starts_with("//") { format!("https:{href}") } else { href.to_string() };
    let u = Url::parse(&full).ok()?;
    let host = u.host_str()?;
    if host.ends_with("duckduckgo.com") {
        if u.path() == "/y.js" {
            return None; // advertisement
        }
        let target = u.query_pairs().find(|(k, _)| k == "uddg").map(|(_, v)| v.into_owned())?;
        return Url::parse(&target).ok().map(|t| t.to_string());
    }
    Some(u.to_string())
}

impl Provider for DuckDuckGo {
    fn name(&self) -> &str {
        "duckduckgo"
    }
    fn search(&self, http: &HttpClient, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let base = self.base.clone().unwrap_or_else(|| "https://html.duckduckgo.com".into());
        let url = with_params(&base, "/html/", &[("q", query)])?;
        let (_, body) = http.get_text(&url, &[("Accept", "text/html")])?;
        let mut hits = parse_ddg(&body)?;
        hits.truncate(limit);
        Ok(hits)
    }
}

// ---------------------------------------------------------------- SearXNG (self-hosted, optional)
pub struct SearxNg {
    pub base: String,
}

impl Provider for SearxNg {
    fn name(&self) -> &str {
        "searxng"
    }
    fn search(&self, http: &HttpClient, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let url = with_params(&self.base, "/search", &[("q", query), ("format", "json")])?;
        let (_, body) = http.get_text(&url, &[("Accept", "application/json")])?;
        let v = json(&body, "searxng")?;
        let items = v.get("results").and_then(Value::as_array).ok_or_else(|| MkrError::Network("searxng: no 'results' (is the JSON format enabled on the instance?)".into()))?;
        Ok(items
            .iter()
            .take(limit)
            .enumerate()
            .filter_map(|(rank, it)| {
                Some(Hit {
                    title: it.get("title")?.as_str()?.to_string(),
                    url: it.get("url")?.as_str()?.to_string(),
                    snippet: it.get("content").and_then(Value::as_str).unwrap_or("").to_string(),
                    published: it.get("publishedDate").and_then(Value::as_str).and_then(parse_date),
                    provider: "searxng".into(),
                    rank,
                    score: 0.0,
                })
            })
            .collect())
    }
}

// ---------------------------------------------------------------- Brave Search API (optional, needs key)
pub struct Brave {
    pub key: String,
    pub base: Option<String>,
}

impl Provider for Brave {
    fn name(&self) -> &str {
        "brave"
    }
    fn search(&self, http: &HttpClient, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let base = self.base.clone().unwrap_or_else(|| "https://api.search.brave.com".into());
        let url = with_params(&base, "/res/v1/web/search", &[("q", query), ("count", &limit.to_string())])?;
        let (_, body) = http.get_text(&url, &[("Accept", "application/json"), ("X-Subscription-Token", &self.key)])?;
        let v = json(&body, "brave")?;
        let items = v.pointer("/web/results").and_then(Value::as_array).ok_or_else(|| MkrError::Network("brave: unexpected response shape".into()))?;
        Ok(items
            .iter()
            .enumerate()
            .filter_map(|(rank, it)| {
                Some(Hit {
                    title: html::strip_tags(it.get("title")?.as_str()?),
                    url: it.get("url")?.as_str()?.to_string(),
                    snippet: html::strip_tags(it.get("description").and_then(Value::as_str).unwrap_or("")),
                    published: it.get("page_age").and_then(Value::as_str).and_then(parse_date),
                    provider: "brave".into(),
                    rank,
                    score: 0.0,
                })
            })
            .collect())
    }
}
