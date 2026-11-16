//! End-to-end tests of the real HTTP stack against a local mock server. The mock serves fixtures
//! shaped like the documented Wikipedia / SearXNG / Brave JSON and DuckDuckGo HTML responses.
//! (The sandbox this was built in has no access to the live services, so provider parsers are
//! verified against these fixtures, not against the live sites.)
use mkr_search::providers::{parse_ddg, Brave, DuckDuckGo, SearxNg, Wikipedia};
use mkr_search::{HttpClient, HttpConfig, Provider, SearchEngine, UrlPolicy};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct Req {
    path: String,
    headers: Vec<(String, String)>,
}
struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    delay_ms: u64,
}
fn ok(ct: &str, body: &str) -> Resp {
    Resp { status: 200, headers: vec![("Content-Type".into(), ct.into())], body: body.as_bytes().to_vec(), delay_ms: 0 }
}
fn status(code: u16) -> Resp {
    Resp { status: code, headers: vec![("Content-Type".into(), "text/plain".into())], body: b"x".to_vec(), delay_ms: 0 }
}

struct Mock {
    base: String,
    hits: Arc<AtomicUsize>,
    seen: Arc<Mutex<Vec<Req>>>,
}

fn start(handler: impl Fn(&Req) -> Resp + Send + Sync + 'static) -> Mock {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let hits = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (h2, s2) = (hits.clone(), seen.clone());
    let handler = Arc::new(handler);
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut c) = conn else { continue };
            let (h, s, handler) = (h2.clone(), s2.clone(), handler.clone());
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 1024];
                while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    match c.read(&mut tmp) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buf.extend_from_slice(&tmp[..n]),
                    }
                }
                let text = String::from_utf8_lossy(&buf).to_string();
                let mut lines = text.split("\r\n");
                let path = lines.next().unwrap_or("").split_whitespace().nth(1).unwrap_or("/").to_string();
                let headers = lines.filter_map(|l| l.split_once(": ")).map(|(k, v)| (k.to_lowercase(), v.to_string())).collect();
                let req = Req { path, headers };
                h.fetch_add(1, Ordering::SeqCst);
                let r = handler(&req);
                s.lock().unwrap().push(req);
                if r.delay_ms > 0 {
                    std::thread::sleep(Duration::from_millis(r.delay_ms));
                }
                let mut out = format!("HTTP/1.1 {} X\r\nContent-Length: {}\r\nConnection: close\r\n", r.status, r.body.len());
                for (k, v) in &r.headers {
                    out.push_str(&format!("{k}: {v}\r\n"));
                }
                out.push_str("\r\n");
                let _ = c.write_all(out.as_bytes());
                let _ = c.write_all(&r.body);
            });
        }
    });
    Mock { base: format!("http://127.0.0.1:{port}"), hits, seen }
}

fn client(retries: u32, timeout_ms: u64, max_bytes: usize) -> HttpClient {
    let mut policy = UrlPolicy::default();
    policy.allow_hosts.insert("127.0.0.1".into());
    HttpClient::new(HttpConfig { timeout: Duration::from_millis(timeout_ms), retries, backoff_ms: 10, max_bytes, policy, ..Default::default() })
}

#[test]
fn fetches_and_decodes() {
    let m = start(|_| ok("text/html; charset=utf-8", "<p>مرحبا</p>"));
    let c = client(0, 2000, 1 << 20);
    let (_, t) = c.get_text(&format!("{}/a", m.base), &[]).unwrap();
    assert_eq!(t, "<p>مرحبا</p>");
    assert!(m.seen.lock().unwrap()[0].headers.iter().any(|(k, v)| k == "user-agent" && v.contains("MKR")));
}

#[test]
fn retries_server_errors_but_not_client_errors() {
    let n = Arc::new(AtomicUsize::new(0));
    let n2 = n.clone();
    let m = start(move |r| {
        if r.path == "/flaky" {
            if n2.fetch_add(1, Ordering::SeqCst) < 2 { status(503) } else { ok("text/plain", "recovered") }
        } else {
            status(404)
        }
    });
    let c = client(3, 2000, 1 << 20);
    assert_eq!(c.get_text(&format!("{}/flaky", m.base), &[]).unwrap().1, "recovered");
    assert_eq!(c.request_count(), 3);
    let before = c.request_count();
    assert!(c.get(&format!("{}/missing", m.base), &[]).is_err());
    assert_eq!(c.request_count() - before, 1, "404 must not be retried");
    let m2 = start(|_| status(500));
    let c2 = client(2, 2000, 1 << 20);
    let e = c2.get(&format!("{}/x", m2.base), &[]).err().unwrap().to_string();
    assert!(e.contains("3 attempt"), "{e}");
}

#[test]
fn enforces_timeouts_size_cap_and_content_type() {
    let m = start(|r| match r.path.as_str() {
        "/slow" => Resp { delay_ms: 1500, ..ok("text/plain", "late") },
        "/big" => ok("text/plain", &"a".repeat(100_000)),
        "/img" => ok("image/png", "PNG"),
        _ => status(404),
    });
    let c = client(1, 300, 1 << 20);
    let t0 = std::time::Instant::now();
    assert!(c.get(&format!("{}/slow", m.base), &[]).is_err());
    assert!(t0.elapsed() < Duration::from_millis(1400), "timeout not enforced: {:?}", t0.elapsed());
    assert_eq!(m.hits.load(Ordering::SeqCst), 2, "one retry after timeout");
    let c = client(0, 2000, 1000);
    let r = c.get(&format!("{}/big", m.base), &[]).unwrap();
    assert!(r.truncated && r.body.len() == 1000);
    assert!(c.get(&format!("{}/img", m.base), &[]).unwrap_err().to_string().contains("not allowed"));
}

#[test]
fn redirects_are_validated_at_every_hop() {
    let m = start(|r| match r.path.as_str() {
        "/ok" => Resp { status: 302, headers: vec![("Location".into(), "/final".into())], body: vec![], delay_ms: 0 },
        "/final" => ok("text/plain", "arrived"),
        "/ssrf" => Resp { status: 302, headers: vec![("Location".into(), "http://169.254.169.254/latest/meta-data/".into())], body: vec![], delay_ms: 0 },
        "/file" => Resp { status: 302, headers: vec![("Location".into(), "file:///etc/passwd".into())], body: vec![], delay_ms: 0 },
        "/loop" => Resp { status: 302, headers: vec![("Location".into(), "/loop".into())], body: vec![], delay_ms: 0 },
        _ => status(404),
    });
    let c = client(0, 2000, 1 << 20);
    assert_eq!(c.get_text(&format!("{}/ok", m.base), &[]).unwrap().1, "arrived");
    assert!(c.get(&format!("{}/ssrf", m.base), &[]).unwrap_err().to_string().contains("redirect blocked"));
    assert!(c.get(&format!("{}/file", m.base), &[]).unwrap_err().to_string().contains("redirect blocked"));
    assert!(c.get(&format!("{}/loop", m.base), &[]).unwrap_err().to_string().contains("too many redirects"));
    // the metadata address must never have been contacted: only our mock saw traffic
    assert!(c.get("http://169.254.169.254/", &[]).is_err());
}

const WIKI_SEARCH: &str = r#"{"batchcomplete":"","query":{"search":[
 {"ns":0,"title":"Rust (programming language)","pageid":1,"snippet":"<span class=\"searchmatch\">Rust</span> is a general-purpose programming language &amp; more","timestamp":"2026-10-01T12:00:00Z"},
 {"ns":0,"title":"Rust","pageid":2,"snippet":"Rust is an iron oxide","timestamp":"2026-09-01T12:00:00Z"}]}}"#;
const WIKI_EXTRACT: &str = r#"{"query":{"pages":{"1":{"pageid":1,"title":"Rust (programming language)","touched":"2026-10-02T08:00:00Z","extract":"Rust is a general-purpose programming language that emphasizes performance, type safety, and concurrency. It was first stable-released in 2015."}}}}"#;
const DDG_HTML: &str = r##"<html><body><div class="results">
<div class="result"><h2><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rust-lang.org%2F&amp;rut=abc">Rust Programming Language</a></h2><a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rust-lang.org%2F">A language empowering <b>everyone</b> to build reliable software.</a></div>
<div class="result result--ad"><a class="result__a" href="https://duckduckgo.com/y.js?ad_provider=x">Buy stuff</a><a class="result__snippet" href="#">Ad text</a></div>
<div class="result"><h2><a class="result__a" href="https://doc.rust-lang.org/book/">The Rust Book</a></h2><a class="result__snippet" href="#">Learn Rust with the official book.</a></div>
</div></body></html>"##;

#[test]
fn wikipedia_provider_search_and_extract() {
    let m = start(|r| {
        if r.path.contains("list=search") { ok("application/json", WIKI_SEARCH) } else if r.path.contains("prop=extracts") { ok("application/json", WIKI_EXTRACT) } else { status(404) }
    });
    let c = client(0, 2000, 1 << 20);
    let w = Wikipedia { base: Some(m.base.clone()) };
    let hits = w.search(&c, "what is the Rust programming language?", 5).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].title, "Rust (programming language)");
    assert!(hits[0].url.ends_with("/wiki/Rust_(programming_language)"));
    assert_eq!(hits[0].snippet, "Rust is a general-purpose programming language & more");
    assert!(hits[0].published.is_some());
    // stopwords stripped from the API query
    let seen = m.seen.lock().unwrap();
    assert!(seen[0].path.contains("srsearch=Rust+programming+language") || seen[0].path.contains("srsearch=Rust%20programming%20language"), "{}", seen[0].path);
    drop(seen);
    let page = w.fetch_content(&c, &hits[0]).unwrap().unwrap();
    assert!(page.text.contains("type safety"));
    assert!(page.published.is_some());
}

#[test]
fn ddg_parser_handles_redirects_ads_and_failures() {
    let hits = parse_ddg(DDG_HTML).unwrap();
    assert_eq!(hits.len(), 2, "ad must be dropped: {hits:?}");
    assert_eq!(hits[0].url, "https://www.rust-lang.org/");
    assert_eq!(hits[0].snippet, "A language empowering everyone to build reliable software.");
    assert_eq!(hits[1].url, "https://doc.rust-lang.org/book/");
    assert!(parse_ddg("<html><body>Unfortunately, bots use DuckDuckGo too. anomaly-modal</body></html>").unwrap_err().to_string().contains("bot-check"));
    assert!(parse_ddg("<html><body>something else entirely</body></html>").unwrap_err().to_string().contains("markup"));
    assert!(parse_ddg("<html><body>No results found for your query</body></html>").unwrap().is_empty());
}

#[test]
fn searxng_and_brave_providers() {
    let m = start(|r| {
        if r.path.starts_with("/search") {
            ok("application/json", r#"{"results":[{"title":"A","url":"https://a.org/1","content":"snippet a","publishedDate":"2026-05-05T00:00:00"},{"title":"B","url":"https://b.org/2"}]}"#)
        } else if r.path.starts_with("/res/v1/web/search") {
            ok("application/json", r#"{"web":{"results":[{"title":"<strong>Brave</strong> hit","url":"https://c.org/3","description":"desc &amp; more","page_age":"2026-06-01T00:00:00"}]}}"#)
        } else {
            status(404)
        }
    });
    let c = client(0, 2000, 1 << 20);
    let s = SearxNg { base: m.base.clone() }.search(&c, "test query", 5).unwrap();
    assert_eq!((s.len(), s[0].snippet.as_str()), (2, "snippet a"));
    assert!(s[0].published.is_some() && s[1].published.is_none());
    let b = Brave { key: "k-123".into(), base: Some(m.base.clone()) }.search(&c, "test", 5).unwrap();
    assert_eq!((b[0].title.as_str(), b[0].snippet.as_str()), ("Brave hit", "desc & more"));
    let seen = m.seen.lock().unwrap();
    assert!(seen.last().unwrap().headers.iter().any(|(k, v)| k == "x-subscription-token" && v == "k-123"));
    drop(seen);
    let bad = start(|_| ok("application/json", "<html>not json</html>"));
    assert!(SearxNg { base: bad.base.clone() }.search(&c, "q", 3).unwrap_err().to_string().contains("not valid JSON"));
}

#[test]
fn engine_fuses_providers_caches_and_survives_partial_failure() {
    let m = start(|r| {
        if r.path.contains("list=search") { ok("application/json", WIKI_SEARCH) }
        else if r.path.contains("prop=extracts") { ok("application/json", WIKI_EXTRACT) }
        else if r.path.starts_with("/html/") { ok("text/html", DDG_HTML) }
        else if r.path == "/article" { ok("text/html", "<html><head><title>Article</title><meta property=\"article:published_time\" content=\"2026-08-01\"></head><body><article><p>This article body is long enough to be extracted as real content by the page reader, which requires some minimum amount of text.</p></article></body></html>") }
        else { status(500) }
    });
    let c = client(0, 2000, 1 << 20);
    let providers: Vec<Box<dyn Provider>> = vec![Box::new(Wikipedia { base: Some(m.base.clone()) }), Box::new(DuckDuckGo { base: Some(m.base.clone()) }), Box::new(SearxNg { base: format!("{}/dead", m.base) })];
    let eng = SearchEngine::new(c, providers, None, 24, 8).unwrap();
    let out = eng.search("rust programming language").unwrap();
    assert_eq!(out.errors.len(), 1, "searxng points at a 500 endpoint: {:?}", out.errors);
    assert_eq!(out.errors[0].0, "searxng");
    assert!(out.hits.len() >= 3);
    assert!(out.hits.windows(2).all(|w| w[0].score >= w[1].score));
    let after_first = m.hits.load(Ordering::SeqCst);
    let again = eng.search("Rust  Programming   language!").unwrap();
    assert_eq!(again.cached.len(), 2, "successful providers are served from cache");
    assert_eq!(again.hits.len(), out.hits.len());
    // only the failing provider is retried on the second call
    assert_eq!(m.hits.load(Ordering::SeqCst) - after_first, 1);

    // page fetch: provider API for wikipedia, generic HTML for others
    let wiki_hit = out.hits.iter().find(|h| h.provider == "wikipedia").unwrap();
    let p = eng.fetch(wiki_hit).unwrap();
    assert!(p.text.contains("concurrency") && p.published.is_some());
    assert!(p.citation(1).starts_with("[1] Rust (programming language)"));
    let mut custom = wiki_hit.clone();
    custom.provider = "duckduckgo".into();
    custom.url = format!("{}/article", m.base);
    let p2 = eng.fetch(&custom).unwrap();
    assert_eq!(p2.title, "Article");
    assert_eq!(mkr_core::time::format_date(p2.published.unwrap()), "2026-08-01");
    let n = m.hits.load(Ordering::SeqCst);
    eng.fetch(&custom).unwrap();
    assert_eq!(m.hits.load(Ordering::SeqCst), n, "page cache hit");
    assert!(eng.search("   ").is_err());
}

#[test]
fn engine_reports_when_every_provider_fails() {
    let m = start(|_| status(500));
    let eng = SearchEngine::new(client(0, 1000, 1 << 20), vec![Box::new(Wikipedia { base: Some(m.base.clone()) })], None, 1, 5).unwrap();
    let e = eng.search("anything").err().unwrap().to_string();
    assert!(e.contains("all search providers failed"), "{e}");
}
