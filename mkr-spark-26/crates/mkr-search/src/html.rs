//! Dependency-free HTML scanner: readable text, title, description, publication date, anchors.
use mkr_core::time::parse_date;

#[derive(Debug, Default, Clone)]
pub struct PageContent {
    pub title: String,
    pub description: String,
    pub published: Option<i64>,
    pub text: String,
}

/// Remove tags and decode entities (for short HTML snippets such as search-result excerpts).
pub fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    decode_entities(&out).split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        if let Some(end) = rest.find(';').filter(|&e| e <= 10) {
            let ent = &rest[1..end];
            let rep: Option<char> = match ent {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some(' '),
                "ndash" => Some('–'),
                "mdash" => Some('—'),
                "hellip" => Some('…'),
                "laquo" => Some('«'),
                "raquo" => Some('»'),
                "copy" => Some('©'),
                "lsquo" => Some('‘'),
                "rsquo" => Some('’'),
                "ldquo" => Some('“'),
                "rdquo" => Some('”'),
                e if e.starts_with("#x") || e.starts_with("#X") => u32::from_str_radix(&e[2..], 16).ok().and_then(char::from_u32),
                e if e.starts_with('#') => e[1..].parse::<u32>().ok().and_then(char::from_u32),
                _ => None,
            };
            if let Some(c) = rep {
                out.push(c);
                rest = &rest[end + 1..];
                continue;
            }
        }
        out.push('&');
        rest = &rest[1..];
    }
    out.push_str(rest);
    out
}

#[derive(Debug)]
pub struct Tag {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub closing: bool,
    pub end: usize, // index just after '>'
}

impl Tag {
    pub fn attr(&self, k: &str) -> Option<&str> {
        self.attrs.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
    }
}

/// Parse a tag starting at `s[i] == '<'`.
pub fn parse_tag(s: &str, i: usize) -> Option<Tag> {
    let b = s.as_bytes();
    let mut j = i + 1;
    let closing = b.get(j) == Some(&b'/');
    if closing {
        j += 1;
    }
    let start = j;
    while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'-' || b[j] == b':') {
        j += 1;
    }
    if j == start {
        return None;
    }
    let name = s[start..j].to_ascii_lowercase();
    let mut attrs = Vec::new();
    loop {
        while j < b.len() && (b[j].is_ascii_whitespace() || b[j] == b'/') {
            j += 1;
        }
        if j >= b.len() {
            return None;
        }
        if b[j] == b'>' {
            j += 1;
            break;
        }
        let ns = j;
        while j < b.len() && !b[j].is_ascii_whitespace() && b[j] != b'=' && b[j] != b'>' && b[j] != b'/' {
            j += 1;
        }
        let an = s[ns..j].to_ascii_lowercase();
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        let mut val = String::new();
        if j < b.len() && b[j] == b'=' {
            j += 1;
            while j < b.len() && b[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < b.len() && (b[j] == b'"' || b[j] == b'\'') {
                let q = b[j];
                j += 1;
                let vs = j;
                while j < b.len() && b[j] != q {
                    j += 1;
                }
                val = s.get(vs..j)?.to_string();
                j += 1;
            } else {
                let vs = j;
                while j < b.len() && !b[j].is_ascii_whitespace() && b[j] != b'>' {
                    j += 1;
                }
                val = s.get(vs..j)?.to_string();
            }
        }
        if !an.is_empty() {
            attrs.push((an, decode_entities(&val)));
        } else {
            j += 1;
        }
    }
    Some(Tag { name, attrs, closing, end: j })
}

const BLOCK: &[&str] = &["p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "blockquote", "pre", "table", "dd", "dt", "hr", "figcaption"];
const SKIP_CONTENT: &[&str] = &["script", "style", "noscript", "template", "svg", "iframe", "head"];
const SKIP_REGION: &[&str] = &["nav", "header", "footer", "aside", "form", "button", "select", "menu", "dialog"];

fn find_ci(hay: &str, needle: &str, from: usize) -> Option<usize> {
    let h = hay.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || h.len() < n.len() {
        return None;
    }
    (from..=h.len() - n.len()).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

fn extract_text(html: &str, skip_regions: bool) -> String {
    let mut out = String::new();
    let mut i = 0;
    let b = html.as_bytes();
    let mut region_stack: Vec<String> = Vec::new();
    while i < b.len() {
        if b[i] == b'<' {
            if html[i..].starts_with("<!--") {
                i = find_ci(html, "-->", i + 4).map_or(b.len(), |e| e + 3);
                continue;
            }
            if let Some(tag) = parse_tag(html, i) {
                let name = tag.name.as_str();
                if !tag.closing && SKIP_CONTENT.contains(&name) {
                    let close = format!("</{name}");
                    i = find_ci(html, &close, tag.end).and_then(|c| html[c..].find('>').map(|g| c + g + 1)).unwrap_or(b.len());
                    continue;
                }
                if skip_regions && SKIP_REGION.contains(&name) {
                    if tag.closing {
                        if let Some(p) = region_stack.iter().rposition(|n| n == name) {
                            region_stack.truncate(p);
                        }
                    } else {
                        region_stack.push(name.to_string());
                    }
                }
                if BLOCK.contains(&name) && !out.ends_with('\n') {
                    out.push('\n');
                }
                i = tag.end;
                continue;
            }
        }
        let next = html[i..].find('<').map_or(b.len(), |n| if n == 0 { i + 1 } else { i + n });
        if region_stack.is_empty() {
            out.push_str(&decode_entities(&html[i..next]));
        }
        i = next;
    }
    // collapse whitespace but keep paragraph breaks
    let mut res = String::with_capacity(out.len());
    for line in out.split('\n') {
        let l = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if !l.is_empty() {
            res.push_str(&l);
            res.push('\n');
        }
    }
    res.trim().to_string()
}

fn region<'a>(html: &'a str, tag: &str) -> Option<&'a str> {
    let open = find_ci(html, &format!("<{tag}"), 0)?;
    let after = html[open..].find('>')? + open + 1;
    let close = find_ci(html, &format!("</{tag}"), after).unwrap_or(html.len());
    Some(&html[after..close])
}

pub fn parse_page(html: &str) -> PageContent {
    let mut pc = PageContent::default();
    // metadata scan
    let mut i = 0;
    let mut time_date: Option<i64> = None;
    while let Some(p) = html[i..].find('<') {
        let at = i + p;
        if html[at..].starts_with("<!--") {
            i = find_ci(html, "-->", at + 4).map_or(html.len(), |e| e + 3);
            continue;
        }
        let Some(tag) = parse_tag(html, at) else { i = at + 1; continue };
        i = tag.end;
        if tag.closing {
            continue;
        }
        match tag.name.as_str() {
            "title" if pc.title.is_empty() => {
                let end = find_ci(html, "</title", tag.end).unwrap_or(html.len());
                pc.title = decode_entities(html[tag.end..end].split_whitespace().collect::<Vec<_>>().join(" ").as_str());
            }
            "meta" => {
                let key = tag.attr("property").or(tag.attr("name")).or(tag.attr("itemprop")).unwrap_or("").to_lowercase();
                let content = tag.attr("content").unwrap_or("");
                match key.as_str() {
                    "description" | "og:description" if pc.description.is_empty() => pc.description = content.trim().to_string(),
                    "og:title" if pc.title.is_empty() => pc.title = content.trim().to_string(),
                    "article:published_time" | "og:article:published_time" | "datepublished" | "date" | "pubdate" | "publish-date" | "dc.date.issued" | "dc.date" | "article:modified_time" | "og:updated_time" | "lastmod" | "datemodified" => {
                        if let Some(d) = parse_date(content) {
                            // prefer an explicit publication date over a modified date
                            if pc.published.is_none() || key.contains("published") || key == "datepublished" {
                                pc.published = Some(d);
                            }
                        }
                    }
                    _ => {}
                }
            }
            "time" if time_date.is_none() => {
                if let Some(d) = tag.attr("datetime").and_then(parse_date) {
                    time_date = Some(d);
                }
            }
            "script" => {
                if tag.attr("type").map_or(false, |t| t.contains("ld+json")) {
                    let end = find_ci(html, "</script", tag.end).unwrap_or(html.len());
                    let js = &html[tag.end..end];
                    if let Some(k) = js.find("\"datePublished\"") {
                        let rest = &js[k + 15..];
                        if let Some(q) = rest.find('"') {
                            let val: String = rest[q + 1..].chars().take_while(|&c| c != '"').collect();
                            if let Some(d) = parse_date(&val) {
                                pc.published = Some(d);
                            }
                        }
                    }
                    i = end;
                }
            }
            _ => {}
        }
    }
    if pc.published.is_none() {
        pc.published = time_date;
    }
    // main content: <article> or <main> if substantial, else body with nav-like regions skipped
    let body = region(html, "body").unwrap_or(html);
    let mut text = String::new();
    for t in ["article", "main"] {
        if let Some(r) = region(body, t) {
            let tx = extract_text(r, true);
            if tx.chars().count() > 200 {
                text = tx;
                break;
            }
        }
    }
    if text.is_empty() {
        text = extract_text(body, true);
        if text.chars().count() < 200 {
            // malformed HTML may have swallowed content inside an unclosed nav-like tag
            let full = extract_text(body, false);
            if full.chars().count() > text.chars().count() {
                text = full;
            }
        }
    }
    pc.text = text;
    pc
}

/// Anchors (href, inner text, class) from an HTML fragment.
pub fn anchors(html: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(p) = html[i..].find('<') {
        let at = i + p;
        let Some(tag) = parse_tag(html, at) else { i = at + 1; continue };
        i = tag.end;
        if tag.name == "a" && !tag.closing {
            let end = find_ci(html, "</a", tag.end).unwrap_or(html.len());
            let inner = extract_text(&html[tag.end..end], false);
            out.push((tag.attr("href").unwrap_or("").to_string(), inner, tag.attr("class").unwrap_or("").to_string()));
            i = end;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mkr_core::time::days_from_civil;

    const PAGE: &str = r#"<!DOCTYPE html><html><head><title>Rust &amp; Safety &mdash; Guide</title>
<meta name="description" content="A guide to Rust.">
<meta property="article:published_time" content="2026-03-05T09:30:00+00:00">
<script>var x = "<p>not text</p>";</script><style>p{color:red}</style></head>
<body><nav><a href="/">Home</a><a href="/about">About us</a></nav>
<header>Site banner</header>
<article><h1>Why Rust?</h1><p>Rust is a systems programming language focused on safety, speed &amp; concurrency.</p>
<p>It was first released as 1.0 in May 2015 and has grown a lot since then. More text to make the article long enough to be selected over the page body. More text to make the article long enough to be selected over the page body.</p>
<!-- hidden comment <p>hidden</p> --><p>مرحبا بكم في دليل لغة رست الشامل.</p></article>
<footer>Copyright junk footer</footer></body></html>"#;

    #[test]
    fn extracts_article_text_and_metadata() {
        let p = parse_page(PAGE);
        assert_eq!(p.title, "Rust & Safety — Guide");
        assert_eq!(p.description, "A guide to Rust.");
        assert_eq!(p.published, Some(days_from_civil(2026, 3, 5) * 86400));
        assert!(p.text.contains("Rust is a systems programming language focused on safety, speed & concurrency."));
        assert!(p.text.contains("مرحبا بكم"));
        for junk in ["Site banner", "Copyright junk", "Home", "color:red", "not text", "hidden"] {
            assert!(!p.text.contains(junk), "leaked: {junk}");
        }
        assert!(p.text.contains('\n'), "paragraph breaks preserved");
    }

    #[test]
    fn falls_back_when_no_article_and_survives_malformed_html() {
        let p = parse_page("<html><body><div>Plain page content without any article tag, but with enough words to be useful to the reader.</div><p>Unclosed <b>bold <i>italic");
        assert!(p.text.contains("Plain page content"));
        assert!(p.text.contains("italic"));
        // unclosed nav swallowing everything must not erase the content
        let q = parse_page("<body><nav><ul><li>menu<div>Real content that follows an unclosed navigation element and should still be extracted because otherwise the page would be empty for the reader of this test.</div></body>");
        assert!(q.text.contains("Real content"));
        let _ = parse_page("<<<>>><a <b <p x='");
        let _ = parse_page("");
    }

    #[test]
    fn entities_and_dates() {
        assert_eq!(decode_entities("a &lt;b&gt; &#65;&#x42; &unknown; &amp"), "a <b> AB &unknown; &amp");
        let p = parse_page(r#"<html><head><script type="application/ld+json">{"@type":"Article","datePublished":"2025-12-31T10:00:00Z"}</script></head><body><time datetime="2024-01-01">x</time><p>body</p></body></html>"#);
        assert_eq!(p.published, Some(days_from_civil(2025, 12, 31) * 86400));
        let t = parse_page(r#"<body><time datetime="2024-01-01">x</time></body>"#);
        assert_eq!(t.published, Some(days_from_civil(2024, 1, 1) * 86400));
    }

    #[test]
    fn anchors_with_classes() {
        let a = anchors(r#"<div><a class="result__a" href="https://x.org/p?a=1&amp;b=2">First <b>result</b></a><a href="/rel">Rel</a></div>"#);
        assert_eq!(a[0].0, "https://x.org/p?a=1&b=2");
        assert_eq!(a[0].1, "First result");
        assert_eq!(a[0].2, "result__a");
        assert_eq!(a.len(), 2);
    }
}
