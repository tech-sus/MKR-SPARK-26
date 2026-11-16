//! Heuristic source-trust prior by domain (0..1). A prior only: it orders sources, it never
//! turns an unverified claim into a verified one.
pub fn domain_trust(domain: &str) -> f32 {
    let d = domain.to_lowercase();
    let d = d.trim_start_matches("www.");
    let tld = d.rsplit('.').next().unwrap_or("");
    const HIGH: &[&str] = &["wikipedia.org", "wikidata.org", "rust-lang.org", "python.org", "mozilla.org", "w3.org", "ietf.org", "who.int", "un.org", "nasa.gov", "nih.gov", "nature.com", "science.org", "arxiv.org", "britannica.com", "iso.org", "openstreetmap.org", "kernel.org", "gnu.org", "apache.org", "github.com", "docs.rs", "crates.io", "sqlite.org"];
    const NEWS: &[&str] = &["reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "aljazeera.com", "aljazeera.net", "npr.org", "theguardian.com"];
    const LOW: &[&str] = &["pinterest.com", "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com", "quora.com", "answers.com", "ehow.com", "blogspot.com", "wordpress.com", "tumblr.com"];
    let matches = |list: &[&str]| list.iter().any(|h| d == *h || d.ends_with(&format!(".{h}")));
    if matches(LOW) {
        0.25
    } else if matches(HIGH) || matches(&["gov", "edu"]) || tld == "gov" || tld == "edu" || d.ends_with(".gov.uk") || d.ends_with(".ac.uk") || d.ends_with(".gov.eg") || d.ends_with(".gov.sa") || d.ends_with(".edu.eg") || d.ends_with(".edu.sa") {
        0.85
    } else if matches(NEWS) {
        0.75
    } else {
        0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiers() {
        assert!(domain_trust("en.wikipedia.org") > 0.8);
        assert!(domain_trust("www.nasa.gov") > 0.8);
        assert!(domain_trust("cs.stanford.edu") > 0.8);
        assert!(domain_trust("random-blog.com") == 0.5);
        assert!(domain_trust("pinterest.com") < 0.3);
        assert!(domain_trust("notwikipedia.org") == 0.5);
    }
}
