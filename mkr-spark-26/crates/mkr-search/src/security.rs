//! URL validation and SSRF protection: only http(s), no credentials, no private/loopback/link-local
//! destinations (checked both for literal IPs and for every address a hostname resolves to).
use mkr_core::{MkrError, Result};
use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use url::{Host, Url};

#[derive(Clone, Debug, Default)]
pub struct UrlPolicy {
    /// allow private/loopback targets (only for a user-configured self-hosted endpoint)
    pub allow_private: bool,
    /// hosts exempt from the private-address check (exact host match)
    pub allow_hosts: HashSet<String>,
}

pub fn is_private_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    ip.is_private() || ip.is_loopback() || ip.is_link_local() || ip.is_unspecified() || ip.is_broadcast() || ip.is_multicast() || ip.is_documentation() || (o[0] == 100 && (64..128).contains(&o[1])) || o[0] == 0 || o[0] >= 240
}

pub fn is_private_v6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return is_private_v4(v4);
    }
    let s = ip.segments();
    ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() || (s[0] & 0xfe00) == 0xfc00 || (s[0] & 0xffc0) == 0xfe80
}

pub fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(a) => is_private_v4(a),
        IpAddr::V6(a) => is_private_v6(a),
    }
}

/// Validate a URL for fetching. Returns the parsed URL.
pub fn validate_url(raw: &str, policy: &UrlPolicy) -> Result<Url> {
    let raw = raw.trim();
    if raw.is_empty() || raw.len() > 2048 {
        return Err(MkrError::Invalid("URL empty or longer than 2048 characters".into()));
    }
    if raw.chars().any(|c| c.is_control() || c == ' ') {
        return Err(MkrError::Invalid("URL contains whitespace or control characters".into()));
    }
    let url = Url::parse(raw).map_err(|e| MkrError::Invalid(format!("malformed URL: {e}")))?;
    if url.scheme() != "http" && url.scheme() != "https" {
        return Err(MkrError::Invalid(format!("scheme '{}' is not allowed (only http/https)", url.scheme())));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(MkrError::Invalid("URLs with embedded credentials are rejected".into()));
    }
    let host = url.host().ok_or_else(|| MkrError::Invalid("URL has no host".into()))?;
    let host_str = url.host_str().unwrap_or("").to_lowercase();
    let exempt = policy.allow_private || policy.allow_hosts.contains(&host_str);
    if exempt {
        return Ok(url);
    }
    match host {
        Host::Ipv4(a) => {
            if is_private_v4(a) {
                return Err(MkrError::Invalid(format!("address {a} is private/reserved")));
            }
        }
        Host::Ipv6(a) => {
            if is_private_v6(a) {
                return Err(MkrError::Invalid(format!("address {a} is private/reserved")));
            }
        }
        Host::Domain(d) => {
            let d = d.to_lowercase();
            if d == "localhost" || d.ends_with(".localhost") || d.ends_with(".local") || d.ends_with(".internal") || d.ends_with(".lan") || d.ends_with(".home.arpa") || !d.contains('.') {
                return Err(MkrError::Invalid(format!("host '{d}' is not a public internet name")));
            }
            let port = url.port_or_known_default().unwrap_or(443);
            match (d.as_str(), port).to_socket_addrs() {
                Ok(addrs) => {
                    for a in addrs {
                        if is_private_ip(a.ip()) {
                            return Err(MkrError::Invalid(format!("host '{d}' resolves to private address {}", a.ip())));
                        }
                    }
                }
                Err(e) => return Err(MkrError::Network(format!("cannot resolve '{d}': {e}"))),
            }
        }
    }
    Ok(url)
}

/// Registrable-domain approximation (last two labels, three for common second-level suffixes).
pub fn registrable_domain(host: &str) -> String {
    let h = host.trim_end_matches('.').to_lowercase();
    let parts: Vec<&str> = h.split('.').collect();
    if parts.len() <= 2 {
        return h;
    }
    const SLD: &[&str] = &["co", "com", "org", "net", "gov", "edu", "ac", "or", "go"];
    let n = parts.len();
    if parts[n - 1].len() == 2 && SLD.contains(&parts[n - 2]) && n >= 3 {
        parts[n - 3..].join(".")
    } else {
        parts[n - 2..].join(".")
    }
}

/// Canonical form for duplicate detection: https, no www, no fragment, no tracking params, sorted query, no trailing slash.
pub fn canonical_url(raw: &str) -> String {
    let Ok(mut u) = Url::parse(raw) else { return raw.to_lowercase() };
    u.set_fragment(None);
    let host = u.host_str().unwrap_or("").trim_start_matches("www.").to_lowercase();
    let mut q: Vec<(String, String)> = u
        .query_pairs()
        .filter(|(k, _)| {
            let k = k.to_lowercase();
            !(k.starts_with("utm_") || matches!(k.as_str(), "fbclid" | "gclid" | "ref" | "ref_src" | "mc_cid" | "mc_eid" | "igshid" | "source" | "spm"))
        })
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();
    q.sort();
    let path = u.path().trim_end_matches('/').to_string();
    let query = if q.is_empty() { String::new() } else { format!("?{}", q.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&")) };
    format!("{host}{path}{query}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_dangerous_urls() {
        let p = UrlPolicy::default();
        for bad in [
            "file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)", "http://localhost/admin", "http://127.0.0.1:8080/", "http://10.0.0.5/", "http://192.168.1.1/", "http://172.16.0.1/", "http://169.254.169.254/latest/meta-data", "http://[::1]/", "http://[fd00::1]/", "http://[::ffff:127.0.0.1]/", "http://0.0.0.0/", "http://100.64.0.1/", "http://user:pass@example.com/", "http://intranet/", "http://printer.local/", "http://example.com/a b", "", "not a url", "http://2130706433/",
        ] {
            assert!(validate_url(bad, &p).is_err(), "should reject {bad}");
        }
        assert!(validate_url(&format!("https://example.com/{}", "a".repeat(2100)), &p).is_err());
    }

    #[test]
    fn allows_public_ips_and_exemptions() {
        let p = UrlPolicy::default();
        assert!(validate_url("https://93.184.216.34/page", &p).is_ok());
        assert!(validate_url("http://[2606:2800:220:1:248:1893:25c8:1946]/", &p).is_ok());
        let mut p2 = UrlPolicy::default();
        p2.allow_hosts.insert("127.0.0.1".into());
        assert!(validate_url("http://127.0.0.1:9000/x", &p2).is_ok());
        assert!(validate_url("http://10.0.0.5/", &p2).is_err());
        assert!(validate_url("http://10.0.0.5/", &UrlPolicy { allow_private: true, ..Default::default() }).is_ok());
    }

    #[test]
    fn domains_and_canonical_urls() {
        assert_eq!(registrable_domain("en.wikipedia.org"), "wikipedia.org");
        assert_eq!(registrable_domain("news.bbc.co.uk"), "bbc.co.uk");
        assert_eq!(registrable_domain("example.com"), "example.com");
        assert_eq!(canonical_url("https://www.Example.com/a/b/?utm_source=x&b=2&a=1#frag"), canonical_url("http://example.com/a/b?a=1&b=2"));
        assert_ne!(canonical_url("https://example.com/a"), canonical_url("https://example.com/b"));
    }
}
