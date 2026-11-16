//! Search subsystem. Free, openly accessible sources first (Wikipedia API, DuckDuckGo HTML);
//! self-hosted SearXNG and Brave Search are optional. No provider is assumed to be unlimited:
//! failures of one provider never block the others, and results are cached.
pub mod engine;
pub mod html;
pub mod http;
pub mod providers;
pub mod security;

pub use engine::{Page, SearchEngine, SearchOutcome};
pub use http::{HttpClient, HttpConfig};
pub use providers::{Hit, Provider};
pub use security::{validate_url, UrlPolicy};
