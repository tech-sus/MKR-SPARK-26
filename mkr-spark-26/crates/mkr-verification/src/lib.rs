//! Evidence verification. The pipeline is purely extractive: retrieved web text is treated as
//! untrusted *data*, never as instructions, and nothing in it is ever executed or fed to the
//! neural model as a prompt. A claim is "verified" only when at least two *independent origins*
//! (different registrable domains whose wording is not a near-verbatim copy) agree and no
//! contradicting claim was found.
pub mod sanitize;
pub mod verify;

pub use sanitize::{sanitize_web_text, SanitizeReport};
pub use verify::{is_time_sensitive, verify, Claim, Conflict, Document, Status, Verdict, VerifyConfig};
