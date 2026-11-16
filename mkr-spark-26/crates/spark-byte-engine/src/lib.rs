//! Byte-level text engine. The "vocabulary" is the 256 byte values plus a few control tokens,
//! so any UTF-8 text (Arabic, English, mixed, emoji) is representable with no tokenizer training
//! and no out-of-vocabulary symbols. Trade-off: sequences are longer than with subword tokens
//! (Arabic letters take 2 bytes each in UTF-8), so a given context covers fewer characters.
pub mod corpus;
pub mod stream;

pub use corpus::{prepare_corpus, script_stats, sanitize_text, PrepStats, ScriptStats};
pub use stream::{Batch, Batcher, Corpus};

pub const PAD: u32 = 256;
pub const BOS: u32 = 257;
pub const EOS: u32 = 258;
pub const USER: u32 = 259;
pub const ASSISTANT: u32 = 260;
pub const SYSTEM: u32 = 261;
/// 256 bytes + 6 control tokens, padded to 264 (multiple of 8) for aligned matrices.
pub const VOCAB_SIZE: usize = 264;

pub fn encode(text: &str) -> Vec<u32> {
    text.bytes().map(|b| b as u32).collect()
}

/// Encode with a leading BOS token.
pub fn encode_with_bos(text: &str) -> Vec<u32> {
    let mut v = Vec::with_capacity(text.len() + 1);
    v.push(BOS);
    v.extend(text.bytes().map(|b| b as u32));
    v
}

/// Decode tokens to a string; control tokens are skipped; invalid UTF-8 becomes U+FFFD.
pub fn decode(tokens: &[u32]) -> String {
    let bytes: Vec<u8> = tokens.iter().filter(|&&t| t < 256).map(|&t| t as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Incremental UTF-8 decoder for streaming generation: multi-byte characters (e.g. Arabic)
/// that are split across tokens are held back until complete, so no replacement characters
/// appear mid-stream.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, token: u32) -> String {
        if token >= 256 {
            return String::new();
        }
        self.pending.push(token as u8);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                let mut out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                match e.error_len() {
                    None => {
                        // incomplete sequence at the end: keep waiting
                        self.pending.drain(..valid);
                    }
                    Some(bad) => {
                        out.push('\u{FFFD}');
                        self.pending.drain(..valid + bad);
                    }
                }
                out
            }
        }
    }
    /// Flush anything left (lossy).
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// Format a chat turn sequence for the model: BOS, then USER/ASSISTANT-tagged bytes.
pub fn chat_prompt(history: &[(bool, String)], user_msg: &str) -> Vec<u32> {
    let mut t = vec![BOS];
    for (is_user, m) in history {
        t.push(if *is_user { USER } else { ASSISTANT });
        t.extend(encode(m));
        t.push(EOS);
    }
    t.push(USER);
    t.extend(encode(user_msg));
    t.push(EOS);
    t.push(ASSISTANT);
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_arabic_english_emoji() {
        for s in ["Hello, world!", "مرحبا بالعالم", "Rust لغة برمجة سريعة 🚀", ""] {
            assert_eq!(decode(&encode(s)), s);
        }
        assert_eq!(encode("ع").len(), 2); // Arabic letters are 2 bytes
    }

    #[test]
    fn stream_holds_back_partial_characters() {
        let mut st = Utf8Stream::new();
        let mut out = String::new();
        for t in encode("أهلاً 🚀 ok") {
            let piece = st.push(t);
            assert!(!piece.contains('\u{FFFD}'));
            out.push_str(&piece);
        }
        out.push_str(&st.finish());
        assert_eq!(out, "أهلاً 🚀 ok");
    }

    #[test]
    fn invalid_bytes_do_not_panic() {
        let mut st = Utf8Stream::new();
        let mut out = String::new();
        for t in [0xFFu32, b'a' as u32, 0xC3, b'b' as u32] {
            out.push_str(&st.push(t));
        }
        out.push_str(&st.finish());
        assert!(out.contains('a') && out.contains('b'));
        assert_eq!(decode(&[0xFF, 65]), "\u{FFFD}A");
    }

    #[test]
    fn control_tokens_skipped_and_chat_format() {
        assert_eq!(decode(&[BOS, 72, 105, EOS]), "Hi");
        let p = chat_prompt(&[(true, "a".into()), (false, "b".into())], "c");
        assert_eq!(p[0], BOS);
        assert_eq!(*p.last().unwrap(), ASSISTANT);
        assert!(p.iter().all(|&t| (t as usize) < VOCAB_SIZE));
    }
}
