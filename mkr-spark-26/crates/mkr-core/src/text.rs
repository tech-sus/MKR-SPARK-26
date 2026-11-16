//! Arabic/English text normalisation and light NLP used for retrieval and verification.

/// Normalise for matching: lowercase, strip Arabic diacritics/tatweel, unify alef/yeh/teh-marbuta
/// variants, map Arabic-Indic digits to ASCII.
pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\u{064B}'..='\u{065F}' | '\u{0670}' | '\u{0640}' | '\u{200C}' | '\u{200D}' | '\u{200E}' | '\u{200F}' => {}
            'أ' | 'إ' | 'آ' | 'ٱ' => out.push('ا'),
            'ى' | 'ئ' => out.push('ي'),
            'ة' => out.push('ه'),
            'ؤ' => out.push('و'),
            '٠'..='٩' => out.push((b'0' + (c as u32 - 0x0660) as u8) as char),
            '۰'..='۹' => out.push((b'0' + (c as u32 - 0x06F0) as u8) as char),
            c => out.extend(c.to_lowercase()),
        }
    }
    out
}

pub fn is_arabic_char(c: char) -> bool {
    matches!(c, '\u{0600}'..='\u{06FF}' | '\u{0750}'..='\u{077F}' | '\u{FB50}'..='\u{FDFF}' | '\u{FE70}'..='\u{FEFF}')
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    Arabic,
    English,
    Mixed,
    Unknown,
}

pub fn detect_lang(s: &str) -> Lang {
    let (mut ar, mut la) = (0usize, 0usize);
    for c in s.chars() {
        if is_arabic_char(c) {
            ar += 1;
        } else if c.is_ascii_alphabetic() {
            la += 1;
        }
    }
    match (ar, la) {
        (0, 0) => Lang::Unknown,
        (a, l) if a * 4 >= (a + l) * 3 => Lang::Arabic,
        (a, l) if l * 4 >= (a + l) * 3 => Lang::English,
        _ => Lang::Mixed,
    }
}

const STOP_EN: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in", "on", "at", "to", "for", "from", "by", "with", "and", "or", "but", "not", "what", "which", "who", "whom", "how", "when", "where", "why", "do", "does", "did", "it", "its", "this", "that", "these", "those", "as", "about", "tell", "me", "please", "can", "you", "i", "my", "your", "has", "have", "had", "will", "would", "should", "could", "than", "then", "there", "their", "they",
];
const STOP_AR: &[&str] = &[
    "ما", "ماذا", "من", "في", "على", "الي", "عن", "هو", "هي", "هل", "كيف", "متي", "اين", "لماذا", "كم", "هذا", "هذه", "ذلك", "تلك", "التي", "الذي", "الذين", "و", "او", "ثم", "لا", "لم", "لن", "قد", "كان", "كانت", "يكون", "مع", "بين", "حتي", "اذا", "ان", "انه", "انها", "بعد", "قبل", "كل", "هناك", "لي", "لك", "اخبرني", "اشرح", "عرف", "ارجو", "من فضلك",
];

pub fn is_stopword(w: &str) -> bool {
    STOP_EN.contains(&w) || STOP_AR.contains(&w)
}

/// Words of the normalised text (alphanumeric runs), stopwords removed when `drop_stop`.
pub fn tokens(s: &str, drop_stop: bool) -> Vec<String> {
    normalize(s)
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && (!drop_stop || !is_stopword(w)))
        .map(|w| strip_ar_prefix(w))
        .collect()
}

/// Original (un-normalised) words of a query minus stopwords, joined by spaces; good for search engines.
pub fn keywords(q: &str) -> String {
    let ws: Vec<&str> = q.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty() && !is_stopword(&normalize(w))).collect();
    if ws.is_empty() { q.trim().to_string() } else { ws.join(" ") }
}

/// Light Arabic clitic stripping (definite article / common one-letter prefixes) for matching only.
fn strip_ar_prefix(w: &str) -> String {
    let chars: Vec<char> = w.chars().collect();
    if chars.is_empty() || !is_arabic_char(chars[0]) {
        return w.to_string();
    }
    let mut s: &[char] = &chars;
    for pre in [&['و', 'ا', 'ل'][..], &['ب', 'ا', 'ل'][..], &['ك', 'ا', 'ل'][..], &['ف', 'ا', 'ل'][..], &['ل', 'ل'][..], &['ا', 'ل'][..]] {
        if s.len() > pre.len() + 2 && s.starts_with(pre) {
            s = &s[pre.len()..];
            break;
        }
    }
    s.iter().collect()
}

/// Split into sentences on . ! ? ؟ ۔ and newlines; keeps sentences between 20 and 600 chars.
pub fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        match c {
            '\n' => {
                push_sentence(&mut out, &mut cur);
            }
            '.' | '!' | '?' | '؟' | '۔' | '؛' => {
                cur.push(c);
                // do not split decimals like 3.14 or abbreviations followed directly by a digit/lowercase
                let next = chars.get(i + 1).copied().unwrap_or(' ');
                let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
                if c == '.' && prev_digit && next.is_ascii_digit() {
                    continue;
                }
                if next.is_whitespace() || next == '\0' {
                    push_sentence(&mut out, &mut cur);
                }
            }
            _ => cur.push(c),
        }
    }
    push_sentence(&mut out, &mut cur);
    out
}

fn push_sentence(out: &mut Vec<String>, cur: &mut String) {
    let s = cur.trim().to_string();
    cur.clear();
    let n = s.chars().count();
    if (20..=600).contains(&n) {
        out.push(s);
    }
}

/// Word-shingle Jaccard similarity of two texts (0..1) for near-duplicate detection.
pub fn similarity(a: &str, b: &str) -> f32 {
    use std::collections::HashSet;
    let ta = tokens(a, false);
    let tb = tokens(b, false);
    let shingle = |t: &Vec<String>| -> HashSet<String> {
        if t.len() < 3 {
            t.iter().cloned().collect()
        } else {
            t.windows(3).map(|w| w.join(" ")).collect()
        }
    };
    let (sa, sb) = (shingle(&ta), shingle(&tb));
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count() as f32;
    2.0 * inter / (sa.len() + sb.len()) as f32 // Dice coefficient
}

pub fn stable_hash(s: &str) -> u64 {
    // FNV-1a 64: stable across platforms/runs (unlike DefaultHasher guarantees)
    let mut h = 0xcbf29ce484222325u64;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arabic_normalisation_unifies_variants() {
        assert_eq!(normalize("أَحْمَد"), normalize("احمد"));
        assert_eq!(normalize("مدرسة"), normalize("مدرسه"));
        assert_eq!(normalize("٢٠٢٦"), "2026");
        assert_eq!(normalize("Rust"), "rust");
        assert_eq!(normalize("كـتـاب"), "كتاب");
    }
    #[test]
    fn language_detection() {
        assert_eq!(detect_lang("What is Rust?"), Lang::English);
        assert_eq!(detect_lang("ما هي لغة رست"), Lang::Arabic);
        assert_eq!(detect_lang("Rust and Cargo لغة برمجة"), Lang::Mixed);
        assert_eq!(detect_lang("1234"), Lang::Unknown);
    }
    #[test]
    fn tokens_drop_stopwords_and_prefixes() {
        assert_eq!(tokens("What is the capital of France?", true), vec!["capital", "france"]);
        let t = tokens("ما هي عاصمة فرنسا", true);
        assert_eq!(t, vec!["عاصمه", "فرنسا"]);
        assert_eq!(tokens("البرمجة", true), vec!["برمجه"]);
    }
    #[test]
    fn sentence_split_keeps_decimals() {
        let s = sentences("Pi is approximately 3.14159 in value. Rust was first released in 2015! هل تعمل هذه الجملة بشكل صحيح؟ short.");
        assert_eq!(s.len(), 3, "{s:?}");
        assert!(s[0].contains("3.14159"));
    }
    #[test]
    fn similarity_detects_copies() {
        let a = "Rust is a systems programming language focused on safety and speed";
        let b = "Rust is a systems programming language focused on safety and speed.";
        let c = "Bananas are yellow fruits that grow in tropical climates around the world";
        assert!(similarity(a, b) > 0.9);
        assert!(similarity(a, c) < 0.1);
    }
}
