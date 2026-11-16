//! Dataset preparation: UTF-8 validation, normalisation, de-duplication, document framing.
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Document separator byte in prepared corpus files (mapped to EOS during batching).
pub const DOC_SEP: u8 = 0;

#[derive(Debug, Default, Clone)]
pub struct PrepStats {
    pub files: usize,
    pub documents: usize,
    pub duplicate_paragraphs_removed: usize,
    pub invalid_utf8_files: usize,
    pub bytes_out: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ScriptStats {
    pub arabic_chars: usize,
    pub latin_chars: usize,
    pub digits: usize,
    pub other_chars: usize,
}

pub fn script_stats(text: &str) -> ScriptStats {
    let mut s = ScriptStats::default();
    for c in text.chars() {
        match c {
            '0'..='9' | '\u{0660}'..='\u{0669}' => s.digits += 1,
            '\u{0600}'..='\u{06FF}' | '\u{0750}'..='\u{077F}' | '\u{FB50}'..='\u{FDFF}' | '\u{FE70}'..='\u{FEFF}' => s.arabic_chars += 1,
            'a'..='z' | 'A'..='Z' => s.latin_chars += 1,
            c if c.is_whitespace() => {}
            _ => s.other_chars += 1,
        }
    }
    s
}

/// Normalise text for training: unify newlines, drop control chars (keeping \n and \t),
/// remove tatweel (U+0640), collapse runs of blank lines, trim trailing spaces.
pub fn sanitize_text(text: &str) -> String {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c == '\u{FEFF}' || c == '\u{0640}' || (c.is_control() && c != '\n' && c != '\t') {
            continue;
        }
        out.push(c);
    }
    let mut res = String::with_capacity(out.len());
    let mut blank = 0;
    for line in out.split('\n') {
        let line = line.trim_end();
        if line.is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        res.push_str(line);
        res.push('\n');
    }
    res.trim().to_string()
}

fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn collect_files(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        if let Ok(rd) = fs::read_dir(path) {
            let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
            entries.sort();
            for e in entries {
                collect_files(&e, out);
            }
        }
    } else if matches!(path.extension().and_then(|e| e.to_str()), Some("txt") | Some("md") | Some("text")) {
        out.push(path.to_path_buf());
    }
}

/// Build a training corpus file from text files/directories. Paragraph-level exact
/// de-duplication; each source file becomes one document terminated by `DOC_SEP`.
pub fn prepare_corpus(inputs: &[PathBuf], output: &Path) -> std::io::Result<PrepStats> {
    let mut files = Vec::new();
    for i in inputs {
        collect_files(i, &mut files);
    }
    let mut stats = PrepStats::default();
    let mut seen: HashSet<u64> = HashSet::new();
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut w = std::io::BufWriter::new(fs::File::create(output)?);
    for f in files {
        let raw = fs::read(&f)?;
        let text = match String::from_utf8(raw) {
            Ok(t) => t,
            Err(e) => {
                stats.invalid_utf8_files += 1;
                String::from_utf8_lossy(e.as_bytes()).replace('\u{FFFD}', "")
            }
        };
        let clean = sanitize_text(&text);
        let mut doc = String::new();
        for para in clean.split("\n\n") {
            let p = para.trim();
            if p.is_empty() {
                continue;
            }
            if !seen.insert(hash_str(p)) {
                stats.duplicate_paragraphs_removed += 1;
                continue;
            }
            doc.push_str(p);
            doc.push_str("\n\n");
        }
        if doc.trim().is_empty() {
            continue;
        }
        w.write_all(doc.trim_end().as_bytes())?;
        w.write_all(&[DOC_SEP])?;
        stats.files += 1;
        stats.documents += 1;
        stats.bytes_out += doc.len() + 1;
    }
    w.flush()?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_basic() {
        let s = sanitize_text("a\r\nb\u{0000}\u{FEFF}  \n\n\n\nمـرحـبا");
        assert_eq!(s, "a\nb\n\nمرحبا");
    }

    #[test]
    fn script_counts() {
        let s = script_stats("abc مرحبا 123");
        assert_eq!((s.latin_chars, s.arabic_chars, s.digits), (3, 5, 3));
    }

    #[test]
    fn prepare_dedupes() {
        let dir = std::env::temp_dir().join(format!("mkr_prep_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "Hello world.\n\nShared paragraph.").unwrap();
        fs::write(dir.join("b.txt"), "Shared paragraph.\n\nمرحبا بالعالم").unwrap();
        let out = dir.join("out.bin");
        let st = prepare_corpus(&[dir.clone()], &out).unwrap();
        assert_eq!(st.documents, 2);
        assert_eq!(st.duplicate_paragraphs_removed, 1);
        let bytes = fs::read(&out).unwrap();
        assert_eq!(bytes.iter().filter(|&&b| b == DOC_SEP).count(), 2);
        let _ = fs::remove_dir_all(&dir);
    }
}
