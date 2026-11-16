//! Memory learning is opt-in: the system may *propose* things to remember, the user approves.
use mkr_core::text::normalize;

/// "remember that ...", "/remember ...", "تذكر أن ..." => the text to store.
pub fn parse_remember_command(msg: &str) -> Option<String> {
    let t = msg.trim();
    let lower = t.to_lowercase();
    for p in ["/remember ", "remember that ", "remember: ", "please remember that ", "please remember ", "remember "] {
        if lower.starts_with(p) {
            let rest = t[p.len()..].trim();
            return (!rest.is_empty()).then(|| rest.to_string());
        }
    }
    for p in ["تذكر أن ", "تذكر ان ", "تذكر: ", "تذكر ", "احفظ أن ", "احفظ ان ", "احفظ: ", "احفظ "] {
        if t.starts_with(p) {
            let rest = t[p.len()..].trim();
            return (!rest.is_empty()).then(|| rest.to_string());
        }
    }
    None
}

/// Statements in a user message that *look* worth remembering (name, role, preference). Returned as
/// proposals only; nothing is stored until the user says yes.
pub fn memory_proposals(msg: &str) -> Vec<String> {
    let mut out = Vec::new();
    for s in mkr_core::text::sentences(&format!("{msg}.")) {
        let n = normalize(&s);
        let starts = ["my name is ", "i am a ", "i am an ", "i work ", "i live in ", "i prefer ", "my favorite ", "my favourite ", "i like ", "i use ", "call me ", "اسمي ", "انا اعمل ", "اعمل ", "اسكن ", "اعيش في ", "افضل ", "احب ", "استخدم "];
        if starts.iter().any(|p| n.starts_with(p) || n.contains(&format!(" {p}"))) && !n.contains('?') && !n.contains('؟') {
            out.push(s.trim().to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn remember_commands() {
        assert_eq!(parse_remember_command("Remember that my server runs Ubuntu").as_deref(), Some("my server runs Ubuntu"));
        assert_eq!(parse_remember_command("/remember buy milk").as_deref(), Some("buy milk"));
        assert_eq!(parse_remember_command("تذكر أن اجتماعي يوم الأحد").as_deref(), Some("اجتماعي يوم الأحد"));
        assert_eq!(parse_remember_command("remember"), None);
        assert_eq!(parse_remember_command("what is rust"), None);
    }
    #[test]
    fn proposals_only_for_personal_statements() {
        let p = memory_proposals("Hi there. My name is Zeyad and I prefer dark mode. What is Rust?");
        assert_eq!(p.len(), 1, "{p:?}");
        assert!(p[0].contains("Zeyad"));
        let a = memory_proposals("اسمي زياد وأعمل مطور ويب");
        assert_eq!(a.len(), 1, "{a:?}");
        assert!(memory_proposals("What is the capital of France? Tell me about Rust.").is_empty());
    }
}
