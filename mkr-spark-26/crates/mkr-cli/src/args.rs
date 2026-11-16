use std::collections::{HashMap, HashSet};

const BOOL_FLAGS: &[&str] = &["raw", "offline", "fetch", "yes", "resume", "save", "quantize", "no-learn", "approve-single", "help", "hits", "all", "verbose", "no-verify"];

#[derive(Debug, Default)]
pub struct Args {
    pub pos: Vec<String>,
    pub flags: HashMap<String, String>,
    pub bools: HashSet<String>,
}

impl Args {
    pub fn parse(argv: &[String]) -> Result<Args, String> {
        let mut a = Args::default();
        let mut i = 0;
        while i < argv.len() {
            let s = &argv[i];
            if let Some(rest) = s.strip_prefix("--") {
                if let Some((k, v)) = rest.split_once('=') {
                    a.flags.insert(k.to_string(), v.to_string());
                } else if BOOL_FLAGS.contains(&rest) {
                    a.bools.insert(rest.to_string());
                } else {
                    i += 1;
                    let v = argv.get(i).ok_or_else(|| format!("option --{rest} needs a value"))?;
                    a.flags.insert(rest.to_string(), v.clone());
                }
            } else {
                a.pos.push(s.clone());
            }
            i += 1;
        }
        Ok(a)
    }
    pub fn flag(&self, k: &str) -> Option<&str> {
        self.flags.get(k).map(|s| s.as_str())
    }
    pub fn has(&self, k: &str) -> bool {
        self.bools.contains(k)
    }
    pub fn num<T: std::str::FromStr>(&self, k: &str) -> Result<Option<T>, String> {
        match self.flags.get(k) {
            None => Ok(None),
            Some(v) => v.parse::<T>().map(Some).map_err(|_| format!("--{k}: '{v}' is not a valid number")),
        }
    }
    pub fn text(&self, from: usize) -> String {
        self.pos.iter().skip(from).cloned().collect::<Vec<_>>().join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn p(s: &str) -> Args {
        Args::parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>()).unwrap()
    }
    #[test]
    fn parses_flags_and_positionals() {
        let a = p("train data --steps 100 --lr=0.001 --resume extra");
        assert_eq!(a.pos, ["train", "data", "extra"]);
        assert_eq!(a.num::<u64>("steps").unwrap(), Some(100));
        assert_eq!(a.flag("lr"), Some("0.001"));
        assert!(a.has("resume"));
        assert!(Args::parse(&["--steps".to_string()]).is_err());
        assert!(p("--steps abc").num::<u64>("steps").is_err());
    }
}
