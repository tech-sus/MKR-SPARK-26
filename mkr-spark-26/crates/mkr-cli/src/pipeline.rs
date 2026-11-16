//! The question-answering pipeline:
//!   user memory / stored knowledge -> internal model confidence -> web search -> verification ->
//!   cited answer -> (optionally) local knowledge storage.
//! Every answer is labelled with the layer it came from.
use mkr_core::text::{self, Lang};
use mkr_core::time::{format_date, now_unix};
use mkr_core::{Config, Layer};
use mkr_knowledge::Knowledge;
use mkr_learning::{learn_from_verdict, memory_proposals, parse_remember_command, LearnOutcome, LearnPolicy};
use mkr_memory::Memory;
use mkr_search::{Page, SearchEngine};
use mkr_verification::{is_time_sensitive, verify, Document, Status, Verdict, VerifyConfig};
use spark_byte_engine::chat_prompt;
use spark_inference::{generate, Engine, GenConfig, Sampler};
use std::io::{BufRead, Write};

pub struct Assistant {
    pub cfg: Config,
    pub engine: Option<Engine>,
    pub mem: Memory,
    pub kb: Knowledge,
    pub search: Option<SearchEngine>,
    pub session: String,
    pub history: Vec<(bool, String)>,
    pub learn: bool,
    pub approve_single: bool,
    /// ask the user yes/no questions on stdin (disabled in tests and non-interactive use)
    pub interactive: bool,
    pub verbose: bool,
}

struct Labels {
    verified: &'static str,
    single: &'static str,
    conflict: &'static str,
    none: &'static str,
    sources: &'static str,
    also: &'static str,
    notes: &'static str,
    confidence: &'static str,
}

fn labels(lang: Lang) -> Labels {
    if lang == Lang::Arabic {
        Labels { verified: "تم التحقق من مصادر مستقلة", single: "مصدر واحد فقط (غير مؤكد)", conflict: "المصادر غير متفقة", none: "لم أجد معلومات موثوقة كافية للإجابة عن هذا السؤال.", sources: "المصادر", also: "معلومات ذات صلة", notes: "ملاحظات", confidence: "الثقة" }
    } else {
        Labels { verified: "Verified by independent sources", single: "Only one source supports this (unconfirmed)", conflict: "Sources disagree", none: "I could not find enough reliable information to answer this.", sources: "Sources", also: "Also relevant", notes: "Notes", confidence: "confidence" }
    }
}

/// Render a verdict as a cited answer. Pure function (unit-tested).
pub fn compose(v: &Verdict, pages: &[Page], lang: Lang) -> String {
    let l = labels(lang);
    let Some(best) = &v.best else { return l.none.to_string() };
    let mut used: Vec<usize> = Vec::new();
    let mut cite = |docs: &[usize]| -> String {
        docs.iter()
            .map(|d| {
                let n = used.iter().position(|u| u == d).unwrap_or_else(|| {
                    used.push(*d);
                    used.len() - 1
                }) + 1;
                format!("[{n}]")
            })
            .collect::<Vec<_>>()
            .join("")
    };
    let mut s = String::new();
    match v.status {
        Status::Verified => s.push_str(&format!("{} ({} {:.0}%, {} sources)\n", l.verified, l.confidence, v.confidence * 100.0, best.origins)),
        Status::SingleSource => s.push_str(&format!("{} ({} {:.0}%)\n", l.single, l.confidence, v.confidence * 100.0)),
        Status::Conflicting => s.push_str(&format!("{}:\n", l.conflict)),
        Status::Insufficient => return l.none.to_string(),
    }
    if v.status == Status::Conflicting {
        for c in v.conflicts.iter().take(3) {
            let a = cite(&[c.a_doc]);
            let b = cite(&[c.b_doc]);
            s.push_str(&format!("  - {} {a}\n  - {} {b}\n    ({})\n", c.a_sentence, c.b_sentence, c.reason));
        }
    } else {
        s.push_str(&format!("  {} {}\n", best.sentence, cite(&best.docs)));
        if !v.extra.is_empty() {
            s.push_str(&format!("{}:\n", l.also));
            for e in &v.extra {
                s.push_str(&format!("  - {} {}\n", e.sentence, cite(&e.docs)));
            }
        }
    }
    if !v.notes.is_empty() {
        s.push_str(&format!("{}:\n", l.notes));
        for n in &v.notes {
            s.push_str(&format!("  * {n}\n"));
        }
    }
    s.push_str(&format!("{}:\n", l.sources));
    for (i, d) in used.iter().enumerate() {
        s.push_str(&format!("  {}\n", pages[*d].citation(i + 1)));
    }
    s
}

pub fn to_documents(pages: &[Page]) -> Vec<Document> {
    pages.iter().map(|p| Document { url: p.url.clone(), title: p.title.clone(), domain: p.domain.clone(), published: p.published, text: p.text.clone() }).collect()
}

impl Assistant {
    pub fn new(mut cfg: Config, learn: bool, offline: bool) -> mkr_core::Result<Assistant> {
        if offline {
            cfg.online = false;
        }
        let paths = cfg.paths();
        std::fs::create_dir_all(&cfg.data_dir)?;
        let engine = if cfg.model_path.exists() { spark_inference::qio::load_engine(&cfg.model_path).ok() } else { None };
        let search = if cfg.online && !offline { SearchEngine::from_config(&cfg).ok() } else { None };
        Ok(Assistant {
            mem: Memory::open(&paths.memory_db())?,
            kb: Knowledge::open(&paths.knowledge_db())?,
            engine,
            search,
            session: format!("chat-{}", now_unix()),
            history: Vec::new(),
            learn,
            approve_single: false,
            interactive: true,
            verbose: false,
            cfg,
        })
    }

    fn say(out: &mut dyn Write, msg: &str) {
        let _ = writeln!(out, "MKR: {msg}");
    }

    fn ask_yes_no(&self, out: &mut dyn Write, prompt: &str) -> bool {
        if !self.interactive {
            return false;
        }
        let _ = write!(out, "MKR: {prompt} [y/N] ");
        let _ = out.flush();
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line).is_ok() && matches!(line.trim().to_lowercase().as_str(), "y" | "yes" | "ن" | "نعم")
    }

    fn is_personal(q: &str) -> bool {
        let t = text::tokens(q, false);
        t.iter().any(|w| matches!(w.as_str(), "my" | "me" | "mine" | "i" | "اسمي" | "لي" | "عندي" | "انا" | "اني" | "افضل" | "احب"))
    }

    /// Handle one user message and write the response.
    pub fn respond(&mut self, input: &str, out: &mut dyn Write) -> mkr_core::Result<()> {
        let input = input.trim();
        if input.is_empty() {
            return Ok(());
        }
        let _ = self.mem.add_message(&self.session, "user", input);
        // explicit memory command
        if let Some(fact) = parse_remember_command(input) {
            let (id, new) = self.mem.remember(&fact, "")?;
            Self::say(out, &if new { format!("Stored in your memory (#{id}): {fact}") } else { format!("I already had that (#{id}).") });
            return Ok(());
        }
        let content_tokens = text::tokens(input, true);
        let lang = text::detect_lang(input);
        if Self::is_personal(input) && !input.contains("http") {
            self.answer_personal(input, out)?;
        } else if content_tokens.len() < 2 {
            self.chat_only(input, out);
        } else {
            self.answer_question(input, lang, out)?;
        }
        // propose (never silently store) personal statements
        for p in memory_proposals(input) {
            if self.ask_yes_no(out, &format!("Remember \"{p}\"?")) {
                let (id, _) = self.mem.remember(&p, "")?;
                Self::say(out, &format!("Stored in your memory (#{id})."));
            }
        }
        Ok(())
    }

    fn answer_personal(&mut self, q: &str, out: &mut dyn Write) -> mkr_core::Result<()> {
        Self::say(out, "Checking what you asked me to remember...");
        let items = self.mem.recall(q, 3)?;
        if items.is_empty() {
            Self::say(out, "I have nothing stored about that. You can tell me with: remember that <fact>");
        } else {
            for it in items {
                let _ = writeln!(out, "  - {} (memory #{})", it.content, it.id);
            }
            let _ = writeln!(out, "  [layer: {}]", Layer::UserMemory.label());
        }
        Ok(())
    }

    fn chat_only(&mut self, input: &str, out: &mut dyn Write) {
        match &self.engine {
            Some(e) => {
                let prompt = chat_prompt(&self.history, input);
                let mut s = Sampler::new(self.cfg.temperature, 40, 0.95, 1.1, now_unix() as u64);
                let r = generate(e, &prompt, &GenConfig { max_new_tokens: self.cfg.max_new_tokens, stop_on_eos: true }, &mut s, &mut |_| {});
                Self::say(out, &format!("{}", r.text.trim()));
                let _ = writeln!(out, "  [layer: {}; mean log-prob {:.2}]", Layer::ModelWeights.label(), r.mean_logprob);
                self.history.push((true, input.to_string()));
                self.history.push((false, r.text));
            }
            None => Self::say(out, "No model is loaded (train one with `mkr train`). For factual questions I can still search."),
        }
    }

    fn can_search(&self) -> bool {
        self.search.is_some()
    }

    fn answer_question(&mut self, q: &str, lang: Lang, out: &mut dyn Write) -> mkr_core::Result<()> {
        let ts = is_time_sensitive(q);
        Self::say(out, "I will check my internal knowledge first.");
        // 1. stored, verified knowledge
        let hits = self.kb.query(q, 3, false)?;
        if let Some(h) = hits.first() {
            let age_days = (now_unix() - h.fact.updated) / 86400;
            if h.coverage >= self.cfg.kb_min_score {
                if ts && age_days > 7 && self.can_search() {
                    Self::say(out, &format!("I have a stored answer but it is {age_days} days old and your question looks time-sensitive; checking for updates."));
                } else {
                    let f = &h.fact;
                    let _ = writeln!(out, "  {}", f.claim);
                    for (i, s) in f.sources.iter().enumerate() {
                        let published = s.published.map(format_date).unwrap_or_else(|| "date unknown".into());
                        let _ = writeln!(out, "  [{}] {} — {} (published {published}) {}", i + 1, s.title, s.domain, s.url);
                    }
                    let _ = writeln!(out, "  [layer: {}; stored {} days ago; confidence {:.0}%{}]", Layer::Retrieved.label(), age_days, f.confidence * 100.0, if f.user_verified { "; user-verified" } else { "" });
                    return Ok(());
                }
            }
        }
        // 2. internal model: evaluate whether it knows
        if let Some(e) = &self.engine {
            let prompt = chat_prompt(&self.history, q);
            let r = generate(e, &prompt, &GenConfig { max_new_tokens: self.cfg.max_new_tokens.min(120), stop_on_eos: true }, &mut Sampler::greedy(), &mut |_| {});
            let confident = r.mean_logprob >= self.cfg.model_confidence_threshold && !r.text.trim().is_empty();
            if confident && self.cfg.trust_model_answers && !ts {
                Self::say(out, r.text.trim());
                let _ = writeln!(out, "  [layer: {}; mean log-prob {:.2} nats/byte >= {:.2}. This comes from the network's weights and has NOT been checked against sources.]", Layer::ModelWeights.label(), r.mean_logprob, self.cfg.model_confidence_threshold);
                return Ok(());
            }
            Self::say(out, &format!("My internal model is not confident enough to answer from its weights (mean log-prob {:.2}, entropy {:.2}).", r.mean_logprob, r.mean_entropy));
        }
        // 3. external evidence
        if !self.can_search() {
            Self::say(out, if self.cfg.online { "Search is unavailable (no usable provider)." } else { "I am offline and have no stored knowledge that answers this. Re-run without --offline (or set online = true) to let me search." });
            return Ok(());
        }
        self.research(q, lang, out)
    }

    /// Search, fetch, verify, answer with citations, optionally store.
    pub fn research(&mut self, q: &str, lang: Lang, out: &mut dyn Write) -> mkr_core::Result<()> {
        let Some(eng) = &self.search else {
            Self::say(out, "Search is disabled.");
            return Ok(());
        };
        Self::say(out, "Searching trusted sources...");
        let outcome = match eng.search(q) {
            Ok(o) => o,
            Err(e) => {
                Self::say(out, &format!("Search failed: {e}"));
                return Ok(());
            }
        };
        for (p, e) in &outcome.errors {
            let _ = writeln!(out, "  (provider {p} unavailable: {e})");
        }
        let mut per_provider: std::collections::BTreeMap<String, usize> = Default::default();
        for h in &outcome.hits {
            *per_provider.entry(h.provider.clone()).or_default() += 1;
        }
        for (p, n) in &per_provider {
            let _ = self.mem.log_search(q, p, *n);
        }
        if outcome.hits.is_empty() {
            Self::say(out, "No results were found.");
            return Ok(());
        }
        let (pages, errs) = eng.fetch_top(&outcome.hits, self.cfg.max_pages);
        if self.verbose {
            for (u, e) in &errs {
                let _ = writeln!(out, "  (could not read {u}: {e})");
            }
        }
        if pages.is_empty() {
            Self::say(out, "I found results but could not read any of the pages.");
            return Ok(());
        }
        let docs = to_documents(&pages);
        let verdict = verify(q, &docs, &VerifyConfig::new(now_unix()));
        Self::say(out, match verdict.status {
            Status::Verified => "Information retrieved and verified.",
            Status::SingleSource => "Information retrieved, but only one independent source supports it.",
            Status::Conflicting => "Information retrieved, but the sources disagree.",
            Status::Insufficient => "Information retrieved, but nothing in it answers your question.",
        });
        let _ = write!(out, "{}", compose(&verdict, &pages, lang));
        let _ = writeln!(out, "  [layer: {}]", Layer::Retrieved.label());
        if self.learn {
            let topic = text::keywords(q);
            let mut policy = LearnPolicy { ttl_days: self.cfg.knowledge_ttl_days, ..Default::default() };
            policy.approve_single_source = self.approve_single;
            if verdict.status == Status::SingleSource && !self.approve_single && self.ask_yes_no(out, "Save this unconfirmed claim to local knowledge anyway?") {
                policy.approve_single_source = true;
            }
            match learn_from_verdict(&self.kb, &topic, &verdict, &docs, now_unix(), &policy)? {
                LearnOutcome::Stored { fact_id, created } => {
                    let _ = writeln!(out, "  Saved to local knowledge (#{fact_id}{}).", if created { "" } else { ", sources merged" });
                }
                LearnOutcome::Skipped(why) => {
                    if self.verbose || verdict.status == Status::SingleSource {
                        let _ = writeln!(out, "  Not saved: {why}.");
                    }
                }
            }
        }
        Ok(())
    }
}
