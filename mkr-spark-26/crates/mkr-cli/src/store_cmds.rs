//! memory / knowledge / learn commands: everything the user can inspect, correct and delete.
use crate::args::Args;
use mkr_core::time::format_date;
use mkr_core::Config;
use mkr_knowledge::Knowledge;
use mkr_learning::{finetune, rollback, CandidateStatus, FineTuneConfig, Queue};
use mkr_memory::Memory;
use std::path::PathBuf;

fn e<T: std::fmt::Display>(x: T) -> String {
    x.to_string()
}
fn id_arg(a: &Args, i: usize) -> Result<i64, String> {
    a.pos.get(i).ok_or("missing id")?.trim_start_matches('#').parse::<i64>().map_err(|_| "id must be a number".to_string())
}
fn confirm(a: &Args, what: &str) -> Result<(), String> {
    if a.has("yes") {
        return Ok(());
    }
    Err(format!("this will {what}. Re-run with --yes to confirm."))
}

pub fn cmd_memory(a: &Args, cfg: &Config) -> Result<(), String> {
    let m = Memory::open(&cfg.paths().memory_db()).map_err(e)?;
    match a.pos.get(1).map(|s| s.as_str()) {
        Some("add") => {
            let t = a.text(2);
            let (id, new) = m.remember(&t, a.flag("tags").unwrap_or("")).map_err(e)?;
            println!("{} #{id}", if new { "Stored" } else { "Already stored as" });
        }
        Some("list") | None => {
            let items = m.list_items().map_err(e)?;
            if items.is_empty() {
                println!("(no memories stored)");
            }
            for it in items {
                println!("#{:<4} {}  [{}]{}", it.id, it.content, format_date(it.created), if it.tags.is_empty() { String::new() } else { format!(" tags: {}", it.tags) });
            }
        }
        Some("recall") => {
            let items = m.recall(&a.text(2), 5).map_err(e)?;
            if items.is_empty() {
                println!("(nothing relevant)");
            }
            for it in items {
                println!("#{:<4} {}", it.id, it.content);
            }
        }
        Some("update") => {
            let id = id_arg(a, 2)?;
            println!("{}", if m.update_item(id, &a.text(3)).map_err(e)? { "Updated." } else { "No such memory." });
        }
        Some("delete") => {
            println!("{}", if m.delete_item(id_arg(a, 2)?).map_err(e)? { "Deleted." } else { "No such memory." });
        }
        Some("clear") => {
            confirm(a, "delete ALL memory items")?;
            println!("Deleted {} item(s).", m.clear_items().map_err(e)?);
        }
        Some("forget-chats") => {
            confirm(a, "delete the stored conversation log")?;
            println!("Deleted {} message(s).", m.clear_conversations().map_err(e)?);
        }
        Some("prefs") => match a.pos.get(2).map(|s| s.as_str()) {
            Some("set") => {
                m.set_pref(a.pos.get(3).ok_or("usage: mkr memory prefs set <key> <value>")?, &a.text(4)).map_err(e)?;
                println!("ok");
            }
            Some("get") => println!("{}", m.get_pref(a.pos.get(3).ok_or("usage: mkr memory prefs get <key>")?).map_err(e)?.unwrap_or_default()),
            Some("delete") => println!("{}", m.delete_pref(a.pos.get(3).ok_or("usage: mkr memory prefs delete <key>")?).map_err(e)?),
            _ => {
                for (k, v) in m.list_prefs().map_err(e)? {
                    println!("{k} = {v}");
                }
            }
        },
        Some("history") => {
            let n = a.pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
            for h in m.search_history(n).map_err(e)? {
                println!("{}  {:<10} {:>2} result(s)  {}", format_date(h.ts), h.provider, h.results, h.query);
            }
        }
        _ => return Err("usage: mkr memory <add|list|recall|update|delete|clear|forget-chats|prefs|history>".into()),
    }
    Ok(())
}

fn print_fact(f: &mkr_knowledge::Fact, full: bool) {
    println!("#{:<4} [{}{}] conf {:.0}%  {}", f.id, f.status.label(), if f.user_verified { ", user-verified" } else { "" }, f.confidence * 100.0, f.claim);
    if full {
        println!("      topic: {}   stored {}   expires {}", f.topic, format_date(f.created), f.expires_at.map(format_date).unwrap_or_else(|| "never".into()));
        for s in &f.sources {
            println!("      - {} ({}) {}\n        \"{}\"", s.title, s.domain, s.url, s.excerpt);
        }
    }
}

pub fn cmd_knowledge(a: &Args, cfg: &Config) -> Result<(), String> {
    let k = Knowledge::open(&cfg.paths().knowledge_db()).map_err(e)?;
    match a.pos.get(1).map(|s| s.as_str()) {
        Some("list") | None => {
            let v = k.list(a.num::<usize>("limit")?.unwrap_or(30), 0).map_err(e)?;
            if v.is_empty() {
                println!("(no stored knowledge yet)");
            }
            for f in v {
                print_fact(&f, false);
            }
        }
        Some("query") => {
            for h in k.query(&a.text(2), 5, a.has("all")).map_err(e)? {
                print_fact(&h.fact, true);
            }
        }
        Some("show") => match k.get(id_arg(a, 2)?).map_err(e)? {
            Some(f) => print_fact(&f, true),
            None => println!("No such fact."),
        },
        Some("correct") => {
            let id = id_arg(a, 2)?;
            println!("{}", if k.correct(id, &a.text(3)).map_err(e)? { "Corrected and marked user-verified." } else { "No such fact." });
        }
        Some("delete") => println!("{}", if k.delete(id_arg(a, 2)?).map_err(e)? { "Deleted." } else { "No such fact." }),
        Some("expire") => println!("Expired {} fact(s).", k.expire_due(mkr_core::time::now_unix()).map_err(e)?),
        Some("prune") => println!("Removed {} long-expired fact(s).", k.prune_expired(mkr_core::time::now_unix(), a.num::<u64>("days")?.unwrap_or(30)).map_err(e)?),
        Some("stats") => {
            let (f, act, s) = k.stats().map_err(e)?;
            println!("{f} fact(s), {act} active, {s} source(s)");
        }
        _ => return Err("usage: mkr knowledge <list|query|show|correct|delete|expire|prune|stats>".into()),
    }
    Ok(())
}

fn read_text_path(p: &str) -> Result<String, String> {
    let path = PathBuf::from(p);
    let mut files = Vec::new();
    if path.is_dir() {
        for en in std::fs::read_dir(&path).map_err(e)?.flatten() {
            if en.path().extension().map_or(false, |x| x == "txt" || x == "md") {
                files.push(en.path());
            }
        }
        files.sort();
    } else {
        files.push(path);
    }
    let mut out = String::new();
    for f in files {
        out.push_str(&mkr_byte_text(&std::fs::read(&f).map_err(|x| format!("{}: {x}", f.display()))?));
        out.push_str("\n\n");
    }
    Ok(out)
}
fn mkr_byte_text(b: &[u8]) -> String {
    spark_byte_engine::sanitize_text(&String::from_utf8_lossy(b))
}

pub fn cmd_learn(a: &Args, cfg: &Config) -> Result<(), String> {
    let learning = cfg.paths().learning_dir();
    let q = Queue::open(&learning.join("queue.db")).map_err(e)?;
    match a.pos.get(1).map(|s| s.as_str()) {
        Some("stage") => {
            let mut added = 0;
            if let Some(id) = a.num::<i64>("fact")? {
                let k = Knowledge::open(&cfg.paths().knowledge_db()).map_err(e)?;
                let f = k.get(id).map_err(e)?.ok_or("no such fact")?;
                added += q.add(&f.claim, &format!("knowledge #{id}")).map_err(e)?.is_some() as usize;
            }
            if let Some(t) = a.flag("text") {
                added += q.add(t, "user text").map_err(e)?.is_some() as usize;
            }
            if let Some(p) = a.flag("file") {
                for para in read_text_path(p)?.split("\n\n") {
                    if para.trim().chars().count() >= 20 {
                        match q.add(para, &format!("file {p}")) {
                            Ok(Some(_)) => added += 1,
                            Ok(None) => {}
                            Err(x) => println!("skipped a paragraph: {x}"),
                        }
                    }
                }
            }
            println!("Staged {added} candidate(s) for review (status: pending). Nothing is used for training until you approve it.");
        }
        Some("queue") | None => {
            let st = match a.pos.get(2).map(|s| s.as_str()) {
                Some("pending") => Some(CandidateStatus::Pending),
                Some("approved") => Some(CandidateStatus::Approved),
                Some("rejected") => Some(CandidateStatus::Rejected),
                Some("used") => Some(CandidateStatus::Used),
                _ => None,
            };
            for c in q.list(st).map_err(e)? {
                println!("#{:<4} {:?}  ({})  {}", c.id, c.status, c.origin, c.text.chars().take(100).collect::<String>());
            }
        }
        Some("approve") | Some("reject") => {
            let approve = a.pos[1] == "approve";
            let ids: Vec<i64> = if a.has("all") { q.list(Some(CandidateStatus::Pending)).map_err(e)?.into_iter().map(|c| c.id).collect() } else { a.pos.iter().skip(2).filter_map(|s| s.trim_start_matches('#').parse().ok()).collect() };
            for id in &ids {
                if approve { q.approve(*id) } else { q.reject(*id) }.map_err(e)?;
            }
            println!("{} {} candidate(s).", if approve { "Approved" } else { "Rejected" }, ids.len());
        }
        Some("finetune") => {
            let anchor = read_text_path(a.flag("anchor").ok_or("--anchor <file|dir> is required: a sample of the ORIGINAL training text, used to detect forgetting")?)?;
            let new_text = q.approved_corpus().map_err(e)?;
            let mp = a.flag("model").map(PathBuf::from).unwrap_or_else(|| cfg.model_path.clone());
            let d = FineTuneConfig::default();
            let fc = FineTuneConfig {
                steps: a.num::<u64>("steps")?.unwrap_or(d.steps),
                batch: a.num::<usize>("batch")?.unwrap_or(d.batch),
                ctx: a.num::<usize>("ctx")?.unwrap_or(d.ctx),
                peak_lr: a.num::<f32>("lr")?.unwrap_or(d.peak_lr),
                replay_ratio: a.num::<f32>("replay")?.unwrap_or(d.replay_ratio),
                max_forget: a.num::<f32>("max-forget")?.unwrap_or(d.max_forget),
                min_gain: a.num::<f32>("min-gain")?.unwrap_or(d.min_gain),
                work_dir: learning.join("work"),
                ..d
            };
            let rep = finetune(&mp, &new_text, &anchor, &fc, &learning.join("history"), &mut |s| println!("  {s}"))?;
            println!("{}", rep.reason);
            println!("  anchor loss {:.4} -> {:.4}   new-data loss {:.4} -> {:.4}", rep.anchor_before, rep.anchor_after, rep.new_before, rep.new_after);
            if rep.accepted {
                println!("Model updated. Previous version saved to {}. Undo with `mkr learn rollback`.", rep.backup.as_ref().unwrap().display());
                q.mark_approved_used().map_err(e)?;
            } else {
                println!("The live model was NOT changed. Candidate kept at {} for inspection.", rep.candidate.display());
            }
        }
        Some("rollback") => {
            let mp = a.flag("model").map(PathBuf::from).unwrap_or_else(|| cfg.model_path.clone());
            let restored = rollback(&mp, &learning.join("history"))?;
            println!("Restored previous model from {}", restored.display());
        }
        _ => return Err("usage: mkr learn <stage|queue|approve|reject|finetune|rollback>".into()),
    }
    Ok(())
}
