//! Guarded neural fine-tuning. A candidate model is trained on approved text (mixed with replayed
//! anchor data) and compared against the current model on two held-out sets:
//!   * the anchor-validation set  — must not get worse than `max_forget` (catastrophic forgetting guard)
//!   * the new-data validation set — must improve by at least `min_gain`
//! Only then is the live model replaced, after the previous version is backed up for rollback.
use mkr_core::time::now_unix;
use spark_byte_engine::Corpus;
use spark_neural::io::{load_model, save_model};
use spark_training::{evaluate, train, TrainConfig};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct FineTuneConfig {
    pub steps: u64,
    pub batch: usize,
    pub ctx: usize,
    pub peak_lr: f32,
    /// fraction of training batches drawn from replayed anchor data (rehearsal against forgetting)
    pub replay_ratio: f32,
    /// maximum tolerated relative increase of anchor validation loss
    pub max_forget: f32,
    /// ...or this many nats, whichever is larger (relative limits are meaningless when the loss is near zero)
    pub max_forget_abs: f32,
    /// minimum required relative decrease of new-data validation loss
    pub min_gain: f32,
    pub min_new_bytes: usize,
    pub eval_batches: usize,
    pub seed: u64,
    pub work_dir: PathBuf,
}

impl Default for FineTuneConfig {
    fn default() -> Self {
        FineTuneConfig { steps: 100, batch: 8, ctx: 128, peak_lr: 3e-4, replay_ratio: 0.5, max_forget: 0.02, max_forget_abs: 0.02, min_gain: 0.03, min_new_bytes: 2000, eval_batches: 32, seed: 7, work_dir: PathBuf::from("data/learning/work") }
    }
}

#[derive(Clone, Debug)]
pub struct FineTuneReport {
    pub accepted: bool,
    pub reason: String,
    pub anchor_before: f32,
    pub anchor_after: f32,
    pub new_before: f32,
    pub new_after: f32,
    pub removed_suspicious: usize,
    pub backup: Option<PathBuf>,
    pub candidate: PathBuf,
}

/// Deterministic paragraph-level split: every `k`-th paragraph goes to validation.
fn split_paragraphs(text: &str, k: usize) -> (String, String) {
    let paras: Vec<&str> = text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()).collect();
    if paras.len() >= k * 2 {
        let (mut tr, mut va) = (Vec::new(), Vec::new());
        for (i, p) in paras.iter().enumerate() {
            if i % k == k - 1 { va.push(*p) } else { tr.push(*p) }
        }
        return (tr.join("\n\n"), va.join("\n\n"));
    }
    // too few paragraphs: split by bytes on a char boundary
    let cut = (text.len() as f32 * (1.0 - 1.0 / k as f32)) as usize;
    let mut c = cut.min(text.len());
    while !text.is_char_boundary(c) {
        c -= 1;
    }
    (text[..c].to_string(), text[c..].to_string())
}

fn repeat_to(text: &str, target: usize) -> String {
    if text.is_empty() || target == 0 {
        return String::new();
    }
    let mut out = String::with_capacity(target + text.len());
    while out.len() < target {
        out.push_str(text);
        out.push_str("\n\n");
    }
    out
}

pub fn finetune(model_path: &Path, new_text: &str, anchor_text: &str, cfg: &FineTuneConfig, history_dir: &Path, log: &mut dyn FnMut(&str)) -> Result<FineTuneReport, String> {
    if !(0.0..0.95).contains(&cfg.replay_ratio) {
        return Err("replay_ratio must be within 0..0.95".into());
    }
    let mut baseline = load_model(model_path).map_err(|e| format!("cannot load model (fine-tuning needs the f32 .model, not a quantized one): {e}"))?;
    let (clean_new, rep) = mkr_verification::sanitize_web_text(new_text);
    if rep.removed > 0 {
        log(&format!("removed {} instruction-like sentence(s) from the approved text", rep.removed));
    }
    if clean_new.len() < cfg.min_new_bytes {
        return Err(format!("not enough approved text: {} bytes (minimum {}). Approve more reviewed material first.", clean_new.len(), cfg.min_new_bytes));
    }
    if anchor_text.len() < cfg.ctx * 4 {
        return Err("anchor text too small: provide a sample of the original training data to guard against forgetting".into());
    }
    let ctx = cfg.ctx.min(baseline.cfg.max_ctx);
    let (new_train, new_val) = split_paragraphs(&clean_new, 7);
    let (anchor_train, anchor_val) = split_paragraphs(anchor_text, 5);
    if new_val.len() <= ctx || anchor_val.len() <= ctx {
        return Err(format!("validation splits too small for context {ctx}; provide more text"));
    }
    let new_val_c = Corpus::from_text(&new_val);
    let anchor_val_c = Corpus::from_text(&anchor_val);
    let anchor_before = evaluate(&mut baseline, &anchor_val_c, cfg.batch, ctx, cfg.eval_batches).loss;
    let new_before = evaluate(&mut baseline, &new_val_c, cfg.batch, ctx, cfg.eval_batches).loss;
    log(&format!("baseline: anchor loss {anchor_before:.4}, new-data loss {new_before:.4}"));

    // training mix: new data + replayed anchor data
    let anchor_target = if cfg.replay_ratio > 0.0 { (new_train.len() as f32 * cfg.replay_ratio / (1.0 - cfg.replay_ratio)) as usize } else { 0 };
    let replay: String = repeat_to(&anchor_train, anchor_target).chars().take(anchor_target.max(1)).collect();
    let mut mix = new_train.clone().into_bytes();
    if !replay.is_empty() {
        mix.push(0);
        mix.extend_from_slice(replay.as_bytes());
    }
    let train_c = Corpus::from_bytes(mix);

    std::fs::create_dir_all(&cfg.work_dir).map_err(|e| e.to_string())?;
    let ckpt_dir = cfg.work_dir.join("candidate");
    let _ = std::fs::remove_dir_all(&ckpt_dir);
    let mut cand = load_model(model_path)?;
    let tc = TrainConfig {
        steps: cfg.steps,
        batch: cfg.batch,
        ctx,
        accum: 1,
        peak_lr: cfg.peak_lr,
        min_lr: cfg.peak_lr / 10.0,
        warmup: (cfg.steps / 10).max(1),
        eval_every: (cfg.steps / 5).max(1),
        eval_batches: cfg.eval_batches,
        ckpt_every: 0,
        out_dir: ckpt_dir.clone(),
        patience: 3,
        stop_on_overfit: true,
        seed: cfg.seed,
        log_every: 0,
        ..Default::default()
    };
    let report = train(&mut cand, &train_c, &new_val_c, &tc, None, &mut |_e| {})?;
    log(&format!("fine-tuned {} steps (overfit detected: {})", report.steps_done, report.overfit_detected));
    // use the checkpoint with the best new-data validation loss
    let best = ckpt_dir.join("best.model");
    let mut cand = if best.exists() { load_model(&best)? } else { cand };
    let anchor_after = evaluate(&mut cand, &anchor_val_c, cfg.batch, ctx, cfg.eval_batches).loss;
    let new_after = evaluate(&mut cand, &new_val_c, cfg.batch, ctx, cfg.eval_batches).loss;
    let forget = (anchor_after - anchor_before) / anchor_before;
    let gain = (new_before - new_after) / new_before;
    log(&format!("candidate: anchor loss {anchor_after:.4} ({:+.1}%), new-data loss {new_after:.4} ({:+.1}% gain)", forget * 100.0, gain * 100.0));

    let candidate_path = cfg.work_dir.join("candidate.model");
    save_model(&cand, &candidate_path)?;
    let mut out = FineTuneReport { accepted: false, reason: String::new(), anchor_before, anchor_after, new_before, new_after, removed_suspicious: rep.removed, backup: None, candidate: candidate_path.clone() };
    if !anchor_after.is_finite() || !new_after.is_finite() {
        out.reason = "candidate produced non-finite loss".into();
    } else if anchor_after - anchor_before > (anchor_before * cfg.max_forget).max(cfg.max_forget_abs) {
        out.reason = format!("rejected: anchor loss rose {:.1}% ({anchor_before:.3} -> {anchor_after:.3}); limit is max({:.1}%, {} nats): catastrophic-forgetting guard", forget * 100.0, cfg.max_forget * 100.0, cfg.max_forget_abs);
    } else if gain < cfg.min_gain {
        out.reason = format!("rejected: new-data loss improved only {:.1}% (need {:.1}%)", gain * 100.0, cfg.min_gain * 100.0);
    } else {
        std::fs::create_dir_all(history_dir).map_err(|e| e.to_string())?;
        let stem = model_path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        let backup = history_dir.join(format!("{stem}-{}.model", now_unix()));
        std::fs::copy(model_path, &backup).map_err(|e| format!("cannot back up current model: {e}"))?;
        save_model(&cand, model_path)?;
        out.accepted = true;
        out.backup = Some(backup);
        out.reason = format!("accepted: new-data loss improved {:.1}%, anchor loss changed {:+.1}%", gain * 100.0, forget * 100.0);
    }
    if !out.accepted {
        let rej = cfg.work_dir.join("rejected.model");
        let _ = std::fs::rename(&candidate_path, &rej);
        out.candidate = rej;
    }
    Ok(out)
}

/// Restore the most recent backup over the live model. The model being replaced is kept in history too.
pub fn rollback(model_path: &Path, history_dir: &Path) -> Result<PathBuf, String> {
    let stem = model_path.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string();
    let mut backups: Vec<PathBuf> = std::fs::read_dir(history_dir)
        .map_err(|e| format!("no history directory: {e}"))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).map_or(false, |n| n.starts_with(&format!("{stem}-")) && n.ends_with(".model") && !n.contains("rolledback")))
        .collect();
    backups.sort();
    let latest = backups.pop().ok_or("no backups to roll back to")?;
    load_model(&latest).map_err(|e| format!("backup {} is not a valid model: {e}", latest.display()))?;
    let replaced = history_dir.join(format!("{stem}-rolledback-{}.model", now_unix()));
    std::fs::copy(model_path, &replaced).map_err(|e| e.to_string())?;
    let tmp = model_path.with_extension("tmp");
    std::fs::copy(&latest, &tmp).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, model_path).map_err(|e| e.to_string())?;
    std::fs::remove_file(&latest).map_err(|e| e.to_string())?;
    Ok(latest)
}
