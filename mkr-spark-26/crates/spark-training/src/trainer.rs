use crate::checkpoint::{save_checkpoint, EvalPoint, ResumeState, TrainState};
use crate::eval::evaluate;
use crate::sys::peak_rss_bytes;
use spark_byte_engine::{Batcher, Corpus};
use spark_neural::{io::save_model, Model};
use spark_optimizer::{clip_grad_norm, AdamW, AdamWConfig, Schedule};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub steps: u64,
    pub batch: usize,
    pub ctx: usize,
    pub accum: usize,
    pub peak_lr: f32,
    pub min_lr: f32,
    pub warmup: u64,
    pub weight_decay: f32,
    pub clip: f32,
    pub eval_every: u64,
    pub eval_batches: usize,
    pub ckpt_every: u64,
    pub keep_ckpts: usize,
    pub out_dir: PathBuf,
    /// Number of evaluations without val improvement (while train loss keeps falling) that signals overfitting.
    pub patience: usize,
    pub stop_on_overfit: bool,
    pub seed: u64,
    pub log_every: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            steps: 1000,
            batch: 8,
            ctx: 128,
            accum: 1,
            peak_lr: 3e-3,
            min_lr: 3e-4,
            warmup: 50,
            weight_decay: 0.05,
            clip: 1.0,
            eval_every: 100,
            eval_batches: 16,
            ckpt_every: 200,
            keep_ckpts: 3,
            out_dir: PathBuf::from("checkpoints"),
            patience: 5,
            stop_on_overfit: true,
            seed: 1234,
            log_every: 10,
        }
    }
}

pub enum Event<'a> {
    Resumed { step: u64 },
    Step { step: u64, loss: f32, lr: f32, grad_norm: f32, tokens_per_sec: f32 },
    Eval { point: &'a EvalPoint, ppl: f32, bpb: f32, improved: bool },
    Overfitting { step: u64, best_step: u64, best_val: f32 },
    Checkpoint { path: &'a Path },
    Done,
}

#[derive(Clone, Debug)]
pub struct TrainReport {
    pub steps_done: u64,
    pub final_train_loss: f32,
    pub first_train_loss: f32,
    pub best_val: f32,
    pub best_step: u64,
    pub overfit_detected: bool,
    pub stopped_early: bool,
    pub elapsed_secs: f32,
    pub peak_rss: u64,
    pub params: usize,
    pub history: Vec<EvalPoint>,
}

fn mix(seed: u64, a: u64, b: u64) -> u64 {
    let mut z = seed ^ a.wrapping_mul(0x9E3779B97F4A7C15) ^ b.wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

pub fn train(
    model: &mut Model,
    train_c: &Corpus,
    val_c: &Corpus,
    cfg: &TrainConfig,
    resume: Option<ResumeState>,
    on_event: &mut dyn FnMut(&Event),
) -> Result<TrainReport, String> {
    if cfg.ctx > model.cfg.max_ctx {
        return Err(format!("ctx {} exceeds model max_ctx {}", cfg.ctx, model.cfg.max_ctx));
    }
    if cfg.steps == 0 || cfg.batch == 0 || cfg.accum == 0 {
        return Err("steps, batch and accum must be > 0".into());
    }
    if train_c.len() < 2 || val_c.len() < 2 {
        return Err("train/validation corpus too small".into());
    }
    let mut opt = AdamW::new(AdamWConfig { weight_decay: cfg.weight_decay, ..Default::default() }, model.g.params());
    let mut state = TrainState::default();
    if let Some(r) = resume {
        opt.load_state(&r.optim)?;
        state = r.state;
        on_event(&Event::Resumed { step: state.step });
    }
    let sched = Schedule { peak_lr: cfg.peak_lr, min_lr: cfg.min_lr, warmup_steps: cfg.warmup, total_steps: cfg.steps };
    let start = Instant::now();
    let mut ema: Option<f32> = None;
    let mut first_loss = f32::NAN;
    let mut overfit = false;
    let mut stopped_early = false;
    let mut evals_since_best = 0usize;
    let mut train_at_best = f32::INFINITY;
    let best_path = cfg.out_dir.join("best.model");

    while state.step < cfg.steps {
        let t0 = Instant::now();
        let lr = sched.lr_at(state.step);
        model.g.zero_grad();
        let mut loss_sum = 0.0f32;
        for micro in 0..cfg.accum {
            let mut b = Batcher::new(train_c, cfg.batch, cfg.ctx, mix(cfg.seed, state.step, micro as u64));
            let batch = b.next_random();
            let l = model.loss(&batch.inputs, &batch.targets, batch.bsz, batch.t);
            loss_sum += model.g.scalar(l);
            let scaled = model.g.scale(l, 1.0 / cfg.accum as f32);
            model.g.backward(scaled);
            model.g.reset();
        }
        let loss = loss_sum / cfg.accum as f32;
        if !loss.is_finite() {
            return Err(format!("training diverged: loss is {loss} at step {} (try a lower learning rate)", state.step));
        }
        let gn = clip_grad_norm(model.g.params_mut(), cfg.clip);
        if !gn.is_finite() {
            return Err(format!("non-finite gradient norm at step {}", state.step));
        }
        opt.update(model.g.params_mut(), lr);
        state.step += 1;
        if first_loss.is_nan() {
            first_loss = loss;
        }
        ema = Some(match ema {
            None => loss,
            Some(e) => 0.9 * e + 0.1 * loss,
        });
        if cfg.log_every > 0 && (state.step % cfg.log_every == 0 || state.step == 1) {
            let tps = (cfg.batch * cfg.ctx * cfg.accum) as f32 / t0.elapsed().as_secs_f32().max(1e-6);
            on_event(&Event::Step { step: state.step, loss, lr, grad_norm: gn, tokens_per_sec: tps });
        }
        let last = state.step == cfg.steps;
        if (cfg.eval_every > 0 && state.step % cfg.eval_every == 0) || last {
            let r = evaluate(model, val_c, cfg.batch, cfg.ctx, cfg.eval_batches);
            let point = EvalPoint { step: state.step, train_loss: ema.unwrap(), val_loss: r.loss };
            let improved = r.loss < state.best_val;
            if improved {
                state.best_val = r.loss;
                state.best_step = state.step;
                train_at_best = ema.unwrap();
                evals_since_best = 0;
                save_model(model, &best_path)?;
            } else {
                evals_since_best += 1;
            }
            state.history.push(point.clone());
            on_event(&Event::Eval { point: &point, ppl: r.perplexity, bpb: r.bits_per_byte, improved });
            if !improved && evals_since_best >= cfg.patience && ema.unwrap() < train_at_best * 0.98 {
                overfit = true;
                on_event(&Event::Overfitting { step: state.step, best_step: state.best_step, best_val: state.best_val });
                if cfg.stop_on_overfit {
                    stopped_early = true;
                    break;
                }
            }
        }
        if cfg.ckpt_every > 0 && (state.step % cfg.ckpt_every == 0 || last) {
            let p = save_checkpoint(&cfg.out_dir, model, &opt.state_bytes(), &state, cfg.keep_ckpts)?;
            on_event(&Event::Checkpoint { path: &p });
        }
    }
    on_event(&Event::Done);
    Ok(TrainReport {
        steps_done: state.step,
        final_train_loss: ema.unwrap_or(f32::NAN),
        first_train_loss: first_loss,
        best_val: state.best_val,
        best_step: state.best_step,
        overfit_detected: overfit,
        stopped_early,
        elapsed_secs: start.elapsed().as_secs_f32(),
        peak_rss: peak_rss_bytes(),
        params: model.param_count(),
        history: state.history,
    })
}
