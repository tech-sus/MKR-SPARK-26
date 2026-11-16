use spark_byte_engine::Corpus;
use spark_neural::{Config, Model};
use spark_training::*;
use std::path::PathBuf;

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("mkr_train_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn text() -> String {
    "the quick brown fox jumps over the lazy dog. rust is a systems language. مرحبا بالعالم، هذا اختبار. ".repeat(60)
}

fn quiet() -> impl FnMut(&Event) {
    |_e: &Event| {}
}

#[test]
fn training_from_scratch_reduces_loss() {
    let dir = tmpdir("loss");
    let (tr, va) = Corpus::from_text(&text()).split(0.1);
    let mut m = Model::new(Config::preset("nano").unwrap(), 1).unwrap();
    let cfg = TrainConfig { steps: 150, batch: 4, ctx: 48, eval_every: 50, ckpt_every: 0, out_dir: dir.clone(), warmup: 10, log_every: 0, stop_on_overfit: false, ..Default::default() };
    let before = evaluate(&mut m, &va, 4, 48, 8).loss;
    let rep = train(&mut m, &tr, &va, &cfg, None, &mut quiet()).unwrap();
    let after = evaluate(&mut m, &va, 4, 48, 8).loss;
    println!("val loss {before:.3} -> {after:.3}; train {:.3} -> {:.3}; {:.1}s", rep.first_train_loss, rep.final_train_loss, rep.elapsed_secs);
    assert!(before > 5.0, "initial loss should be near ln(264)=5.58, got {before}");
    assert!(after < before * 0.4, "validation loss did not drop enough: {before} -> {after}");
    assert!(rep.best_val.is_finite());
    assert!(dir.join("best.model").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn resume_matches_uninterrupted_run() {
    let dir_a = tmpdir("res_a");
    let dir_b = tmpdir("res_b");
    let (tr, va) = Corpus::from_text(&text()).split(0.1);
    let base = TrainConfig { batch: 2, ctx: 32, eval_every: 0, ckpt_every: 0, warmup: 5, log_every: 0, ..Default::default() };

    // uninterrupted: 40 steps
    let mut m1 = Model::new(Config::preset("nano").unwrap(), 2).unwrap();
    train(&mut m1, &tr, &va, &TrainConfig { steps: 40, out_dir: dir_a.clone(), ..base.clone() }, None, &mut quiet()).unwrap();

    // interrupted: crash right after the step-20 checkpoint, reload it, continue to step 40
    let mut m2 = Model::new(Config::preset("nano").unwrap(), 2).unwrap();
    let stop_cfg = TrainConfig { steps: 40, ckpt_every: 20, out_dir: dir_b.clone(), ..base.clone() };
    // Use the real API: run to 40 total but stop after the checkpoint at step 20 by catching it in the callback.
    let mut saved: Option<PathBuf> = None;
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        train(&mut m2, &tr, &va, &stop_cfg, None, &mut |e: &Event| {
            if let Event::Checkpoint { path } = e {
                saved = Some(path.to_path_buf());
                panic!("simulated crash after checkpoint");
            }
        })
    }));
    assert!(r.is_err());
    let (mut m3, rs) = load_checkpoint(&dir_b).unwrap();
    assert_eq!(rs.state.step, 20);
    train(&mut m3, &tr, &va, &stop_cfg, Some(rs), &mut quiet()).unwrap();
    let mut worst = 0.0f32;
    for (a, b) in m1.g.params().iter().zip(m3.g.params()) {
        for (x, y) in a.data.iter().zip(&b.data) {
            worst = worst.max((x - y).abs());
        }
    }
    println!("max |param diff| resumed vs uninterrupted: {worst:e}");
    assert!(worst < 1e-5, "resume diverged from uninterrupted run: {worst}");
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn corrupt_newest_checkpoint_falls_back_to_previous() {
    let dir = tmpdir("corrupt");
    let (tr, va) = Corpus::from_text(&text()).split(0.1);
    let mut m = Model::new(Config::preset("nano").unwrap(), 3).unwrap();
    let cfg = TrainConfig { steps: 20, batch: 2, ctx: 32, eval_every: 0, ckpt_every: 10, out_dir: dir.clone(), log_every: 0, ..Default::default() };
    train(&mut m, &tr, &va, &cfg, None, &mut quiet()).unwrap();
    let newest = dir.join("step_0000020.ckpt");
    let mut bytes = std::fs::read(&newest).unwrap();
    let n = bytes.len();
    bytes[n / 2] ^= 0xFF;
    std::fs::write(&newest, bytes).unwrap();
    assert!(load_checkpoint(&newest).is_err());
    let (_, rs) = load_checkpoint(&dir).unwrap();
    assert_eq!(rs.state.step, 10, "should recover from the older valid checkpoint");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn overfitting_is_detected_when_validation_diverges() {
    let dir = tmpdir("overfit");
    // tiny training text, validation text from a different distribution
    let tr_text = "abcabcabcabcabcabcabcabcabcabcabcabcabcabc ".repeat(8);
    let va_text = "zyxwvutsrqponmlkjihgfedcba 9876543210 ".repeat(8);
    let tr = Corpus::from_text(&tr_text);
    let va = Corpus::from_text(&va_text);
    let mut m = Model::new(Config::preset("nano").unwrap(), 4).unwrap();
    let cfg = TrainConfig { steps: 600, batch: 4, ctx: 32, eval_every: 20, patience: 3, peak_lr: 5e-3, ckpt_every: 0, out_dir: dir.clone(), log_every: 0, stop_on_overfit: true, ..Default::default() };
    let rep = train(&mut m, &tr, &va, &cfg, None, &mut quiet()).unwrap();
    println!("overfit={} stopped_early={} at step {} (best val {:.3} @ {})", rep.overfit_detected, rep.stopped_early, rep.steps_done, rep.best_val, rep.best_step);
    assert!(rep.overfit_detected && rep.stopped_early && rep.steps_done < 600);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ram_plans_fit_and_estimate_is_realistic() {
    for gb in [2u32, 4, 8, 16] {
        let p = plan_for_ram_gb(gb);
        let frac = p.est_train_bytes as f64 / (gb as f64 * 1e9);
        println!("{gb}GB -> {} batch {} ctx {} accum {}: est {:.2} GB ({:.0}% of RAM)", p.preset, p.batch, p.ctx, p.accum, p.est_train_bytes as f64 / 1e9, frac * 100.0);
        assert!(frac < 0.6, "{gb}GB plan uses too much of the machine");
    }
    // compare estimate to measured tape memory for a real forward+backward
    let cfg = Config::preset("micro").unwrap();
    let (b, t) = (4, 128);
    let est = estimate_train_memory(&cfg, b, t) as f64;
    let mut m = Model::new(cfg, 1).unwrap();
    let ids = vec![65u32; b * t];
    let tg = vec![66i32; b * t];
    m.g.zero_grad();
    let l = m.loss(&ids, &tg, b, t);
    let fwd = m.g.memory_bytes() as f64;
    m.g.backward(l);
    let adam = m.param_count() as f64 * 8.0;
    let measured = fwd + adam;
    println!("estimate {:.1} MB, measured tape+adam {:.1} MB", est / 1e6, measured / 1e6);
    assert!(est > measured * 0.7 && est < measured * 3.0, "estimator off by too much");
}
