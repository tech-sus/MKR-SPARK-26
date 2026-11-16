use crate::args::Args;
use crate::pipeline::Assistant;
use mkr_core::Config;
use spark_byte_engine::{encode_with_bos, prepare_corpus, script_stats, Corpus};
use spark_inference::{generate, qio, Engine, GenConfig, Sampler};
use spark_neural::{io as nio, Config as ModelConfig, Model};
use spark_training::sys::{human_bytes, peak_rss_bytes};
use spark_training::{estimate_train_memory, load_checkpoint, plan_for_ram_gb, train, Event, TrainConfig};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

const SEED_EN: &str = include_str!("../../../datasets/seed/en.txt");
const SEED_AR: &str = include_str!("../../../datasets/seed/ar.txt");

fn e<T: std::fmt::Display>(x: T) -> String {
    x.to_string()
}

fn model_path(a: &Args, cfg: &Config) -> PathBuf {
    a.flag("model").map(PathBuf::from).unwrap_or_else(|| cfg.model_path.clone())
}

// ------------------------------------------------------------------ data
pub fn cmd_data(a: &Args, cfg: &Config) -> Result<(), String> {
    match a.pos.get(1).map(|s| s.as_str()) {
        Some("seed") => {
            let dir = PathBuf::from(a.flag("out").unwrap_or("datasets/seed"));
            std::fs::create_dir_all(&dir).map_err(e)?;
            std::fs::write(dir.join("en.txt"), SEED_EN).map_err(e)?;
            std::fs::write(dir.join("ar.txt"), SEED_AR).map_err(e)?;
            println!("Wrote the small bilingual seed corpus to {} ({} + {} bytes).", dir.display(), SEED_EN.len(), SEED_AR.len());
            println!("It is only enough to demonstrate training; a useful model needs far more text (see docs/TRAINING.md).");
            Ok(())
        }
        Some("prepare") => {
            let inputs: Vec<PathBuf> = a.pos.iter().skip(2).map(PathBuf::from).collect();
            if inputs.is_empty() {
                return Err("usage: mkr data prepare <file|dir>... [--out datasets/corpus.bin]".into());
            }
            let out = PathBuf::from(a.flag("out").unwrap_or("datasets/corpus.bin"));
            let st = prepare_corpus(&inputs, &out).map_err(e)?;
            println!("Prepared {} document(s) -> {} ({}), {} duplicate paragraph(s) removed, {} file(s) with invalid UTF-8 repaired.", st.documents, out.display(), human_bytes(st.bytes_out as u64), st.duplicate_paragraphs_removed, st.invalid_utf8_files);
            Ok(())
        }
        Some("stats") => {
            let p = a.pos.get(2).ok_or("usage: mkr data stats <file>")?;
            let bytes = std::fs::read(p).map_err(e)?;
            let text = String::from_utf8_lossy(&bytes);
            let s = script_stats(&text);
            let total = (s.arabic_chars + s.latin_chars + s.digits + s.other_chars).max(1) as f32;
            println!("{}: {} ({} bytes/char avg {:.2})", p, human_bytes(bytes.len() as u64), "UTF-8", bytes.len() as f32 / text.chars().count().max(1) as f32);
            println!("  arabic {:.1}%  latin {:.1}%  digits {:.1}%  other {:.1}%", s.arabic_chars as f32 / total * 100.0, s.latin_chars as f32 / total * 100.0, s.digits as f32 / total * 100.0, s.other_chars as f32 / total * 100.0);
            println!("  documents: {}", bytes.iter().filter(|&&b| b == 0).count().max(1));
            let _ = cfg;
            Ok(())
        }
        _ => Err("usage: mkr data <seed|prepare|stats> ...".into()),
    }
}

// ------------------------------------------------------------------ train
pub fn cmd_train(a: &Args, cfg: &Config) -> Result<(), String> {
    let out_dir = PathBuf::from(a.flag("out").unwrap_or("checkpoints"));
    let mut sources: Vec<PathBuf> = a.flag("data").map(|d| d.split(',').map(PathBuf::from).collect()).unwrap_or_default();
    sources.extend(a.pos.iter().skip(1).map(PathBuf::from));
    if sources.is_empty() {
        sources.push(PathBuf::from("datasets/seed"));
    }
    for s in &sources {
        if !s.exists() {
            return Err(format!("data path {} does not exist (create the seed corpus with `mkr data seed`)", s.display()));
        }
    }
    let corpus_path = if sources.len() == 1 && sources[0].is_file() && sources[0].extension().map_or(false, |x| x == "bin") {
        sources[0].clone()
    } else {
        let p = cfg.data_dir.join("corpus.bin");
        let st = prepare_corpus(&sources, &p).map_err(e)?;
        println!("Prepared corpus: {} document(s), {} ({} duplicate paragraphs removed)", st.documents, human_bytes(st.bytes_out as u64), st.duplicate_paragraphs_removed);
        p
    };

    let resume = a.has("resume");
    let (mut model, resume_state, plan_note);
    let plan = plan_for_ram_gb(a.num::<u32>("ram")?.unwrap_or(4));
    if resume {
        let (m, rs) = load_checkpoint(&out_dir)?;
        println!("Resuming from step {} ({} parameters, preset '{}').", rs.state.step, m.param_count(), m.cfg.name);
        model = m;
        resume_state = Some(rs);
        plan_note = String::new();
    } else {
        let preset = a.flag("preset").unwrap_or(plan.preset);
        let mcfg = ModelConfig::preset(preset).ok_or_else(|| format!("unknown preset '{preset}' (valid: {})", ModelConfig::preset_names().join(", ")))?;
        model = Model::new(mcfg, a.num::<u64>("seed")?.unwrap_or(1234))?;
        resume_state = None;
        plan_note = format!("preset '{preset}', {:.2}M parameters", model.param_count() as f64 / 1e6);
    }
    let batch = a.num::<usize>("batch")?.unwrap_or(plan.batch);
    let ctx = a.num::<usize>("ctx")?.unwrap_or(plan.ctx).min(model.cfg.max_ctx);
    let accum = a.num::<usize>("accum")?.unwrap_or(plan.accum);
    let steps = a.num::<u64>("steps")?.unwrap_or(1000);
    let est = estimate_train_memory(&model.cfg, batch, ctx);
    if !plan_note.is_empty() {
        println!("Model: {plan_note}. batch {batch} x ctx {ctx} x accum {accum}; estimated training memory {}.", human_bytes(est as u64));
    }
    let corpus = Corpus::open_streaming(&corpus_path).map_err(e)?;
    if corpus.len() < (ctx as u64 + 1) * 4 {
        return Err(format!("corpus is too small ({} bytes) for context {ctx}; add more text or lower --ctx", corpus.len()));
    }
    let (tr, va) = corpus.split(a.num::<f32>("val")?.unwrap_or(0.1).clamp(0.01, 0.5));
    if tr.len() < 20_000 {
        println!("Warning: only {} of training text. The model will memorise it; validation numbers will not reflect real generalisation.", human_bytes(tr.len()));
    }
    let tcfg = TrainConfig {
        steps,
        batch,
        ctx,
        accum,
        peak_lr: a.num::<f32>("lr")?.unwrap_or(3e-3),
        min_lr: a.num::<f32>("lr")?.unwrap_or(3e-3) / 10.0,
        warmup: a.num::<u64>("warmup")?.unwrap_or((steps / 20).max(5)),
        eval_every: a.num::<u64>("eval-every")?.unwrap_or((steps / 10).max(1)),
        ckpt_every: a.num::<u64>("ckpt-every")?.unwrap_or((steps / 5).max(1)),
        out_dir: out_dir.clone(),
        seed: a.num::<u64>("seed")?.unwrap_or(1234),
        log_every: a.num::<u64>("log-every")?.unwrap_or((steps / 100).max(1)),
        ..Default::default()
    };
    let total = tcfg.steps;
    let report = train(&mut model, &tr, &va, &tcfg, resume_state, &mut |ev| match ev {
        Event::Resumed { .. } => {}
        Event::Step { step, loss, lr, grad_norm, tokens_per_sec } => println!("step {step:>6}/{total}  loss {loss:.4}  lr {lr:.2e}  |g| {grad_norm:.2}  {tokens_per_sec:.0} tok/s"),
        Event::Eval { point, ppl, bpb, improved } => println!("  eval @ {}: train {:.4}  val {:.4}  ppl {:.2}  bits/byte {:.3}{}", point.step, point.train_loss, point.val_loss, ppl, bpb, if *improved { "  *best" } else { "" }),
        Event::Overfitting { step, best_step, best_val } => println!("  OVERFITTING detected at step {step}: validation loss has not improved since step {best_step} (best {best_val:.4}) while training loss keeps falling."),
        Event::Checkpoint { path } => println!("  checkpoint saved: {}", path.display()),
        Event::Done => {}
    })?;
    let mp = model_path(a, cfg);
    if let Some(p) = mp.parent() {
        std::fs::create_dir_all(p).map_err(e)?;
    }
    let best = out_dir.join("best.model");
    if best.exists() {
        std::fs::copy(&best, &mp).map_err(e)?;
        println!("Saved best-validation model (step {}) to {}", report.best_step, mp.display());
    } else {
        nio::save_model(&model, &mp)?;
        println!("Saved model to {}", mp.display());
    }
    nio::load_model(&mp).map_err(|x| format!("saved model failed verification: {x}"))?;
    println!("Done: {} steps in {:.1}s, train loss {:.3} -> {:.3}, best val {:.3}, peak RAM {}.{}", report.steps_done, report.elapsed_secs, report.first_train_loss, report.final_train_loss, report.best_val, human_bytes(report.peak_rss), if report.overfit_detected { " (overfitting was detected)" } else { "" });
    Ok(())
}

// ------------------------------------------------------------------ model
pub fn cmd_model(a: &Args, cfg: &Config) -> Result<(), String> {
    match a.pos.get(1).map(|s| s.as_str()) {
        Some("info") | None => {
            let p = model_path(a, cfg);
            let eng = qio::load_engine(&p)?;
            let c = &eng.cfg;
            let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            println!("MKR - SPARK 26 model: {}", p.display());
            println!("  file integrity     : OK (checksum verified), {}", human_bytes(size));
            println!("  preset/name        : {}", c.name);
            println!("  parameters (approx): {:.2}M", c.param_estimate() as f64 / 1e6);
            println!("  architecture       : SNA, d_model {}, layers {}, heads {}, window {}, ffn {}, max ctx {}", c.d_model, c.n_layers, c.n_heads, c.window, c.ffn_hidden, c.max_ctx);
            println!("  vocabulary         : {} (256 bytes + control tokens)", c.vocab);
            println!("  weights in memory  : {} ({})", human_bytes(eng.weight_bytes() as u64), if eng.is_quantized() { "int8 quantized" } else { "f32" });
            println!("  per-stream state   : {} (constant, independent of generated length)", human_bytes(eng.state_bytes() as u64));
            println!("  training memory est: {} at batch 4 x ctx {}", human_bytes(estimate_train_memory(c, 4, c.max_ctx.min(256)) as u64), c.max_ctx.min(256));
            Ok(())
        }
        Some("quantize") => {
            let input = a.pos.get(2).map(PathBuf::from).unwrap_or_else(|| cfg.model_path.clone());
            let output = a.flag("out").map(PathBuf::from).unwrap_or_else(|| input.with_extension("int8.model"));
            let m = nio::load_model(&input).map_err(|x| format!("{x} (only f32 models can be quantized)"))?;
            qio::save_quantized(&m, &output)?;
            let (a1, a2) = (std::fs::metadata(&input).map(|m| m.len()).unwrap_or(0), std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0));
            println!("Quantized {} ({}) -> {} ({}), {:.1}x smaller. Matrices are int8 with per-column scales; embeddings and norms stay f32.", input.display(), human_bytes(a1), output.display(), human_bytes(a2), a1 as f32 / a2.max(1) as f32);
            Ok(())
        }
        Some("generate") => {
            let p = model_path(a, cfg);
            let eng = qio::load_engine(&p)?;
            let prompt = a.flag("prompt").map(String::from).unwrap_or_else(|| a.text(2));
            let n = a.num::<usize>("tokens")?.unwrap_or(120);
            let mut s = Sampler::new(a.num::<f32>("temp")?.unwrap_or(cfg.temperature), 40, 0.95, 1.1, a.num::<u64>("seed")?.unwrap_or(1));
            print!("{prompt}");
            let r = generate(&eng, &encode_with_bos(&prompt), &GenConfig { max_new_tokens: n, stop_on_eos: false }, &mut s, &mut |t| {
                print!("{t}");
                let _ = std::io::stdout().flush();
            });
            println!("\n[{} tokens, {:.0} tok/s, mean log-prob {:.2}]", r.tokens.len(), r.tokens_per_sec, r.mean_logprob);
            Ok(())
        }
        _ => Err("usage: mkr model <info|quantize|generate>".into()),
    }
}

// ------------------------------------------------------------------ chat / search
pub fn cmd_chat(a: &Args, cfg: &Config) -> Result<(), String> {
    let mut asst = Assistant::new(cfg.clone(), !a.has("no-learn"), a.has("offline")).map_err(e)?;
    asst.approve_single = a.has("approve-single");
    asst.verbose = a.has("verbose");
    print!("{}", crate::BANNER);
    println!();
    println!("Model: {}", match &asst.engine {
        Some(en) => format!("{} ({:.1}M params{})", en.cfg.name, en.cfg.param_estimate() as f64 / 1e6, if en.is_quantized() { ", int8" } else { "" }),
        None => "none loaded (run `mkr train` first; factual questions still use search)".into(),
    });
    println!("Search: {}", match (&asst.search, a.has("offline")) {
        (_, true) => "offline mode".to_string(),
        (Some(s), _) => s.provider_names().join(", "),
        (None, _) => "unavailable".into(),
    });
    println!("Type a question. Commands: /remember <fact>, /exit\n");
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    loop {
        print!("You: ");
        let _ = out.flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(e)? == 0 {
            break;
        }
        let l = line.trim();
        if l == "/exit" || l == "/quit" || l == "exit" {
            break;
        }
        if let Err(x) = asst.respond(l, &mut out) {
            println!("MKR: error: {x}");
        }
        println!();
    }
    Ok(())
}

pub fn cmd_search(a: &Args, cfg: &Config) -> Result<(), String> {
    let q = a.text(1);
    if q.trim().is_empty() {
        return Err("usage: mkr search <question> [--hits] [--no-learn] [--approve-single]".into());
    }
    let mut c = cfg.clone();
    c.online = true;
    let mut asst = Assistant::new(c, !a.has("no-learn"), false).map_err(e)?;
    asst.interactive = false;
    asst.approve_single = a.has("approve-single");
    asst.verbose = a.has("verbose");
    if asst.search.is_none() {
        return Err("no usable search provider; check `mkr config` (providers, searxng_url, brave_api_key)".into());
    }
    let mut out = std::io::stdout();
    if a.has("hits") {
        let o = asst.search.as_ref().unwrap().search(&q).map_err(e)?;
        for (p, er) in &o.errors {
            println!("(provider {p} unavailable: {er})");
        }
        for (i, h) in o.hits.iter().enumerate() {
            println!("{:>2}. {} [{}] score {:.4}\n    {}\n    {}", i + 1, h.title, h.provider, h.score, h.url, h.snippet);
        }
        return Ok(());
    }
    asst.research(&q, mkr_core::text::detect_lang(&q), &mut out).map_err(e)
}

// ------------------------------------------------------------------ config / plan
pub fn cmd_config(a: &Args, path: &std::path::Path, cfg: &Config) -> Result<(), String> {
    match a.pos.get(1).map(|s| s.as_str()) {
        None | Some("list") => {
            println!("# {}{}", path.display(), if path.exists() { "" } else { " (not created yet; showing defaults)" });
            for k in Config::KEYS {
                println!("{k} = {}", cfg.get(k).unwrap_or_default());
            }
            Ok(())
        }
        Some("get") => {
            let k = a.pos.get(2).ok_or("usage: mkr config get <key>")?;
            println!("{}", cfg.get(k).ok_or_else(|| format!("unknown key '{k}'"))?);
            Ok(())
        }
        Some("set") => {
            let (k, v) = (a.pos.get(2).ok_or("usage: mkr config set <key> <value>")?, a.text(3));
            let mut c = cfg.clone();
            c.set(k, &v).map_err(e)?;
            c.save(path).map_err(e)?;
            println!("{k} = {}", c.get(k).unwrap_or_default());
            Ok(())
        }
        Some("path") => {
            println!("{}", path.display());
            Ok(())
        }
        Some("init") => {
            cfg.save(path).map_err(e)?;
            println!("Wrote {}", path.display());
            Ok(())
        }
        _ => Err("usage: mkr config [list|get|set|path|init]".into()),
    }
}

pub fn cmd_plan() -> Result<(), String> {
    println!("{:<6} {:<8} {:>9} {:>6} {:>6} {:>6} {:>14}", "RAM", "preset", "params", "batch", "ctx", "accum", "est. train mem");
    for gb in [2u32, 4, 8, 16] {
        let p = plan_for_ram_gb(gb);
        let c = ModelConfig::preset(p.preset).unwrap();
        println!("{:<6} {:<8} {:>8.1}M {:>6} {:>6} {:>6} {:>14}", format!("{gb}GB"), p.preset, c.param_estimate() as f64 / 1e6, p.batch, p.ctx, p.accum, human_bytes(p.est_train_bytes as u64));
    }
    println!("\nInference needs only the weights plus a small constant state (see `mkr model info`); int8 quantization cuts weight memory ~3.5x.");
    println!("These are estimates; run `mkr benchmark` to measure on your machine.");
    Ok(())
}

// ------------------------------------------------------------------ benchmark
pub fn cmd_benchmark(a: &Args) -> Result<(), String> {
    use spark_tensor::matmul::gemm_nn;
    let preset = a.flag("preset").unwrap_or("micro");
    let mcfg = ModelConfig::preset(preset).ok_or_else(|| format!("unknown preset '{preset}'"))?;
    let mut report = String::new();
    let mut line = |s: String| {
        println!("{s}");
        report.push_str(&s);
        report.push('\n');
    };
    line(format!("# MKR - SPARK 26 benchmark (preset '{preset}')"));
    line(format!("threads: {}   os: {}   arch: {}", rayon::current_num_threads(), std::env::consts::OS, std::env::consts::ARCH));

    // matmul throughput
    let n = 256;
    let (x, y) = (vec![0.5f32; n * n], vec![0.25f32; n * n]);
    let mut z = vec![0f32; n * n];
    gemm_nn(&x, &y, &mut z, n, n, n, false);
    let reps = 20;
    let t = Instant::now();
    for _ in 0..reps {
        gemm_nn(&x, &y, &mut z, n, n, n, false);
    }
    let gf = 2.0 * (n * n * n) as f64 * reps as f64 / t.elapsed().as_secs_f64() / 1e9;
    line(format!("matmul {n}x{n}x{n}: {gf:.2} GFLOP/s"));

    // training step
    let (batch, ctx) = (a.num::<usize>("batch")?.unwrap_or(4), a.num::<usize>("ctx")?.unwrap_or(128).min(mcfg.max_ctx));
    let mut m = Model::new(mcfg.clone(), 1)?;
    let ids: Vec<u32> = (0..batch * ctx).map(|i| (i * 31 % 251) as u32).collect();
    let tg: Vec<i32> = ids.iter().map(|&x| ((x + 1) % 251) as i32).collect();
    let mut times = Vec::new();
    let mut tape = 0usize;
    for _ in 0..4 {
        let t = Instant::now();
        m.g.zero_grad();
        let l = m.loss(&ids, &tg, batch, ctx);
        tape = tape.max(m.g.memory_bytes());
        m.g.backward(l);
        m.g.reset();
        times.push(t.elapsed().as_secs_f64());
    }
    times.remove(0); // warm-up
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    line(format!("training step (fwd+bwd, batch {batch} x ctx {ctx}): {:.0} ms median, {:.0} tokens/s; peak tensor memory {}", med * 1000.0, (batch * ctx) as f64 / med, human_bytes(tape as u64)));

    // inference
    for (name, q) in [("f32", false), ("int8", true)] {
        let eng = Engine::from_model(&m, q);
        let r = generate(&eng, &encode_with_bos("benchmark"), &GenConfig { max_new_tokens: 200, stop_on_eos: false }, &mut Sampler::new(1.0, 0, 1.0, 1.0, 3), &mut |_| {});
        line(format!("inference {name}: {:.0} tokens/s, weights {}, state {}", r.tokens_per_sec, human_bytes(eng.weight_bytes() as u64), human_bytes(eng.state_bytes() as u64)));
    }
    line(format!("parameters: {:.2}M   peak process RAM: {}", m.param_count() as f64 / 1e6, human_bytes(peak_rss_bytes())));
    if a.has("save") {
        std::fs::create_dir_all("benchmarks").map_err(e)?;
        std::fs::write("benchmarks/latest.md", report).map_err(e)?;
        println!("saved benchmarks/latest.md");
    }
    Ok(())
}
