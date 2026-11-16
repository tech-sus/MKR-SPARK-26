use spark_inference::*;
use spark_neural::{Config, Model};
use spark_tensor::Rng;

fn rand_ids(n: usize, seed: u64) -> Vec<u32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.below(256) as u32).collect()
}

#[test]
fn incremental_engine_matches_training_graph() {
    // t=40 exceeds the window (16) so the ring buffer wraps; also test recurrence-only (window=0)
    for (name, cfg) in [("nano", Config::preset("nano").unwrap()), ("no-attn", Config::new("na", 32, 2, 2, 0, 64))] {
        let mut m = Model::new(cfg, 5).unwrap();
        let t = 40;
        let ids = rand_ids(t, 3);
        let l = m.forward(&ids, 1, t);
        let graph_logits = m.g.value(l).to_vec();
        let eng = Engine::from_model(&m, false);
        let mut st = eng.new_state();
        let v = m.cfg.vocab;
        let mut worst = 0.0f32;
        for (i, &tok) in ids.iter().enumerate() {
            let lg = eng.step(&mut st, tok);
            for j in 0..v {
                worst = worst.max((lg[j] - graph_logits[i * v + j]).abs());
            }
        }
        println!("{name}: max |engine - graph| logit diff = {worst:e}");
        assert!(worst < 2e-3, "{name}: inference diverges from training graph: {worst}");
    }
}

#[test]
fn int8_quantization_is_close_and_smaller() {
    let m = Model::new(Config::preset("micro").unwrap(), 6).unwrap();
    let f = Engine::from_model(&m, false);
    let q = Engine::from_model(&m, true);
    assert!(q.is_quantized() && !f.is_quantized());
    assert!((q.weight_bytes() as f64) < f.weight_bytes() as f64 * 0.4);
    let ids = rand_ids(24, 9);
    let (mut sf, mut sq) = (f.new_state(), q.new_state());
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for &t in &ids {
        let a = f.step(&mut sf, t);
        let b = q.step(&mut sq, t);
        for (x, y) in a.iter().zip(&b) {
            num += ((x - y) as f64).powi(2);
            den += (*x as f64).powi(2);
        }
    }
    let rel = (num / den).sqrt();
    println!("int8 relative logit error {rel:.4}, size {:.2}x smaller", f.weight_bytes() as f64 / q.weight_bytes() as f64);
    assert!(rel < 0.05, "quantization error too large: {rel}");
}

#[test]
fn quantized_file_roundtrip_and_corruption() {
    let m = Model::new(Config::preset("nano").unwrap(), 7).unwrap();
    let dir = std::env::temp_dir().join(format!("mkr_q_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (pq, pf) = (dir.join("q.model"), dir.join("f.model"));
    qio::save_quantized(&m, &pq).unwrap();
    spark_neural::io::save_model(&m, &pf).unwrap();
    let eq = qio::load_engine(&pq).unwrap();
    let ef = qio::load_engine(&pf).unwrap();
    assert!(eq.is_quantized() && !ef.is_quantized());
    let direct = Engine::from_model(&m, true);
    let (mut s1, mut s2) = (eq.new_state(), direct.new_state());
    assert_eq!(eq.step(&mut s1, 65), direct.step(&mut s2, 65));
    let mut bytes = std::fs::read(&pq).unwrap();
    let n = bytes.len();
    bytes[n / 3] ^= 0x20;
    std::fs::write(&pq, bytes).unwrap();
    assert!(qio::load_engine(&pq).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn state_is_constant_size_and_generation_is_deterministic() {
    let m = Model::new(Config::preset("nano").unwrap(), 8).unwrap();
    let e = Engine::from_model(&m, false);
    let prompt = spark_byte_engine::encode_with_bos("hello");
    let gc = GenConfig { max_new_tokens: 60, stop_on_eos: false };
    let r1 = generate(&e, &prompt, &gc, &mut Sampler::new(0.8, 40, 0.95, 1.1, 42), &mut |_| {});
    let r2 = generate(&e, &prompt, &gc, &mut Sampler::new(0.8, 40, 0.95, 1.1, 42), &mut |_| {});
    assert_eq!(r1.tokens, r2.tokens);
    assert_eq!(r1.tokens.len(), 60);
    assert!(r1.tokens.iter().all(|&t| t < 256), "control tokens must be masked");
    let r3 = generate(&e, &prompt, &gc, &mut Sampler::new(0.8, 40, 0.95, 1.1, 43), &mut |_| {});
    assert_ne!(r1.tokens, r3.tokens);
    // untrained model: confidence should be near-uniform (entropy close to ln 257)
    assert!(r1.mean_entropy > 4.5, "entropy {}", r1.mean_entropy);
    println!("state size {} bytes, {:.0} tok/s", e.state_bytes(), r1.tokens_per_sec);
}

#[test]
fn sampler_properties() {
    let mut logits = vec![0.0f32; 264];
    logits[100] = 10.0;
    logits[101] = 9.0;
    logits[260] = 100.0; // control token must never win
    assert_eq!(Sampler::greedy().sample(&logits, &[]), 100);
    let mut s = Sampler::new(1.0, 1, 1.0, 1.0, 1);
    for _ in 0..20 {
        assert_eq!(s.sample(&logits, &[]), 100); // top-k=1
    }
    let lp = Sampler::log_probs(&logits);
    let total: f32 = lp.iter().filter(|v| v.is_finite()).map(|v| v.exp()).sum();
    assert!((total - 1.0).abs() < 1e-4);
}
