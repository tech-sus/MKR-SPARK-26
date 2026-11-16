use spark_neural::{io, Config, Model};
use spark_tensor::Rng;

fn rand_ids(n: usize, seed: u64) -> Vec<u32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.below(256) as u32).collect()
}

#[test]
fn all_presets_validate_and_estimates_are_close() {
    for name in Config::preset_names() {
        let c = Config::preset(name).unwrap();
        c.validate().unwrap();
        println!("{name}: ~{:.2}M params", c.param_estimate() as f64 / 1e6);
    }
    // estimate must match reality for a built model
    let c = Config::preset("nano").unwrap();
    let est = c.param_estimate();
    let m = Model::new(c, 1).unwrap();
    let real = m.param_count();
    assert!((est as f64 - real as f64).abs() / (real as f64) < 0.05, "estimate {est} vs real {real}");
}

#[test]
fn initial_loss_is_near_uniform_entropy() {
    let mut m = Model::new(Config::preset("nano").unwrap(), 7).unwrap();
    let (b, t) = (2, 32);
    let ids = rand_ids(b * t, 1);
    let tg: Vec<i32> = rand_ids(b * t, 2).iter().map(|&x| x as i32).collect();
    let l = m.loss(&ids, &tg, b, t);
    let loss = m.g.scalar(l);
    let uniform = (m.cfg.vocab as f32).ln();
    assert!((loss - uniform).abs() < 0.3, "initial loss {loss} vs ln(V) {uniform}");
}

#[test]
fn model_is_causal() {
    // changing a future byte must not change earlier logits
    let mut m = Model::new(Config::preset("nano").unwrap(), 3).unwrap();
    let t = 24;
    let mut ids = rand_ids(t, 5);
    let l1 = m.forward(&ids, 1, t);
    let a: Vec<f32> = m.g.value(l1)[..10 * m.cfg.vocab].to_vec();
    m.g.reset();
    ids[15] = (ids[15] + 7) % 256;
    let l2 = m.forward(&ids, 1, t);
    let b: Vec<f32> = m.g.value(l2)[..10 * m.cfg.vocab].to_vec();
    assert_eq!(a, b, "logits for positions < 15 changed when position 15 changed");
}

#[test]
fn batch_independence() {
    let mut m = Model::new(Config::preset("nano").unwrap(), 4).unwrap();
    let t = 16;
    let s1 = rand_ids(t, 10);
    let s2 = rand_ids(t, 11);
    let mut both = s1.clone();
    both.extend(&s2);
    let lb = m.forward(&both, 2, t);
    let batched: Vec<f32> = m.g.value(lb).to_vec();
    m.g.reset();
    let l1 = m.forward(&s1, 1, t);
    let single: Vec<f32> = m.g.value(l1).to_vec();
    let v = m.cfg.vocab;
    for i in 0..t * v {
        assert!((batched[i] - single[i]).abs() < 1e-4);
    }
}

#[test]
fn model_gradients_flow_to_every_parameter() {
    let mut m = Model::new(Config::preset("nano").unwrap(), 9).unwrap();
    let (b, t) = (2, 16);
    let ids = rand_ids(b * t, 1);
    let tg: Vec<i32> = rand_ids(b * t, 2).iter().map(|&x| x as i32).collect();
    m.g.zero_grad();
    let l = m.loss(&ids, &tg, b, t);
    m.g.backward(l);
    for (i, p) in m.g.params().iter().enumerate() {
        let norm: f32 = p.grad.iter().map(|g| g * g).sum();
        assert!(norm.is_finite(), "param {i} has non-finite gradient");
        assert!(norm > 0.0, "param {i} ({}x{}) received no gradient", p.rows, p.cols);
    }
}

#[test]
fn model_end_to_end_gradient_check() {
    // numerical check on a handful of entries of several tensors through the whole network
    let cfg = Config::new("gc", 16, 2, 2, 4, 32);
    let mut m = Model::new(cfg, 5).unwrap();
    let (b, t) = (2, 8);
    let ids = rand_ids(b * t, 21);
    let tg: Vec<i32> = rand_ids(b * t, 22).iter().map(|&x| x as i32).collect();
    m.g.zero_grad();
    let l = m.loss(&ids, &tg, b, t);
    m.g.backward(l);
    m.g.reset();
    let n_params = m.g.params().len();
    let mut rng = Rng::new(99);
    let eps = 3e-3f32;
    let mut checked = 0;
    for pi in 0..n_params {
        let len = m.g.params()[pi].data.len();
        for _ in 0..3 {
            let idx = rng.below(len);
            let ana = m.g.params()[pi].grad[idx];
            let orig = m.g.params()[pi].data[idx];
            m.g.params_mut()[pi].data[idx] = orig + eps;
            let lp = { let l = m.loss(&ids, &tg, b, t); let v = m.g.scalar(l); m.g.reset(); v };
            m.g.params_mut()[pi].data[idx] = orig - eps;
            let lm = { let l = m.loss(&ids, &tg, b, t); let v = m.g.scalar(l); m.g.reset(); v };
            m.g.params_mut()[pi].data[idx] = orig;
            let num = (lp - lm) / (2.0 * eps);
            let err = (num - ana).abs() / (1e-3 + num.abs().max(ana.abs()));
            assert!(err < 0.08 || (num - ana).abs() < 2e-4, "param {pi} idx {idx}: numeric {num} vs analytic {ana}");
            checked += 1;
        }
    }
    assert!(checked > 40);
}

#[test]
fn serialization_roundtrip_and_corruption_detection() {
    let m = Model::new(Config::preset("nano").unwrap(), 12).unwrap();
    let dir = std::env::temp_dir().join(format!("mkr_neural_{}", std::process::id()));
    let path = dir.join("t.model");
    io::save_model(&m, &path).unwrap();
    let mut m2 = io::load_model(&path).unwrap();
    assert_eq!(m2.cfg, m.cfg);
    for (a, b) in m.g.params().iter().zip(m2.g.params()) {
        assert_eq!(a.data, b.data);
    }
    let ids = rand_ids(16, 1);
    let mut m1 = m;
    let l1 = m1.forward(&ids, 1, 16);
    let l2 = m2.forward(&ids, 1, 16);
    assert_eq!(m1.g.value(l1), m2.g.value(l2));

    let mut bytes = std::fs::read(&path).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x55;
    assert!(io::unwrap(&bytes).unwrap_err().contains("checksum"));
    assert!(io::unwrap(&bytes[..bytes.len() - 7]).is_err());
    assert!(io::unwrap(b"garbage garbage garbage garbage").is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
