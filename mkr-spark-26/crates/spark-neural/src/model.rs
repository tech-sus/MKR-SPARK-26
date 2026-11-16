use crate::config::Config;
use spark_tensor::{Graph, Rng, Var};

/// Parameter handles for one SPARK block.
#[derive(Clone, Debug)]
pub struct BlockVars {
    pub n1: Var,
    pub wa: Var,
    pub ba: Var,
    pub wv: Var,
    pub wg: Var,
    pub wso: Var,
    pub n2: Option<Var>,
    pub wq: Option<Var>,
    pub wk: Option<Var>,
    pub wva: Option<Var>,
    pub wao: Option<Var>,
    pub n3: Var,
    pub w1: Var,
    pub w3: Var,
    pub w2: Var,
}

pub struct Model {
    pub cfg: Config,
    pub g: Graph,
    pub emb: Var,
    pub blocks: Vec<BlockVars>,
    pub nf: Var,
    pub head: Var,
}

fn normal_vec(rng: &mut Rng, n: usize, std: f32) -> Vec<f32> {
    (0..n).map(|_| rng.normal() * std).collect()
}

impl Model {
    /// Create a randomly initialised model (training from scratch).
    pub fn new(cfg: Config, seed: u64) -> Result<Model, String> {
        cfg.validate()?;
        let mut rng = Rng::new(seed);
        let mut g = Graph::new();
        let d = cfg.d_model;
        let std = 0.02f32;
        let out_std = std / (2.0 * cfg.n_layers as f32).sqrt();
        let ones = |n: usize| vec![1.0f32; n];
        let emb = g.param(cfg.vocab, d, normal_vec(&mut rng, cfg.vocab * d, std));
        let mut blocks = Vec::new();
        for _ in 0..cfg.n_layers {
            let n1 = g.param(1, d, ones(d));
            let wa = g.param(d, d, normal_vec(&mut rng, d * d, std * 0.5));
            // Multi-timescale decay init: (1-a) log-uniform in [1e-3, 0.5] => memory from ~2 to ~1000 tokens.
            let ba: Vec<f32> = (0..d)
                .map(|j| {
                    let f = j as f32 / (d.max(2) - 1) as f32;
                    let one_minus_a = (1e-3f32.ln() * (1.0 - f) + 0.5f32.ln() * f).exp();
                    let a = 1.0 - one_minus_a;
                    (a / (1.0 - a)).ln()
                })
                .collect();
            let ba = g.param(1, d, ba);
            let wv = g.param(d, d, normal_vec(&mut rng, d * d, std));
            let wg = g.param(d, d, normal_vec(&mut rng, d * d, std));
            let wso = g.param(d, d, normal_vec(&mut rng, d * d, out_std));
            let (n2, wq, wk, wva, wao) = if cfg.window > 0 {
                (
                    Some(g.param(1, d, ones(d))),
                    Some(g.param(d, d, normal_vec(&mut rng, d * d, std))),
                    Some(g.param(d, d, normal_vec(&mut rng, d * d, std))),
                    Some(g.param(d, d, normal_vec(&mut rng, d * d, std))),
                    Some(g.param(d, d, normal_vec(&mut rng, d * d, out_std))),
                )
            } else {
                (None, None, None, None, None)
            };
            let h = cfg.ffn_hidden;
            let n3 = g.param(1, d, ones(d));
            let w1 = g.param(d, h, normal_vec(&mut rng, d * h, std));
            let w3 = g.param(d, h, normal_vec(&mut rng, d * h, std));
            let w2 = g.param(h, d, normal_vec(&mut rng, h * d, out_std));
            blocks.push(BlockVars { n1, wa, ba, wv, wg, wso, n2, wq, wk, wva, wao, n3, w1, w3, w2 });
        }
        let nf = g.param(1, d, ones(d));
        let head = g.param(d, cfg.vocab, normal_vec(&mut rng, d * cfg.vocab, std));
        Ok(Model { cfg, g, emb, blocks, nf, head })
    }

    pub fn param_count(&self) -> usize {
        self.g.param_count()
    }

    /// Forward pass over `bsz` sequences of length `t`; returns logits [bsz*t, vocab].
    pub fn forward(&mut self, ids: &[u32], bsz: usize, t: usize) -> Var {
        assert_eq!(ids.len(), bsz * t);
        assert!(t <= self.cfg.max_ctx, "sequence longer than max_ctx");
        let cfg = self.cfg.clone();
        let d = cfg.d_model;
        let g = &mut self.g;
        let mut x = g.embedding(self.emb, ids);
        let ones = g.input(bsz * t, d, vec![1.0; bsz * t * d]);
        for b in self.blocks.clone() {
            // 1. SPARK-Scan
            let h = g.rms_norm(x, b.n1, cfg.norm_eps);
            let a_pre = g.matmul(h, b.wa);
            let a_pre = g.add_bias(a_pre, b.ba);
            let a = g.sigmoid(a_pre);
            let v = g.matmul(h, b.wv);
            let neg_a = g.scale(a, -1.0);
            let one_minus_a = g.add(neg_a, ones);
            let bin = g.mul(one_minus_a, v);
            let s = g.linear_scan(a, bin, bsz, t);
            let gate_pre = g.matmul(h, b.wg);
            let gate = g.silu(gate_pre);
            let mixed = g.mul(s, gate);
            let mixed = g.matmul(mixed, b.wso);
            x = g.add(x, mixed);
            // 2. sliding-window attention
            if cfg.window > 0 {
                let h = g.rms_norm(x, b.n2.unwrap(), cfg.norm_eps);
                let q = g.matmul(h, b.wq.unwrap());
                let k = g.matmul(h, b.wk.unwrap());
                let v = g.matmul(h, b.wva.unwrap());
                let q = g.rope(q, t, cfg.n_heads, cfg.rope_base);
                let k = g.rope(k, t, cfg.n_heads, cfg.rope_base);
                let at = g.attention(q, k, v, bsz, t, cfg.n_heads, cfg.window);
                let at = g.matmul(at, b.wao.unwrap());
                x = g.add(x, at);
            }
            // 3. SwiGLU
            let h = g.rms_norm(x, b.n3, cfg.norm_eps);
            let u1 = g.matmul(h, b.w1);
            let u1 = g.silu(u1);
            let u3 = g.matmul(h, b.w3);
            let u = g.mul(u1, u3);
            let u = g.matmul(u, b.w2);
            x = g.add(x, u);
        }
        let x = g.rms_norm(x, self.nf, cfg.norm_eps);
        g.matmul(x, self.head)
    }

    /// Mean next-byte cross-entropy (nats) for a batch; targets < 0 are masked.
    pub fn loss(&mut self, ids: &[u32], targets: &[i32], bsz: usize, t: usize) -> Var {
        let logits = self.forward(ids, bsz, t);
        self.g.cross_entropy(logits, targets)
    }
}
