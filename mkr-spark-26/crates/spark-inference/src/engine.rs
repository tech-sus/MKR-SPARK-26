use rayon::prelude::*;
use spark_neural::{Config, Model};
use spark_tensor::{matmul::{dot, gemm_nn}, rope::rope_apply};

const CH: usize = 1024;
const PAR_MIN: usize = 1 << 17;

/// Weight matrix stored [in, out] row-major, f32 or int8 with a per-output-column scale.
pub enum Mat {
    F32 { d: Vec<f32>, rows: usize, cols: usize },
    I8 { q: Vec<i8>, scale: Vec<f32>, rows: usize, cols: usize },
}

impl Mat {
    pub fn rows(&self) -> usize {
        match self { Mat::F32 { rows, .. } | Mat::I8 { rows, .. } => *rows }
    }
    pub fn cols(&self) -> usize {
        match self { Mat::F32 { cols, .. } | Mat::I8 { cols, .. } => *cols }
    }
    pub fn bytes(&self) -> usize {
        match self {
            Mat::F32 { d, .. } => d.len() * 4,
            Mat::I8 { q, scale, .. } => q.len() + scale.len() * 4,
        }
    }
    pub fn quantize(d: &[f32], rows: usize, cols: usize) -> Mat {
        let mut scale = vec![0f32; cols];
        for r in 0..rows {
            for c in 0..cols {
                scale[c] = scale[c].max(d[r * cols + c].abs());
            }
        }
        for s in scale.iter_mut() {
            *s = if *s > 0.0 { *s / 127.0 } else { 1.0 };
        }
        let mut q = vec![0i8; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                q[r * cols + c] = (d[r * cols + c] / scale[c]).round().clamp(-127.0, 127.0) as i8;
            }
        }
        Mat::I8 { q, scale, rows, cols }
    }
    /// out = x @ W
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        match self {
            Mat::F32 { d, rows, cols } => gemm_nn(x, d, out, 1, *rows, *cols, false),
            Mat::I8 { q, scale, rows, cols } => {
                let (rows, cols) = (*rows, *cols);
                let body = |(ci, blk): (usize, &mut [f32])| {
                    let j0 = ci * CH;
                    blk.fill(0.0);
                    for p in 0..rows {
                        let xv = x[p];
                        let row = &q[p * cols + j0..p * cols + j0 + blk.len()];
                        for (o, &w) in blk.iter_mut().zip(row) {
                            *o += xv * w as f32;
                        }
                    }
                    for (o, s) in blk.iter_mut().zip(&scale[j0..]) {
                        *o *= *s;
                    }
                };
                if rows * cols >= PAR_MIN {
                    out.par_chunks_mut(CH).enumerate().for_each(body);
                } else {
                    out.chunks_mut(CH).enumerate().for_each(body);
                }
            }
        }
    }
}

pub struct AttnW {
    pub n: Vec<f32>,
    pub wq: Mat,
    pub wk: Mat,
    pub wv: Mat,
    pub wo: Mat,
}

pub struct LayerW {
    pub n1: Vec<f32>,
    pub wa: Mat,
    pub ba: Vec<f32>,
    pub wv: Mat,
    pub wg: Mat,
    pub wso: Mat,
    pub attn: Option<AttnW>,
    pub n3: Vec<f32>,
    pub w1: Mat,
    pub w3: Mat,
    pub w2: Mat,
}

pub struct Engine {
    pub cfg: Config,
    pub emb: Vec<f32>,
    pub layers: Vec<LayerW>,
    pub nf: Vec<f32>,
    pub head: Mat,
}

pub struct LayerState {
    h: Vec<f32>,
    kc: Vec<f32>,
    vc: Vec<f32>,
}

pub struct State {
    pub pos: usize,
    layers: Vec<LayerState>,
}

/// A raw parameter tensor as stored on disk / extracted from a Model.
pub enum Raw {
    F32 { rows: usize, cols: usize, d: Vec<f32> },
    I8 { rows: usize, cols: usize, q: Vec<i8>, scale: Vec<f32> },
}

impl Raw {
    fn into_vec(self) -> Result<Vec<f32>, String> {
        match self {
            Raw::F32 { d, .. } => Ok(d),
            _ => Err("expected f32 vector tensor".into()),
        }
    }
    fn into_mat(self) -> Mat {
        match self {
            Raw::F32 { rows, cols, d } => Mat::F32 { d, rows, cols },
            Raw::I8 { rows, cols, q, scale } => Mat::I8 { q, scale, rows, cols },
        }
    }
}

fn rms_norm(x: &[f32], g: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * g[i];
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl Engine {
    /// Extract raw tensors from a trained model. Matrices (except the embedding) are int8 when `quantize`.
    pub fn raw_from_model(m: &Model, quantize: bool) -> Vec<Raw> {
        m.g.params()
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let is_matrix = p.rows > 1 && p.cols > 1 && i != 0; // index 0 = embedding
                if quantize && is_matrix {
                    match Mat::quantize(&p.data, p.rows, p.cols) {
                        Mat::I8 { q, scale, rows, cols } => Raw::I8 { rows, cols, q, scale },
                        _ => unreachable!(),
                    }
                } else {
                    Raw::F32 { rows: p.rows, cols: p.cols, d: p.data.clone() }
                }
            })
            .collect()
    }

    pub fn from_model(m: &Model, quantize: bool) -> Engine {
        Self::assemble(m.cfg.clone(), Self::raw_from_model(m, quantize)).expect("model layout is self-consistent")
    }

    /// Build an engine from tensors in `Model::new` creation order.
    pub fn assemble(cfg: Config, raws: Vec<Raw>) -> Result<Engine, String> {
        let expected = 1 + cfg.n_layers * (if cfg.window > 0 { 15 } else { 10 }) + 2;
        if raws.len() != expected {
            return Err(format!("expected {expected} tensors, found {}", raws.len()));
        }
        let mut it = raws.into_iter();
        let mut next = || it.next().ok_or_else(|| "tensor list exhausted".to_string());
        let emb = next()?.into_vec()?;
        let mut layers = Vec::new();
        for _ in 0..cfg.n_layers {
            let n1 = next()?.into_vec()?;
            let wa = next()?.into_mat();
            let ba = next()?.into_vec()?;
            let wv = next()?.into_mat();
            let wg = next()?.into_mat();
            let wso = next()?.into_mat();
            let attn = if cfg.window > 0 {
                Some(AttnW { n: next()?.into_vec()?, wq: next()?.into_mat(), wk: next()?.into_mat(), wv: next()?.into_mat(), wo: next()?.into_mat() })
            } else {
                None
            };
            let n3 = next()?.into_vec()?;
            let w1 = next()?.into_mat();
            let w3 = next()?.into_mat();
            let w2 = next()?.into_mat();
            layers.push(LayerW { n1, wa, ba, wv, wg, wso, attn, n3, w1, w3, w2 });
        }
        let nf = next()?.into_vec()?;
        let head = next()?.into_mat();
        if emb.len() != cfg.vocab * cfg.d_model || head.cols() != cfg.vocab {
            return Err("embedding/head shape mismatch".into());
        }
        Ok(Engine { cfg, emb, layers, nf, head })
    }

    pub fn weight_bytes(&self) -> usize {
        let mut n = self.emb.len() * 4 + self.nf.len() * 4 + self.head.bytes();
        for l in &self.layers {
            n += (l.n1.len() + l.ba.len() + l.n3.len()) * 4 + l.wa.bytes() + l.wv.bytes() + l.wg.bytes() + l.wso.bytes() + l.w1.bytes() + l.w3.bytes() + l.w2.bytes();
            if let Some(a) = &l.attn {
                n += a.n.len() * 4 + a.wq.bytes() + a.wk.bytes() + a.wv.bytes() + a.wo.bytes();
            }
        }
        n
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self.head, Mat::I8 { .. })
    }

    pub fn new_state(&self) -> State {
        let d = self.cfg.d_model;
        let w = self.cfg.window.max(1);
        State {
            pos: 0,
            layers: (0..self.cfg.n_layers).map(|_| LayerState { h: vec![0.0; d], kc: vec![0.0; w * d], vc: vec![0.0; w * d] }).collect(),
        }
    }

    pub fn state_bytes(&self) -> usize {
        let d = self.cfg.d_model;
        self.cfg.n_layers * (d + 2 * self.cfg.window.max(1) * d) * 4
    }

    /// Consume one token at position `state.pos`; return next-byte logits.
    pub fn step(&self, st: &mut State, token: u32) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let hd = if cfg.window > 0 { d / cfg.n_heads } else { 0 };
        let pos = st.pos;
        let mut x = self.emb[token as usize * d..(token as usize + 1) * d].to_vec();
        let (mut h, mut tmp_a, mut tmp_v, mut tmp_g, mut mix) = (vec![0f32; d], vec![0f32; d], vec![0f32; d], vec![0f32; d], vec![0f32; d]);
        for (li, l) in self.layers.iter().enumerate() {
            let ls = &mut st.layers[li];
            // SPARK-Scan
            rms_norm(&x, &l.n1, cfg.norm_eps, &mut h);
            l.wa.matvec(&h, &mut tmp_a);
            l.wv.matvec(&h, &mut tmp_v);
            l.wg.matvec(&h, &mut tmp_g);
            for j in 0..d {
                let a = 1.0 / (1.0 + (-(tmp_a[j] + l.ba[j])).exp());
                ls.h[j] = a * ls.h[j] + (1.0 - a) * tmp_v[j];
                mix[j] = ls.h[j] * silu(tmp_g[j]);
            }
            l.wso.matvec(&mix, &mut tmp_a);
            for j in 0..d {
                x[j] += tmp_a[j];
            }
            // sliding-window attention
            if let Some(a) = &l.attn {
                let w = cfg.window;
                rms_norm(&x, &a.n, cfg.norm_eps, &mut h);
                let (mut q, mut k, mut v) = (vec![0f32; d], vec![0f32; d], vec![0f32; d]);
                a.wq.matvec(&h, &mut q);
                a.wk.matvec(&h, &mut k);
                a.wv.matvec(&h, &mut v);
                rope_apply(&mut q, pos, cfg.n_heads, hd, cfg.rope_base, false);
                rope_apply(&mut k, pos, cfg.n_heads, hd, cfg.rope_base, false);
                let slot = pos % w;
                ls.kc[slot * d..(slot + 1) * d].copy_from_slice(&k);
                ls.vc[slot * d..(slot + 1) * d].copy_from_slice(&v);
                let lo = (pos + 1).saturating_sub(w);
                let len = pos + 1 - lo;
                let scale = 1.0 / (hd as f32).sqrt();
                let mut o = vec![0f32; d];
                let mut p = vec![0f32; len];
                for hh in 0..cfg.n_heads {
                    let off = hh * hd;
                    let qh = &q[off..off + hd];
                    let mut mx = f32::NEG_INFINITY;
                    for (j, s) in (lo..=pos).enumerate() {
                        let sl = s % w;
                        p[j] = dot(qh, &ls.kc[sl * d + off..sl * d + off + hd]) * scale;
                        mx = mx.max(p[j]);
                    }
                    let mut sum = 0.0;
                    for x in p.iter_mut() {
                        *x = (*x - mx).exp();
                        sum += *x;
                    }
                    for (j, s) in (lo..=pos).enumerate() {
                        let sl = s % w;
                        let pj = p[j] / sum;
                        for e in 0..hd {
                            o[off + e] += pj * ls.vc[sl * d + off + e];
                        }
                    }
                }
                a.wo.matvec(&o, &mut tmp_a);
                for j in 0..d {
                    x[j] += tmp_a[j];
                }
            }
            // SwiGLU
            rms_norm(&x, &l.n3, cfg.norm_eps, &mut h);
            let hid = cfg.ffn_hidden;
            let (mut u1, mut u3) = (vec![0f32; hid], vec![0f32; hid]);
            l.w1.matvec(&h, &mut u1);
            l.w3.matvec(&h, &mut u3);
            for j in 0..hid {
                u1[j] = silu(u1[j]) * u3[j];
            }
            l.w2.matvec(&u1, &mut tmp_a);
            for j in 0..d {
                x[j] += tmp_a[j];
            }
        }
        rms_norm(&x, &self.nf, cfg.norm_eps, &mut h);
        let mut logits = vec![0f32; cfg.vocab];
        self.head.matvec(&h, &mut logits);
        st.pos += 1;
        logits
    }
}
