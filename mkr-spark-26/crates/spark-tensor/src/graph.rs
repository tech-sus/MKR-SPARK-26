//! Tape-based reverse-mode autograd over dense f32 matrices.
//!
//! Persistent tensors (parameters) occupy the first `persistent` slots. Activations are
//! appended after them and dropped by `reset()`. Every op appends exactly one tensor, so the
//! op list is parallel to the tensor list and the backward pass can split the tensor slice at
//! the output index to borrow inputs mutably while reading the output gradient.
use crate::matmul::{dot, gemm_nn, gemm_nt, gemm_tn};
use crate::rope::rope_apply;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(pub usize);

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub requires_grad: bool,
}

enum Op {
    Leaf,
    MatMul(usize, usize),
    Add(usize, usize),
    AddBias(usize, usize),
    Mul(usize, usize),
    Scale(usize, f32),
    Silu(usize),
    Sigmoid(usize),
    Tanh(usize),
    Relu(usize),
    RmsNorm { x: usize, g: usize, eps: f32 },
    LayerNorm { x: usize, g: usize, b: usize, eps: f32 },
    Embedding { table: usize, ids: Vec<u32> },
    CrossEntropy { logits: usize, targets: Vec<i32> },
    LinearScan { a: usize, b: usize, bsz: usize, t: usize },
    Attention { q: usize, k: usize, v: usize, bsz: usize, t: usize, h: usize, w: usize, probs: Vec<f32> },
    Rope { x: usize, t: usize, h: usize, base: f32 },
    WeightedSum { x: usize, w: Vec<f32> },
}

pub struct Graph {
    tensors: Vec<Tensor>,
    ops: Vec<Op>,
    persistent: usize,
    grad_enabled: bool,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}


fn op_inputs(op: &Op) -> Vec<usize> {
    match op {
        Op::Leaf => vec![],
        Op::MatMul(a, b) | Op::Add(a, b) | Op::AddBias(a, b) | Op::Mul(a, b) => vec![*a, *b],
        Op::Scale(a, _) | Op::Silu(a) | Op::Sigmoid(a) | Op::Tanh(a) | Op::Relu(a) => vec![*a],
        Op::RmsNorm { x, g, .. } => vec![*x, *g],
        Op::LayerNorm { x, g, b, .. } => vec![*x, *g, *b],
        Op::Embedding { table, .. } => vec![*table],
        Op::CrossEntropy { logits, .. } => vec![*logits],
        Op::LinearScan { a, b, .. } => vec![*a, *b],
        Op::Attention { q, k, v, .. } => vec![*q, *k, *v],
        Op::Rope { x, .. } => vec![*x],
        Op::WeightedSum { x, .. } => vec![*x],
    }
}

fn pair(v: &mut [Tensor], w: usize, r: usize) -> (&mut Tensor, &Tensor) {
    assert!(w != r, "autograd: op input aliasing is not supported here");
    if w < r {
        let (l, rt) = v.split_at_mut(r);
        (&mut l[w], &rt[0])
    } else {
        let (l, rt) = v.split_at_mut(w);
        (&mut rt[0], &l[r])
    }
}

#[inline]
fn add_into(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph { tensors: Vec::new(), ops: Vec::new(), persistent: 0, grad_enabled: true }
    }

    // ---------- bookkeeping ----------
    pub fn set_grad_enabled(&mut self, on: bool) {
        self.grad_enabled = on;
    }
    /// Register a persistent trainable parameter. Must be called before any activation is recorded.
    pub fn param(&mut self, rows: usize, cols: usize, data: Vec<f32>) -> Var {
        assert_eq!(data.len(), rows * cols);
        assert_eq!(self.tensors.len(), self.persistent, "params must be created before activations");
        self.tensors.push(Tensor { grad: vec![0.0; data.len()], data, rows, cols, requires_grad: true });
        self.ops.push(Op::Leaf);
        self.persistent = self.tensors.len();
        Var(self.tensors.len() - 1)
    }
    /// Drop all activations; keep parameters and their gradients.
    pub fn reset(&mut self) {
        self.tensors.truncate(self.persistent);
        self.ops.truncate(self.persistent);
    }
    pub fn zero_grad(&mut self) {
        for t in &mut self.tensors[..self.persistent] {
            t.grad.fill(0.0);
        }
    }
    pub fn params(&self) -> &[Tensor] {
        &self.tensors[..self.persistent]
    }
    pub fn params_mut(&mut self) -> &mut [Tensor] {
        &mut self.tensors[..self.persistent]
    }
    pub fn param_count(&self) -> usize {
        self.params().iter().map(|t| t.data.len()).sum()
    }
    pub fn value(&self, v: Var) -> &[f32] {
        &self.tensors[v.0].data
    }
    pub fn value_mut(&mut self, v: Var) -> &mut [f32] {
        &mut self.tensors[v.0].data
    }
    pub fn grad(&self, v: Var) -> &[f32] {
        &self.tensors[v.0].grad
    }
    pub fn shape(&self, v: Var) -> (usize, usize) {
        (self.tensors[v.0].rows, self.tensors[v.0].cols)
    }
    pub fn scalar(&self, v: Var) -> f32 {
        self.tensors[v.0].data[0]
    }
    /// Approximate live memory of tensors in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.tensors.iter().map(|t| (t.data.capacity() + t.grad.capacity()) * 4).sum()
    }

    fn rg(&self, ids: &[usize]) -> bool {
        self.grad_enabled && ids.iter().any(|&i| self.tensors[i].requires_grad)
    }
    fn push(&mut self, rows: usize, cols: usize, data: Vec<f32>, rg: bool, op: Op) -> Var {
        debug_assert_eq!(data.len(), rows * cols);
        self.tensors.push(Tensor { data, grad: Vec::new(), rows, cols, requires_grad: rg });
        self.ops.push(if rg { op } else { Op::Leaf });
        Var(self.tensors.len() - 1)
    }
    /// A constant (non-differentiable) input tensor.
    pub fn input(&mut self, rows: usize, cols: usize, data: Vec<f32>) -> Var {
        assert_eq!(data.len(), rows * cols);
        self.push(rows, cols, data, false, Op::Leaf)
    }

    // ---------- forward ops ----------
    pub fn matmul(&mut self, a: Var, b: Var) -> Var {
        let (m, k) = self.shape(a);
        let (k2, n) = self.shape(b);
        assert_eq!(k, k2, "matmul shape mismatch");
        let mut out = vec![0.0; m * n];
        gemm_nn(&self.tensors[a.0].data, &self.tensors[b.0].data, &mut out, m, k, n, false);
        let rg = self.rg(&[a.0, b.0]);
        self.push(m, n, out, rg, Op::MatMul(a.0, b.0))
    }
    pub fn add(&mut self, a: Var, b: Var) -> Var {
        assert_eq!(self.shape(a), self.shape(b));
        let (r, c) = self.shape(a);
        let out: Vec<f32> = self.tensors[a.0].data.iter().zip(&self.tensors[b.0].data).map(|(x, y)| x + y).collect();
        let rg = self.rg(&[a.0, b.0]);
        self.push(r, c, out, rg, Op::Add(a.0, b.0))
    }
    /// x[N,D] + b[1,D] broadcast over rows.
    pub fn add_bias(&mut self, x: Var, b: Var) -> Var {
        let (n, d) = self.shape(x);
        assert_eq!(self.shape(b), (1, d));
        let mut out = self.tensors[x.0].data.clone();
        let bd = &self.tensors[b.0].data;
        for row in out.chunks_mut(d) {
            add_into(row, bd);
        }
        let rg = self.rg(&[x.0, b.0]);
        self.push(n, d, out, rg, Op::AddBias(x.0, b.0))
    }
    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        assert_eq!(self.shape(a), self.shape(b));
        let (r, c) = self.shape(a);
        let out: Vec<f32> = self.tensors[a.0].data.iter().zip(&self.tensors[b.0].data).map(|(x, y)| x * y).collect();
        let rg = self.rg(&[a.0, b.0]);
        self.push(r, c, out, rg, Op::Mul(a.0, b.0))
    }
    pub fn scale(&mut self, a: Var, s: f32) -> Var {
        let (r, c) = self.shape(a);
        let out: Vec<f32> = self.tensors[a.0].data.iter().map(|x| x * s).collect();
        let rg = self.rg(&[a.0]);
        self.push(r, c, out, rg, Op::Scale(a.0, s))
    }
    fn unary(&mut self, a: Var, f: impl Fn(f32) -> f32 + Sync, mk: fn(usize) -> Op) -> Var {
        let (r, c) = self.shape(a);
        let out: Vec<f32> = self.tensors[a.0].data.par_iter().map(|&x| f(x)).collect();
        let rg = self.rg(&[a.0]);
        self.push(r, c, out, rg, mk(a.0))
    }
    pub fn silu(&mut self, a: Var) -> Var {
        self.unary(a, |x| x / (1.0 + (-x).exp()), Op::Silu)
    }
    pub fn sigmoid(&mut self, a: Var) -> Var {
        self.unary(a, |x| 1.0 / (1.0 + (-x).exp()), Op::Sigmoid)
    }
    pub fn tanh(&mut self, a: Var) -> Var {
        self.unary(a, |x| x.tanh(), Op::Tanh)
    }
    pub fn relu(&mut self, a: Var) -> Var {
        self.unary(a, |x| x.max(0.0), Op::Relu)
    }

    /// RMSNorm: y = x / sqrt(mean(x^2)+eps) * g, per row. g is [1,D].
    pub fn rms_norm(&mut self, x: Var, g: Var, eps: f32) -> Var {
        let (n, d) = self.shape(x);
        assert_eq!(self.shape(g), (1, d));
        let mut out = vec![0.0; n * d];
        {
            let xd = &self.tensors[x.0].data;
            let gd = &self.tensors[g.0].data;
            out.par_chunks_mut(d).enumerate().for_each(|(r, o)| {
                let row = &xd[r * d..(r + 1) * d];
                let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
                let inv = 1.0 / (ms + eps).sqrt();
                for j in 0..d {
                    o[j] = row[j] * inv * gd[j];
                }
            });
        }
        let rg = self.rg(&[x.0, g.0]);
        self.push(n, d, out, rg, Op::RmsNorm { x: x.0, g: g.0, eps })
    }
    /// LayerNorm: y = (x-mean)/sqrt(var+eps) * g + b, per row.
    pub fn layer_norm(&mut self, x: Var, g: Var, b: Var, eps: f32) -> Var {
        let (n, d) = self.shape(x);
        assert_eq!(self.shape(g), (1, d));
        assert_eq!(self.shape(b), (1, d));
        let mut out = vec![0.0; n * d];
        {
            let xd = &self.tensors[x.0].data;
            let gd = &self.tensors[g.0].data;
            let bd = &self.tensors[b.0].data;
            out.par_chunks_mut(d).enumerate().for_each(|(r, o)| {
                let row = &xd[r * d..(r + 1) * d];
                let mean = row.iter().sum::<f32>() / d as f32;
                let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
                let inv = 1.0 / (var + eps).sqrt();
                for j in 0..d {
                    o[j] = (row[j] - mean) * inv * gd[j] + bd[j];
                }
            });
        }
        let rg = self.rg(&[x.0, g.0, b.0]);
        self.push(n, d, out, rg, Op::LayerNorm { x: x.0, g: g.0, b: b.0, eps })
    }
    /// Gather rows of `table` ([V,D]) by id.
    pub fn embedding(&mut self, table: Var, ids: &[u32]) -> Var {
        let (v, d) = self.shape(table);
        let mut out = vec![0.0; ids.len() * d];
        let td = &self.tensors[table.0].data;
        for (r, &id) in ids.iter().enumerate() {
            assert!((id as usize) < v, "embedding id out of range");
            out[r * d..(r + 1) * d].copy_from_slice(&td[id as usize * d..(id as usize + 1) * d]);
        }
        let rg = self.rg(&[table.0]);
        self.push(ids.len(), d, out, rg, Op::Embedding { table: table.0, ids: ids.to_vec() })
    }
    /// Mean cross-entropy over rows with target >= 0 (target < 0 is ignored / padding).
    pub fn cross_entropy(&mut self, logits: Var, targets: &[i32]) -> Var {
        let (n, v) = self.shape(logits);
        assert_eq!(targets.len(), n);
        let ld = &self.tensors[logits.0].data;
        let per_row: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|r| {
                let t = targets[r];
                if t < 0 {
                    return 0.0f32;
                }
                let row = &ld[r * v..(r + 1) * v];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let lse = row.iter().map(|x| (x - mx).exp()).sum::<f32>().ln() + mx;
                lse - row[t as usize]
            })
            .collect();
        // sequential f64 sum => bit-for-bit deterministic regardless of thread scheduling
        let sum: f64 = per_row.iter().map(|&x| x as f64).sum();
        let count = targets.iter().filter(|&&t| t >= 0).count();
        let loss = if count == 0 { 0.0 } else { (sum / count as f64) as f32 };
        let rg = self.rg(&[logits.0]);
        self.push(1, 1, vec![loss], rg, Op::CrossEntropy { logits: logits.0, targets: targets.to_vec() })
    }
    /// Gated linear recurrence: h_t = a_t * h_{t-1} + b_t (elementwise), h_{-1}=0,
    /// independently per sequence. a,b are [bsz*t, D].
    pub fn linear_scan(&mut self, a: Var, b: Var, bsz: usize, t: usize) -> Var {
        let (n, d) = self.shape(a);
        assert_eq!(self.shape(b), (n, d));
        assert_eq!(n, bsz * t);
        let mut out = vec![0.0; n * d];
        {
            let ad = &self.tensors[a.0].data;
            let bd = &self.tensors[b.0].data;
            out.par_chunks_mut(t * d).enumerate().for_each(|(bi, o)| {
                for ti in 0..t {
                    let row = bi * t + ti;
                    for j in 0..d {
                        let prev = if ti > 0 { o[(ti - 1) * d + j] } else { 0.0 };
                        o[ti * d + j] = ad[row * d + j] * prev + bd[row * d + j];
                    }
                }
            });
        }
        let rg = self.rg(&[a.0, b.0]);
        self.push(n, d, out, rg, Op::LinearScan { a: a.0, b: b.0, bsz, t })
    }
    /// Causal sliding-window multi-head attention. q,k,v are [bsz*t, h*hd].
    pub fn attention(&mut self, q: Var, k: Var, v: Var, bsz: usize, t: usize, h: usize, w: usize) -> Var {
        let (n, cols) = self.shape(q);
        assert_eq!(self.shape(k), (n, cols));
        assert_eq!(self.shape(v), (n, cols));
        assert_eq!(n, bsz * t);
        assert!(cols % h == 0 && w >= 1);
        let hd = cols / h;
        let scale = 1.0 / (hd as f32).sqrt();
        let (qd, kd, vd) = (&self.tensors[q.0].data, &self.tensors[k.0].data, &self.tensors[v.0].data);
        let res: Vec<(Vec<f32>, Vec<f32>)> = (0..bsz * h)
            .into_par_iter()
            .map(|bh| {
                let (b, hh) = (bh / h, bh % h);
                let off = hh * hd;
                let mut o = vec![0f32; t * hd];
                let mut pr = vec![0f32; t * w];
                for ti in 0..t {
                    let lo = (ti + 1).saturating_sub(w);
                    let len = ti + 1 - lo;
                    let qrow = &qd[(b * t + ti) * cols + off..][..hd];
                    let p = &mut pr[ti * w..ti * w + len];
                    let mut mx = f32::NEG_INFINITY;
                    for (j, s) in (lo..=ti).enumerate() {
                        let sc = dot(qrow, &kd[(b * t + s) * cols + off..][..hd]) * scale;
                        p[j] = sc;
                        if sc > mx {
                            mx = sc;
                        }
                    }
                    let mut sum = 0.0;
                    for x in p.iter_mut() {
                        *x = (*x - mx).exp();
                        sum += *x;
                    }
                    let inv = 1.0 / sum;
                    for x in p.iter_mut() {
                        *x *= inv;
                    }
                    let orow = &mut o[ti * hd..(ti + 1) * hd];
                    for (j, s) in (lo..=ti).enumerate() {
                        let vrow = &vd[(b * t + s) * cols + off..][..hd];
                        let pj = p[j];
                        for d in 0..hd {
                            orow[d] += pj * vrow[d];
                        }
                    }
                }
                (o, pr)
            })
            .collect();
        let mut out = vec![0f32; n * cols];
        let mut probs = Vec::with_capacity(bsz * h * t * w);
        for (bh, (o, pr)) in res.into_iter().enumerate() {
            let (b, hh) = (bh / h, bh % h);
            for ti in 0..t {
                out[(b * t + ti) * cols + hh * hd..][..hd].copy_from_slice(&o[ti * hd..(ti + 1) * hd]);
            }
            probs.extend_from_slice(&pr);
        }
        let rg = self.rg(&[q.0, k.0, v.0]);
        let probs = if rg { probs } else { Vec::new() };
        self.push(n, cols, out, rg, Op::Attention { q: q.0, k: k.0, v: v.0, bsz, t, h, w, probs })
    }
    /// Rotary embedding on [bsz*t, h*hd]; position = row % t.
    pub fn rope(&mut self, x: Var, t: usize, h: usize, base: f32) -> Var {
        let (n, cols) = self.shape(x);
        let hd = cols / h;
        assert!(hd % 2 == 0);
        let mut out = self.tensors[x.0].data.clone();
        out.par_chunks_mut(cols).enumerate().for_each(|(r, row)| rope_apply(row, r % t, h, hd, base, false));
        let rg = self.rg(&[x.0]);
        self.push(n, cols, out, rg, Op::Rope { x: x.0, t, h, base })
    }
    /// Scalar sum(x_i * w_i); used for tests and custom losses.
    pub fn weighted_sum(&mut self, x: Var, w: Vec<f32>) -> Var {
        assert_eq!(self.tensors[x.0].data.len(), w.len());
        let s: f32 = self.tensors[x.0].data.iter().zip(&w).map(|(a, b)| a * b).sum();
        let rg = self.rg(&[x.0]);
        self.push(1, 1, vec![s], rg, Op::WeightedSum { x: x.0, w })
    }

    // ---------- backward ----------
    /// Backpropagate from a scalar `loss`. Parameter gradients accumulate (call `zero_grad` between steps).
    pub fn backward(&mut self, loss: Var) {
        assert_eq!(self.tensors[loss.0].data.len(), 1, "loss must be scalar");
        assert!(self.tensors[loss.0].requires_grad, "loss does not depend on any parameter");
        let persistent = self.persistent;
        for t in self.tensors[persistent..].iter_mut() {
            t.grad = Vec::new();
        }
        self.tensors[loss.0].grad = vec![1.0];
        let ops = &self.ops;
        let tensors = &mut self.tensors;
        for i in (persistent..=loss.0).rev() {
            if !tensors[i].requires_grad {
                continue;
            }
            // Lazily allocate input gradients (keeps peak memory near the live frontier).
            for j in op_inputs(&ops[i]) {
                if j >= persistent && tensors[j].requires_grad && tensors[j].grad.is_empty() {
                    tensors[j].grad = vec![0.0; tensors[j].data.len()];
                }
            }
            let (left, right) = tensors.split_at_mut(i);
            let out = &right[0];
            match &ops[i] {
                Op::Leaf => {}
                Op::MatMul(a, b) => {
                    let (m, k) = (left[*a].rows, left[*a].cols);
                    let n = left[*b].cols;
                    if left[*a].requires_grad {
                        let (ga, bt) = pair(left, *a, *b);
                        // dA[m,k] += dC[m,n] * B[k,n]^T
                        gemm_nt(&out.grad, &bt.data, &mut ga.grad, m, n, k, true);
                    }
                    if left[*b].requires_grad {
                        let (gb, at) = pair(left, *b, *a);
                        // dB[k,n] += A[m,k]^T * dC[m,n]
                        gemm_tn(&at.data, &out.grad, &mut gb.grad, m, k, n, true);
                    }
                }
                Op::Add(a, b) => {
                    if left[*a].requires_grad {
                        add_into(&mut left[*a].grad, &out.grad);
                    }
                    if left[*b].requires_grad {
                        add_into(&mut left[*b].grad, &out.grad);
                    }
                }
                Op::AddBias(x, b) => {
                    let d = out.cols;
                    if left[*x].requires_grad {
                        add_into(&mut left[*x].grad, &out.grad);
                    }
                    if left[*b].requires_grad {
                        let gb = &mut left[*b].grad;
                        for row in out.grad.chunks(d) {
                            add_into(gb, row);
                        }
                    }
                }
                Op::Mul(a, b) => {
                    let ta: Option<Vec<f32>> = left[*a].requires_grad.then(|| out.grad.iter().zip(&left[*b].data).map(|(g, y)| g * y).collect());
                    let tb: Option<Vec<f32>> = left[*b].requires_grad.then(|| out.grad.iter().zip(&left[*a].data).map(|(g, x)| g * x).collect());
                    if let Some(t) = ta {
                        add_into(&mut left[*a].grad, &t);
                    }
                    if let Some(t) = tb {
                        add_into(&mut left[*b].grad, &t);
                    }
                }
                Op::Scale(a, s) => {
                    if left[*a].requires_grad {
                        for (d, g) in left[*a].grad.iter_mut().zip(&out.grad) {
                            *d += s * g;
                        }
                    }
                }
                Op::Silu(a) => {
                    let t = &mut left[*a];
                    for ((d, &x), &g) in t.grad.iter_mut().zip(&t.data).zip(&out.grad) {
                        let s = 1.0 / (1.0 + (-x).exp());
                        *d += g * s * (1.0 + x * (1.0 - s));
                    }
                }
                Op::Sigmoid(a) => {
                    let t = &mut left[*a];
                    for ((d, &y), &g) in t.grad.iter_mut().zip(&out.data).zip(&out.grad) {
                        *d += g * y * (1.0 - y);
                    }
                }
                Op::Tanh(a) => {
                    let t = &mut left[*a];
                    for ((d, &y), &g) in t.grad.iter_mut().zip(&out.data).zip(&out.grad) {
                        *d += g * (1.0 - y * y);
                    }
                }
                Op::Relu(a) => {
                    let t = &mut left[*a];
                    for ((d, &x), &g) in t.grad.iter_mut().zip(&t.data).zip(&out.grad) {
                        if x > 0.0 {
                            *d += g;
                        }
                    }
                }
                Op::RmsNorm { x, g, eps } => {
                    let d = out.cols;
                    let n = out.rows;
                    let dy = &out.grad;
                    if left[*x].requires_grad {
                        let (gx, gt) = pair(left, *x, *g);
                        let gd = &gt.data;
                        let xd = &gx.data;
                        let upd: Vec<f32> = (0..n)
                            .into_par_iter()
                            .flat_map_iter(|r| {
                                let row = &xd[r * d..(r + 1) * d];
                                let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
                                let inv = 1.0 / (ms + eps).sqrt();
                                let s: f32 = (0..d).map(|j| dy[r * d + j] * gd[j] * row[j]).sum();
                                let c = inv * inv * inv * s / d as f32;
                                (0..d).map(move |j| inv * dy[r * d + j] * gd[j] - row[j] * c).collect::<Vec<_>>()
                            })
                            .collect();
                        add_into(&mut gx.grad, &upd);
                    }
                    if left[*g].requires_grad {
                        let (gg, xt) = pair(left, *g, *x);
                        for r in 0..n {
                            let row = &xt.data[r * d..(r + 1) * d];
                            let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
                            let inv = 1.0 / (ms + eps).sqrt();
                            for j in 0..d {
                                gg.grad[j] += dy[r * d + j] * row[j] * inv;
                            }
                        }
                    }
                }
                Op::LayerNorm { x, g, b, eps } => {
                    let d = out.cols;
                    let n = out.rows;
                    let dy = &out.grad;
                    let (mut dxv, mut dgv, mut dbv) = (vec![0f32; n * d], vec![0f32; d], vec![0f32; d]);
                    {
                        let xd = &left[*x].data;
                        let gd = &left[*g].data;
                        for r in 0..n {
                            let row = &xd[r * d..(r + 1) * d];
                            let mean = row.iter().sum::<f32>() / d as f32;
                            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
                            let inv = 1.0 / (var + eps).sqrt();
                            let (mut m1, mut m2) = (0.0f32, 0.0f32);
                            for j in 0..d {
                                let xh = (row[j] - mean) * inv;
                                let dxh = dy[r * d + j] * gd[j];
                                m1 += dxh;
                                m2 += dxh * xh;
                                dgv[j] += dy[r * d + j] * xh;
                                dbv[j] += dy[r * d + j];
                            }
                            m1 /= d as f32;
                            m2 /= d as f32;
                            for j in 0..d {
                                let xh = (row[j] - mean) * inv;
                                dxv[r * d + j] = inv * (dy[r * d + j] * gd[j] - m1 - xh * m2);
                            }
                        }
                    }
                    if left[*x].requires_grad {
                        add_into(&mut left[*x].grad, &dxv);
                    }
                    if left[*g].requires_grad {
                        add_into(&mut left[*g].grad, &dgv);
                    }
                    if left[*b].requires_grad {
                        add_into(&mut left[*b].grad, &dbv);
                    }
                }
                Op::Embedding { table, ids } => {
                    let d = out.cols;
                    let tg = &mut left[*table].grad;
                    for (r, &id) in ids.iter().enumerate() {
                        add_into(&mut tg[id as usize * d..(id as usize + 1) * d], &out.grad[r * d..(r + 1) * d]);
                    }
                }
                Op::CrossEntropy { logits, targets } => {
                    let v = left[*logits].cols;
                    let count = targets.iter().filter(|&&t| t >= 0).count();
                    if count > 0 {
                        let go = out.grad[0] / count as f32;
                        let lt = &mut left[*logits];
                        let (ld, lg) = (&lt.data, &mut lt.grad);
                        lg.par_chunks_mut(v).enumerate().for_each(|(r, grow)| {
                            let t = targets[r];
                            if t < 0 {
                                return;
                            }
                            let row = &ld[r * v..(r + 1) * v];
                            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f32 = row.iter().map(|x| (x - mx).exp()).sum();
                            for j in 0..v {
                                let p = (row[j] - mx).exp() / sum;
                                grow[j] += go * (p - if j == t as usize { 1.0 } else { 0.0 });
                            }
                        });
                    }
                }
                Op::LinearScan { a, b, bsz, t } => {
                    let d = out.cols;
                    let (mut da, mut db) = (vec![0f32; out.data.len()], vec![0f32; out.data.len()]);
                    {
                        let ad = &left[*a].data;
                        for bi in 0..*bsz {
                            let mut g = vec![0f32; d];
                            for ti in (0..*t).rev() {
                                let row = bi * t + ti;
                                for j in 0..d {
                                    let carry = if ti + 1 < *t { ad[(row + 1) * d + j] * g[j] } else { 0.0 };
                                    g[j] = out.grad[row * d + j] + carry;
                                    db[row * d + j] = g[j];
                                    if ti > 0 {
                                        da[row * d + j] = g[j] * out.data[(row - 1) * d + j];
                                    }
                                }
                            }
                        }
                    }
                    if left[*a].requires_grad {
                        add_into(&mut left[*a].grad, &da);
                    }
                    if left[*b].requires_grad {
                        add_into(&mut left[*b].grad, &db);
                    }
                }
                Op::Attention { q, k, v, bsz, t, h, w, probs } => {
                    let (bsz, t, h, w) = (*bsz, *t, *h, *w);
                    let cols = out.cols;
                    let hd = cols / h;
                    let scale = 1.0 / (hd as f32).sqrt();
                    let res: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = {
                        let (qd, kd, vd) = (&left[*q].data, &left[*k].data, &left[*v].data);
                        (0..bsz * h)
                            .into_par_iter()
                            .map(|bh| {
                                let (b, hh) = (bh / h, bh % h);
                                let off = hh * hd;
                                let base = bh * t * w;
                                let (mut dq, mut dk, mut dv) = (vec![0f32; t * hd], vec![0f32; t * hd], vec![0f32; t * hd]);
                                let mut dp = vec![0f32; w];
                                for ti in 0..t {
                                    let lo = (ti + 1).saturating_sub(w);
                                    let len = ti + 1 - lo;
                                    let p = &probs[base + ti * w..base + ti * w + len];
                                    let dorow = &out.grad[(b * t + ti) * cols + off..][..hd];
                                    let qrow = &qd[(b * t + ti) * cols + off..][..hd];
                                    let mut dsum = 0.0;
                                    for (j, s) in (lo..=ti).enumerate() {
                                        dp[j] = dot(dorow, &vd[(b * t + s) * cols + off..][..hd]);
                                        dsum += p[j] * dp[j];
                                    }
                                    for (j, s) in (lo..=ti).enumerate() {
                                        let ds = p[j] * (dp[j] - dsum) * scale;
                                        let krow = &kd[(b * t + s) * cols + off..][..hd];
                                        for d in 0..hd {
                                            dq[ti * hd + d] += ds * krow[d];
                                            dk[(s - 0) * hd + d] += ds * qrow[d];
                                            dv[s * hd + d] += p[j] * dorow[d];
                                        }
                                    }
                                }
                                (dq, dk, dv)
                            })
                            .collect()
                    };
                    for (bh, (dq, dk, dv)) in res.into_iter().enumerate() {
                        let (b, hh) = (bh / h, bh % h);
                        for ti in 0..t {
                            let o = (b * t + ti) * cols + hh * hd;
                            if left[*q].requires_grad {
                                add_into(&mut left[*q].grad[o..o + hd], &dq[ti * hd..(ti + 1) * hd]);
                            }
                            if left[*k].requires_grad {
                                add_into(&mut left[*k].grad[o..o + hd], &dk[ti * hd..(ti + 1) * hd]);
                            }
                            if left[*v].requires_grad {
                                add_into(&mut left[*v].grad[o..o + hd], &dv[ti * hd..(ti + 1) * hd]);
                            }
                        }
                    }
                }
                Op::Rope { x, t, h, base } => {
                    let cols = out.cols;
                    let hd = cols / h;
                    let mut g = out.grad.clone();
                    g.par_chunks_mut(cols).enumerate().for_each(|(r, row)| rope_apply(row, r % t, *h, hd, *base, true));
                    add_into(&mut left[*x].grad, &g);
                }
                Op::WeightedSum { x, w } => {
                    let go = out.grad[0];
                    for (d, wi) in left[*x].grad.iter_mut().zip(w) {
                        *d += go * wi;
                    }
                }
            }
            if i != loss.0 {
                tensors[i].grad = Vec::new();
            }
        }
    }
}
