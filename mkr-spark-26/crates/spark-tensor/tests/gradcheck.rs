//! Numerical gradient checks (central differences) for every differentiable op,
//! plus matmul kernel correctness against a naive reference.
use spark_tensor::matmul::{gemm_nn, gemm_nt, gemm_tn};
use spark_tensor::{Graph, Rng, Var};

fn rand_vec(rng: &mut Rng, n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|_| rng.normal() * s).collect()
}

/// `build` creates the scalar loss from the given params. Checks d loss / d param numerically.
fn check(name: &str, shapes: &[(usize, usize)], seed: u64, build: impl Fn(&mut Graph, &[Var]) -> Var) {
    let mut rng = Rng::new(seed);
    let mut g = Graph::new();
    let ps: Vec<Var> = shapes.iter().map(|&(r, c)| g.param(r, c, rand_vec(&mut rng, r * c, 0.7))).collect();
    g.zero_grad();
    let loss = build(&mut g, &ps);
    g.backward(loss);
    let analytic: Vec<Vec<f32>> = ps.iter().map(|&p| g.grad(p).to_vec()).collect();
    g.reset();
    let eps = 2e-3f32;
    let mut worst = 0.0f32;
    for (pi, &p) in ps.iter().enumerate() {
        let n = g.value(p).len();
        for idx in 0..n {
            let orig = g.value(p)[idx];
            g.value_mut(p)[idx] = orig + eps;
            let l1 = build(&mut g, &ps);
            let lp = g.scalar(l1);
            g.reset();
            g.value_mut(p)[idx] = orig - eps;
            let l2 = build(&mut g, &ps);
            let lm = g.scalar(l2);
            g.reset();
            g.value_mut(p)[idx] = orig;
            let num = (lp - lm) / (2.0 * eps);
            let ana = analytic[pi][idx];
            let err = (num - ana).abs() / (1.0 + num.abs().max(ana.abs()));
            worst = worst.max(err);
            assert!(err < 2e-2, "{name}: param {pi} idx {idx}: numeric {num} vs analytic {ana} (err {err})");
        }
    }
    println!("{name}: worst relative error {worst:.5}");
}

fn wsum(g: &mut Graph, y: Var, seed: u64) -> Var {
    let mut rng = Rng::new(seed);
    let n = g.value(y).len();
    g.weighted_sum(y, rand_vec(&mut rng, n, 1.0))
}

#[test]
fn matmul_kernels_match_naive() {
    let mut rng = Rng::new(1);
    for &(m, k, n) in &[(1, 7, 9), (5, 3, 4), (33, 17, 29), (64, 64, 64), (1, 3000, 2500)] {
        let a = rand_vec(&mut rng, m * k, 1.0);
        let b = rand_vec(&mut rng, k * n, 1.0);
        let mut c = vec![0.0; m * n];
        gemm_nn(&a, &b, &mut c, m, k, n, false);
        for i in 0..m.min(4) {
            for j in 0..n.min(8) {
                let r: f32 = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
                assert!((c[i * n + j] - r).abs() < 1e-2 * (1.0 + r.abs()), "nn {m}x{k}x{n}");
            }
        }
        // nt: B2 stored [n,k]
        let b2 = rand_vec(&mut rng, n * k, 1.0);
        let mut c2 = vec![0.0; m * n];
        gemm_nt(&a, &b2, &mut c2, m, k, n, false);
        for i in 0..m.min(4) {
            for j in 0..n.min(8) {
                let r: f32 = (0..k).map(|p| a[i * k + p] * b2[j * k + p]).sum();
                assert!((c2[i * n + j] - r).abs() < 1e-2 * (1.0 + r.abs()), "nt");
            }
        }
        // tn: A[m,k]^T * B3[m,n] -> [k,n]
        let b3 = rand_vec(&mut rng, m * n, 1.0);
        let mut c3 = vec![0.0; k * n];
        gemm_tn(&a, &b3, &mut c3, m, k, n, false);
        for p in 0..k.min(4) {
            for j in 0..n.min(8) {
                let r: f32 = (0..m).map(|i| a[i * k + p] * b3[i * n + j]).sum();
                assert!((c3[p * n + j] - r).abs() < 1e-2 * (1.0 + r.abs()), "tn");
            }
        }
    }
}

#[test]
fn grad_matmul_add_bias() {
    check("matmul+bias", &[(4, 5), (5, 3), (1, 3)], 11, |g, p| {
        let y = g.matmul(p[0], p[1]);
        let y = g.add_bias(y, p[2]);
        wsum(g, y, 5)
    });
}

#[test]
fn grad_add_mul_scale() {
    check("add/mul/scale", &[(3, 4), (3, 4)], 12, |g, p| {
        let s = g.add(p[0], p[1]);
        let m = g.mul(s, p[0]); // reuses p0 on both Add and Mul
        let m = g.scale(m, 0.37);
        wsum(g, m, 6)
    });
    check("mul self", &[(2, 3)], 13, |g, p| {
        let m = g.mul(p[0], p[0]);
        wsum(g, m, 7)
    });
}

#[test]
fn grad_activations() {
    check("silu", &[(3, 5)], 14, |g, p| { let y = g.silu(p[0]); wsum(g, y, 8) });
    check("sigmoid", &[(3, 5)], 15, |g, p| { let y = g.sigmoid(p[0]); wsum(g, y, 9) });
    check("tanh", &[(3, 5)], 16, |g, p| { let y = g.tanh(p[0]); wsum(g, y, 10) });
    check("relu", &[(3, 5)], 17, |g, p| { let y = g.relu(p[0]); wsum(g, y, 11) });
}

#[test]
fn grad_norms() {
    check("rmsnorm", &[(4, 6), (1, 6)], 18, |g, p| { let y = g.rms_norm(p[0], p[1], 1e-5); wsum(g, y, 12) });
    check("layernorm", &[(4, 6), (1, 6), (1, 6)], 19, |g, p| { let y = g.layer_norm(p[0], p[1], p[2], 1e-5); wsum(g, y, 13) });
}

#[test]
fn grad_embedding_and_cross_entropy() {
    check("embedding+CE", &[(7, 5), (5, 7)], 20, |g, p| {
        let e = g.embedding(p[0], &[1, 3, 3, 6, 0, 2]);
        let l = g.matmul(e, p[1]);
        g.cross_entropy(l, &[2, -1, 5, 0, 6, 1])
    });
}

#[test]
fn grad_linear_scan() {
    check("linear_scan", &[(6, 4), (6, 4)], 21, |g, p| {
        let a = g.sigmoid(p[0]);
        let y = g.linear_scan(a, p[1], 2, 3);
        wsum(g, y, 14)
    });
}

#[test]
fn grad_attention_and_rope() {
    // bsz=2, t=5, heads=2, hd=4, window=3
    check("attention", &[(10, 8), (10, 8), (10, 8)], 22, |g, p| {
        let y = g.attention(p[0], p[1], p[2], 2, 5, 2, 3);
        wsum(g, y, 15)
    });
    check("rope", &[(10, 8)], 23, |g, p| { let y = g.rope(p[0], 5, 2, 10000.0); wsum(g, y, 16) });
    check("rope+attention", &[(10, 8), (10, 8), (10, 8)], 24, |g, p| {
        let q = g.rope(p[0], 5, 2, 10000.0);
        let k = g.rope(p[1], 5, 2, 10000.0);
        let y = g.attention(q, k, p[2], 2, 5, 2, 5);
        wsum(g, y, 17)
    });
}

#[test]
fn attention_is_causal_and_windowed() {
    let mut rng = Rng::new(3);
    let mut g = Graph::new();
    let (t, cols) = (6, 4);
    let q = g.input(t, cols, rand_vec(&mut rng, t * cols, 1.0));
    let k = g.input(t, cols, rand_vec(&mut rng, t * cols, 1.0));
    let vdata = rand_vec(&mut rng, t * cols, 1.0);
    let v = g.input(t, cols, vdata.clone());
    let y = g.attention(q, k, v, 1, t, 1, 2);
    // row 0 attends only to itself -> equals v[0]
    for d in 0..cols {
        assert!((g.value(y)[d] - vdata[d]).abs() < 1e-5);
    }
    // changing a token outside the window (position 0) must not change row 3 output
    let before: Vec<f32> = g.value(y)[3 * cols..4 * cols].to_vec();
    let mut v2 = vdata.clone();
    for d in 0..cols { v2[d] += 5.0; }
    let v2 = g.input(t, cols, v2);
    let y2 = g.attention(q, k, v2, 1, t, 1, 2);
    for d in 0..cols {
        assert!((g.value(y2)[3 * cols + d] - before[d]).abs() < 1e-6);
    }
}
