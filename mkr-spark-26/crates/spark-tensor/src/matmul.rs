//! CPU matrix multiplication kernels (row-major f32), multithreaded with rayon.
use rayon::prelude::*;

const PAR_FLOPS: usize = 1 << 17;

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// C[m,n] (+)= A[m,k] * B[k,n]
pub fn gemm_nn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, acc: bool) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 {
        return;
    }
    if m == 1 {
        // matrix-vector: parallelise over output column blocks
        const CH: usize = 1024;
        let body = |(ci, cblk): (usize, &mut [f32])| {
            let j0 = ci * CH;
            if !acc {
                cblk.fill(0.0);
            }
            for p in 0..k {
                let av = a[p];
                let brow = &b[p * n + j0..p * n + j0 + cblk.len()];
                for (cv, &bv) in cblk.iter_mut().zip(brow) {
                    *cv += av * bv;
                }
            }
        };
        if k * n >= PAR_FLOPS {
            c.par_chunks_mut(CH).enumerate().for_each(body);
        } else {
            c.chunks_mut(CH).enumerate().for_each(body);
        }
        return;
    }
    let body = |(i, crow): (usize, &mut [f32])| {
        if !acc {
            crow.fill(0.0);
        }
        let arow = &a[i * k..(i + 1) * k];
        for (p, &av) in arow.iter().enumerate() {
            let brow = &b[p * n..(p + 1) * n];
            for (cv, &bv) in crow.iter_mut().zip(brow) {
                *cv += av * bv;
            }
        }
    };
    if m * k * n >= PAR_FLOPS {
        c.par_chunks_mut(n).enumerate().for_each(body);
    } else {
        c.chunks_mut(n).enumerate().for_each(body);
    }
}

/// C[m,n] (+)= A[m,k] * B[n,k]^T   (B stored row-major as [n,k])
pub fn gemm_nt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, acc: bool) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 {
        return;
    }
    let body = |(i, crow): (usize, &mut [f32])| {
        let arow = &a[i * k..(i + 1) * k];
        for (j, cv) in crow.iter_mut().enumerate() {
            let d = dot(arow, &b[j * k..(j + 1) * k]);
            if acc {
                *cv += d;
            } else {
                *cv = d;
            }
        }
    };
    if m * k * n >= PAR_FLOPS {
        c.par_chunks_mut(n).enumerate().for_each(body);
    } else {
        c.chunks_mut(n).enumerate().for_each(body);
    }
}

/// C[k,n] (+)= A[m,k]^T * B[m,n]
pub fn gemm_tn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, acc: bool) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), m * n);
    assert_eq!(c.len(), k * n);
    if k == 0 || n == 0 {
        return;
    }
    let body = |(p, crow): (usize, &mut [f32])| {
        if !acc {
            crow.fill(0.0);
        }
        for i in 0..m {
            let av = a[i * k + p];
            let brow = &b[i * n..(i + 1) * n];
            for (cv, &bv) in crow.iter_mut().zip(brow) {
                *cv += av * bv;
            }
        }
    };
    if m * k * n >= PAR_FLOPS {
        c.par_chunks_mut(n).enumerate().for_each(body);
    } else {
        c.chunks_mut(n).enumerate().for_each(body);
    }
}
