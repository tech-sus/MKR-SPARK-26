//! AdamW with decoupled weight decay, warmup+cosine schedule, global-norm gradient clipping.
use rayon::prelude::*;
use spark_tensor::Tensor;

#[derive(Clone, Debug)]
pub struct AdamWConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        AdamWConfig { beta1: 0.9, beta2: 0.95, eps: 1e-8, weight_decay: 0.05 }
    }
}

pub struct AdamW {
    pub cfg: AdamWConfig,
    pub step: u64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamW {
    pub fn new(cfg: AdamWConfig, params: &[Tensor]) -> Self {
        AdamW {
            cfg,
            step: 0,
            m: params.iter().map(|p| vec![0.0; p.data.len()]).collect(),
            v: params.iter().map(|p| vec![0.0; p.data.len()]).collect(),
        }
    }

    /// One update. Weight decay applies only to matrices (rows>1 and cols>1); gains/biases are exempt.
    pub fn update(&mut self, params: &mut [Tensor], lr: f32) {
        self.step += 1;
        let c = &self.cfg;
        let bc1 = 1.0 - c.beta1.powi(self.step as i32);
        let bc2 = 1.0 - c.beta2.powi(self.step as i32);
        params
            .par_iter_mut()
            .zip(self.m.par_iter_mut())
            .zip(self.v.par_iter_mut())
            .for_each(|((p, m), v)| {
                let wd = if p.rows > 1 && p.cols > 1 { c.weight_decay } else { 0.0 };
                for i in 0..p.data.len() {
                    let g = p.grad[i];
                    m[i] = c.beta1 * m[i] + (1.0 - c.beta1) * g;
                    v[i] = c.beta2 * v[i] + (1.0 - c.beta2) * g * g;
                    let mh = m[i] / bc1;
                    let vh = v[i] / bc2;
                    p.data[i] -= lr * (mh / (vh.sqrt() + c.eps) + wd * p.data[i]);
                }
            });
    }

    /// Serialise optimiser state (for checkpoint/resume).
    pub fn state_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.step.to_le_bytes());
        out.extend_from_slice(&(self.m.len() as u32).to_le_bytes());
        for (m, v) in self.m.iter().zip(&self.v) {
            out.extend_from_slice(&(m.len() as u64).to_le_bytes());
            for x in m.iter().chain(v.iter()) {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    pub fn load_state(&mut self, bytes: &[u8]) -> Result<(), String> {
        let mut pos = 0usize;
        let mut take = |n: usize| -> Result<&[u8], String> {
            if pos + n > bytes.len() {
                return Err("optimizer state truncated".into());
            }
            let s = &bytes[pos..pos + n];
            pos += n;
            Ok(s)
        };
        let step = u64::from_le_bytes(take(8)?.try_into().unwrap());
        let n = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        if n != self.m.len() {
            return Err("optimizer state does not match model".into());
        }
        for i in 0..n {
            let len = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
            if len != self.m[i].len() {
                return Err("optimizer tensor size mismatch".into());
            }
            for j in 0..len {
                self.m[i][j] = f32::from_le_bytes(take(4)?.try_into().unwrap());
            }
            for j in 0..len {
                self.v[i][j] = f32::from_le_bytes(take(4)?.try_into().unwrap());
            }
        }
        self.step = step;
        Ok(())
    }
}

/// Scale gradients so their global L2 norm is at most `max_norm`. Returns the pre-clip norm.
pub fn clip_grad_norm(params: &mut [Tensor], max_norm: f32) -> f32 {
    let sq: f64 = params.par_iter().map(|p| p.grad.iter().map(|&g| (g as f64) * (g as f64)).sum::<f64>()).sum();
    let norm = sq.sqrt() as f32;
    if norm.is_finite() && norm > max_norm {
        let s = max_norm / (norm + 1e-6);
        params.par_iter_mut().for_each(|p| p.grad.iter_mut().for_each(|g| *g *= s));
    }
    norm
}

#[derive(Clone, Debug)]
pub struct Schedule {
    pub peak_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: u64,
    pub total_steps: u64,
}

impl Schedule {
    pub fn lr_at(&self, step: u64) -> f32 {
        if step < self.warmup_steps {
            return self.peak_lr * (step + 1) as f32 / self.warmup_steps.max(1) as f32;
        }
        if step >= self.total_steps {
            return self.min_lr;
        }
        let p = (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps).max(1) as f32;
        self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * p).cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_param(x0: &[f32]) -> Tensor {
        Tensor { data: x0.to_vec(), grad: vec![0.0; x0.len()], rows: 2, cols: x0.len() / 2, requires_grad: true }
    }

    #[test]
    fn adamw_minimises_quadratic() {
        let mut ps = vec![quad_param(&[3.0, -2.0, 1.5, 4.0])];
        let target = [1.0f32, 2.0, -1.0, 0.5];
        let mut opt = AdamW::new(AdamWConfig { weight_decay: 0.0, ..Default::default() }, &ps);
        for _ in 0..800 {
            for i in 0..4 {
                ps[0].grad[i] = 2.0 * (ps[0].data[i] - target[i]);
            }
            opt.update(&mut ps, 0.05);
        }
        for i in 0..4 {
            assert!((ps[0].data[i] - target[i]).abs() < 0.05, "{} vs {}", ps[0].data[i], target[i]);
        }
    }

    #[test]
    fn weight_decay_shrinks_matrices_only() {
        let mut ps = vec![quad_param(&[1.0; 4]), Tensor { data: vec![1.0; 4], grad: vec![0.0; 4], rows: 1, cols: 4, requires_grad: true }];
        let mut opt = AdamW::new(AdamWConfig { weight_decay: 0.1, ..Default::default() }, &ps);
        for _ in 0..10 {
            opt.update(&mut ps, 0.1);
        }
        assert!(ps[0].data[0] < 1.0);
        assert_eq!(ps[1].data[0], 1.0);
    }

    #[test]
    fn clipping_and_schedule() {
        let mut ps = vec![quad_param(&[0.0; 4])];
        ps[0].grad = vec![3.0, 4.0, 0.0, 0.0];
        let n = clip_grad_norm(&mut ps, 1.0);
        assert!((n - 5.0).abs() < 1e-5);
        let after: f32 = ps[0].grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!((after - 1.0).abs() < 1e-4);
        let s = Schedule { peak_lr: 1.0, min_lr: 0.1, warmup_steps: 10, total_steps: 110 };
        assert!(s.lr_at(0) < s.lr_at(9));
        assert!((s.lr_at(10) - 1.0).abs() < 1e-5);
        assert!((s.lr_at(110) - 0.1).abs() < 1e-5);
        assert!(s.lr_at(60) < 1.0 && s.lr_at(60) > 0.1);
    }

    #[test]
    fn state_roundtrip() {
        let mut ps = vec![quad_param(&[1.0, 2.0, 3.0, 4.0])];
        let mut a = AdamW::new(AdamWConfig::default(), &ps);
        ps[0].grad = vec![0.1, 0.2, 0.3, 0.4];
        a.update(&mut ps, 0.01);
        let bytes = a.state_bytes();
        let mut b = AdamW::new(AdamWConfig::default(), &ps);
        b.load_state(&bytes).unwrap();
        assert_eq!(b.step, 1);
        assert_eq!(b.state_bytes(), bytes);
        assert!(b.load_state(&bytes[..10]).is_err());
    }
}
