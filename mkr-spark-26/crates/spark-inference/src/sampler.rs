use spark_byte_engine::EOS;
use spark_tensor::Rng;

#[derive(Clone, Debug)]
pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    rng: Rng,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32, seed: u64) -> Self {
        Sampler { temperature, top_k, top_p, repetition_penalty, rng: Rng::new(seed) }
    }
    pub fn greedy() -> Self {
        Self::new(0.0, 0, 1.0, 1.0, 0)
    }

    /// Only bytes and EOS may be emitted; other control tokens are masked out.
    fn allowed(t: usize) -> bool {
        t < 256 || t == EOS as usize
    }

    /// Log-softmax probabilities restricted to allowed tokens (for confidence estimation).
    pub fn log_probs(logits: &[f32]) -> Vec<f32> {
        let mx = logits.iter().enumerate().filter(|(i, _)| Self::allowed(*i)).map(|(_, &v)| v).fold(f32::NEG_INFINITY, f32::max);
        let lse = logits.iter().enumerate().filter(|(i, _)| Self::allowed(*i)).map(|(_, &v)| (v - mx).exp()).sum::<f32>().ln() + mx;
        logits.iter().enumerate().map(|(i, &v)| if Self::allowed(i) { v - lse } else { f32::NEG_INFINITY }).collect()
    }

    pub fn sample(&mut self, logits: &[f32], recent: &[u32]) -> u32 {
        let mut l: Vec<f32> = logits.to_vec();
        if self.repetition_penalty != 1.0 {
            for &t in recent {
                let v = &mut l[t as usize];
                *v = if *v > 0.0 { *v / self.repetition_penalty } else { *v * self.repetition_penalty };
            }
        }
        for (i, v) in l.iter_mut().enumerate() {
            if !Self::allowed(i) {
                *v = f32::NEG_INFINITY;
            }
        }
        if self.temperature <= 0.0 {
            return l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i as u32).unwrap();
        }
        let mut idx: Vec<usize> = (0..l.len()).filter(|&i| l[i].is_finite()).collect();
        idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap_or(std::cmp::Ordering::Equal));
        if self.top_k > 0 && idx.len() > self.top_k {
            idx.truncate(self.top_k);
        }
        let mx = l[idx[0]];
        let mut probs: Vec<f32> = idx.iter().map(|&i| ((l[i] - mx) / self.temperature).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        if self.top_p < 1.0 {
            let mut cum = 0.0;
            let mut cut = probs.len();
            for (i, p) in probs.iter().enumerate() {
                cum += p;
                if cum >= self.top_p {
                    cut = i + 1;
                    break;
                }
            }
            probs.truncate(cut);
            idx.truncate(cut);
            let s: f32 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= s;
            }
        }
        let r = self.rng.uniform();
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if r <= acc {
                return idx[i] as u32;
            }
        }
        idx[idx.len() - 1] as u32
    }
}
