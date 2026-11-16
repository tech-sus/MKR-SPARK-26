use crate::{engine::Engine, sampler::Sampler};
use spark_byte_engine::{Utf8Stream, EOS};
use std::time::Instant;

pub struct GenConfig {
    pub max_new_tokens: usize,
    pub stop_on_eos: bool,
}

impl Default for GenConfig {
    fn default() -> Self {
        GenConfig { max_new_tokens: 200, stop_on_eos: true }
    }
}

pub struct GenResult {
    pub tokens: Vec<u32>,
    pub text: String,
    /// mean log-probability of the emitted tokens under the model (higher = more confident)
    pub mean_logprob: f32,
    /// mean predictive entropy in nats over generated positions
    pub mean_entropy: f32,
    pub tokens_per_sec: f32,
    pub hit_eos: bool,
}

/// Generate a continuation of `prompt`. `on_text` receives UTF-8-safe text pieces as they are produced.
pub fn generate(engine: &Engine, prompt: &[u32], gc: &GenConfig, sampler: &mut Sampler, on_text: &mut dyn FnMut(&str)) -> GenResult {
    let mut st = engine.new_state();
    let mut logits = Vec::new();
    for &t in prompt {
        logits = engine.step(&mut st, t);
    }
    if prompt.is_empty() {
        logits = engine.step(&mut st, spark_byte_engine::BOS);
    }
    let mut out = Vec::new();
    let mut text = String::new();
    let mut utf8 = Utf8Stream::new();
    let (mut lp_sum, mut ent_sum) = (0.0f32, 0.0f32);
    let mut hit_eos = false;
    let start = Instant::now();
    for _ in 0..gc.max_new_tokens {
        let lps = Sampler::log_probs(&logits);
        let tok = sampler.sample(&logits, &out);
        lp_sum += lps[tok as usize];
        ent_sum += -lps.iter().filter(|v| v.is_finite()).map(|&lp| lp.exp() * lp).sum::<f32>();
        if tok == EOS && gc.stop_on_eos {
            hit_eos = true;
            break;
        }
        out.push(tok);
        let piece = utf8.push(tok);
        if !piece.is_empty() {
            on_text(&piece);
            text.push_str(&piece);
        }
        logits = engine.step(&mut st, tok);
    }
    let tail = utf8.finish();
    if !tail.is_empty() {
        on_text(&tail);
        text.push_str(&tail);
    }
    let n = out.len().max(1) as f32;
    GenResult { tokens: out, text, mean_logprob: lp_sum / n, mean_entropy: ent_sum / n, tokens_per_sec: n / start.elapsed().as_secs_f32().max(1e-6), hit_eos }
}
