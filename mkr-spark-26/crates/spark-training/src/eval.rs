use spark_byte_engine::{Batcher, Corpus};
use spark_neural::Model;

#[derive(Clone, Debug)]
pub struct EvalResult {
    pub loss: f32,
    pub perplexity: f32,
    /// bits per byte = loss / ln 2 (the standard tokenizer-independent metric)
    pub bits_per_byte: f32,
    pub tokens: usize,
}

/// Deterministic evaluation over sequential, non-overlapping windows of `corpus`.
pub fn evaluate(model: &mut Model, corpus: &Corpus, bsz: usize, t: usize, max_batches: usize) -> EvalResult {
    let batches = Batcher::new(corpus, bsz, t, 0).sequential(max_batches);
    model.g.set_grad_enabled(false);
    let (mut sum, mut n) = (0.0f64, 0usize);
    for b in &batches {
        let l = model.loss(&b.inputs, &b.targets, b.bsz, b.t);
        let valid = b.targets.iter().filter(|&&x| x >= 0).count();
        sum += model.g.scalar(l) as f64 * valid as f64;
        n += valid;
        model.g.reset();
    }
    model.g.set_grad_enabled(true);
    let loss = if n == 0 { f32::NAN } else { (sum / n as f64) as f32 };
    EvalResult { loss, perplexity: loss.exp(), bits_per_byte: loss / std::f32::consts::LN_2, tokens: n }
}
