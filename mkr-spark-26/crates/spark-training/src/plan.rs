//! Memory planning for low-resource machines.
use spark_neural::Config;

/// Estimated peak training memory in bytes: weights + gradients + Adam moments (16 B/param)
/// plus activation tape (forward values, with lazily allocated gradients) and logits.
pub fn estimate_train_memory(cfg: &Config, batch: usize, ctx: usize) -> usize {
    let tokens = batch * ctx;
    let d = cfg.d_model;
    let per_layer = 27 * d + 4 * cfg.ffn_hidden + cfg.n_heads * cfg.window;
    let act = tokens * cfg.n_layers * per_layer * 4;
    let logits = tokens * cfg.vocab * 4 * 3;
    cfg.param_estimate() * 16 + (act as f64 * 1.3) as usize + logits
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub ram_gb: u32,
    pub preset: &'static str,
    pub batch: usize,
    pub ctx: usize,
    pub accum: usize,
    pub est_train_bytes: usize,
}

/// Recommended training configuration per RAM tier (leaves ≥40% headroom for the OS).
pub fn plan_for_ram_gb(gb: u32) -> Plan {
    let (preset, batch, ctx, accum) = match gb {
        0..=2 => ("micro", 4, 128, 4),
        3..=4 => ("mini", 4, 256, 4),
        5..=8 => ("small", 4, 384, 4),
        _ => ("base", 8, 512, 2),
    };
    let cfg = Config::preset(preset).unwrap();
    Plan { ram_gb: gb, preset, batch, ctx, accum, est_train_bytes: estimate_train_memory(&cfg, batch, ctx) }
}
