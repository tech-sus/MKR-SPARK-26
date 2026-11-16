use spark_byte_engine::VOCAB_SIZE;

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub name: String,
    pub vocab: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    /// Sliding attention window in tokens (bytes). 0 disables attention (recurrence-only ablation).
    pub window: usize,
    pub ffn_hidden: usize,
    pub max_ctx: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
}

impl Config {
    pub fn new(name: &str, d_model: usize, n_layers: usize, n_heads: usize, window: usize, max_ctx: usize) -> Self {
        // SwiGLU hidden ~ 8/3 d, rounded up to a multiple of 8
        let ffn_hidden = (d_model * 8 / 3).div_ceil(8) * 8;
        Config { name: name.into(), vocab: VOCAB_SIZE, d_model, n_layers, n_heads, window, ffn_hidden, max_ctx, rope_base: 10000.0, norm_eps: 1e-5 }
    }

    /// Named presets. Approximate parameter counts are reported by `param_estimate`.
    pub fn preset(name: &str) -> Option<Config> {
        Some(match name {
            "nano" => Config::new("nano", 64, 2, 2, 16, 256),     // tests / smoke runs (~0.1M)
            "micro" => Config::new("micro", 128, 4, 4, 32, 512),  // 2GB RAM experimental (~1M)
            "mini" => Config::new("mini", 256, 6, 4, 64, 512),    // 4GB RAM (~7M)
            "small" => Config::new("small", 384, 8, 6, 64, 768),  // 8GB RAM (~20M)
            "base" => Config::new("base", 512, 12, 8, 96, 1024),  // 8-16GB RAM (~50M)
            "large" => Config::new("large", 768, 12, 12, 128, 1024), // 16GB RAM (~115M)
            _ => return None,
        })
    }

    pub fn preset_names() -> &'static [&'static str] {
        &["nano", "micro", "mini", "small", "base", "large"]
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.d_model == 0 || self.n_layers == 0 || self.n_heads == 0 {
            return Err("dimensions must be non-zero".into());
        }
        if self.window > 0 {
            if self.d_model % self.n_heads != 0 {
                return Err("d_model must be divisible by n_heads".into());
            }
            if (self.d_model / self.n_heads) % 2 != 0 {
                return Err("head dimension must be even for RoPE".into());
            }
        }
        if self.max_ctx == 0 {
            return Err("max_ctx must be > 0".into());
        }
        Ok(())
    }

    pub fn param_estimate(&self) -> usize {
        let d = self.d_model;
        let scan = 4 * d * d + 2 * d; // wa, wv, wg, wo + ba + norm
        let attn = if self.window > 0 { 4 * d * d + d } else { 0 };
        let ffn = 3 * d * self.ffn_hidden + d;
        2 * self.vocab * d + self.n_layers * (scan + attn + ffn + d) + d
    }
}
