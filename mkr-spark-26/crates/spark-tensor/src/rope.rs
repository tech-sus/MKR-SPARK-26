//! Rotary position embedding helper shared by training graph and inference engine.
/// Rotate `row` (heads*hd values) in place for position `pos`.
/// `inverse = true` applies the transpose rotation (used for the backward pass).
pub fn rope_apply(row: &mut [f32], pos: usize, heads: usize, hd: usize, base: f32, inverse: bool) {
    let half = hd / 2;
    for h in 0..heads {
        let off = h * hd;
        for i in 0..half {
            let theta = pos as f32 * base.powf(-2.0 * i as f32 / hd as f32);
            let (s, c) = theta.sin_cos();
            let s = if inverse { -s } else { s };
            let x0 = row[off + 2 * i];
            let x1 = row[off + 2 * i + 1];
            row[off + 2 * i] = x0 * c - x1 * s;
            row[off + 2 * i + 1] = x0 * s + x1 * c;
        }
    }
}
