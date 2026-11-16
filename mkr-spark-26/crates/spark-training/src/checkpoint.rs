//! Training checkpoints: model + optimiser state + progress, integrity-checked and written atomically.
use spark_neural::io::{model_from_payload, model_to_payload, unwrap, wrap, Reader, Writer};
use spark_neural::Model;
use std::path::{Path, PathBuf};

const TAG: &[u8; 4] = b"CKPT";

#[derive(Clone, Debug, Default)]
pub struct EvalPoint {
    pub step: u64,
    pub train_loss: f32,
    pub val_loss: f32,
}

#[derive(Clone, Debug)]
pub struct TrainState {
    pub step: u64,
    pub best_val: f32,
    pub best_step: u64,
    pub history: Vec<EvalPoint>,
}

impl Default for TrainState {
    fn default() -> Self {
        TrainState { step: 0, best_val: f32::INFINITY, best_step: 0, history: Vec::new() }
    }
}

pub struct ResumeState {
    pub optim: Vec<u8>,
    pub state: TrainState,
}

pub fn save_checkpoint(dir: &Path, model: &Model, optim: &[u8], state: &TrainState, keep: usize) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut w = Writer(Vec::new());
    w.0.extend_from_slice(TAG);
    w.bytes(&model_to_payload(model));
    w.bytes(optim);
    w.u64(state.step);
    w.f32(state.best_val);
    w.u64(state.best_step);
    w.u32(state.history.len() as u32);
    for h in &state.history {
        w.u64(h.step);
        w.f32(h.train_loss);
        w.f32(h.val_loss);
    }
    let bytes = wrap(&w.0);
    let path = dir.join(format!("step_{:07}.ckpt", state.step));
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    // rotate: keep only the newest `keep` step checkpoints
    let mut all: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "ckpt"))
        .collect();
    all.sort();
    while all.len() > keep.max(1) {
        let _ = std::fs::remove_file(all.remove(0));
    }
    Ok(path)
}

/// Load a checkpoint file, or if `path` is a directory, the newest valid checkpoint inside it
/// (corrupt newer files are skipped, which gives crash recovery).
pub fn load_checkpoint(path: &Path) -> Result<(Model, ResumeState), String> {
    if path.is_dir() {
        let mut all: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|e| e.to_string())?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |x| x == "ckpt"))
            .collect();
        all.sort();
        let mut last_err = "no checkpoints found".to_string();
        for p in all.into_iter().rev() {
            match load_checkpoint(&p) {
                Ok(r) => return Ok(r),
                Err(e) => last_err = format!("{}: {e}", p.display()),
            }
        }
        return Err(last_err);
    }
    let data = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let payload = unwrap(&data)?;
    let mut r = Reader::new(payload);
    if r.take(4)? != TAG {
        return Err("not a training checkpoint".into());
    }
    let model = model_from_payload(r.bytes()?)?;
    let optim = r.bytes()?.to_vec();
    let step = r.u64()?;
    let best_val = r.f32()?;
    let best_step = r.u64()?;
    let n = r.u32()? as usize;
    let mut history = Vec::new();
    for _ in 0..n {
        history.push(EvalPoint { step: r.u64()?, train_loss: r.f32()?, val_loss: r.f32()? });
    }
    Ok((model, ResumeState { optim, state: TrainState { step, best_val, best_step, history } }))
}
