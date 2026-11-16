//! Load any `.model` file into an Engine. f32 models use the training container; int8 models
//! use a sibling container ("MKRSPARQ") with the same CRC integrity check.
use crate::engine::{Engine, Raw};
use spark_neural::io::{crc32, load_model, read_config, write_config, Reader, Writer, VERSION};
use spark_neural::Model;
use std::path::Path;

const QMAGIC: &[u8; 8] = b"MKRSPARQ";

pub fn save_quantized(m: &Model, path: &Path) -> Result<(), String> {
    let mut w = Writer(Vec::new());
    write_config(&mut w, &m.cfg);
    let raws = Engine::raw_from_model(m, true);
    w.u32(raws.len() as u32);
    for r in &raws {
        match r {
            Raw::F32 { rows, cols, d } => {
                w.0.push(0);
                w.u32(*rows as u32);
                w.u32(*cols as u32);
                w.f32s(d);
            }
            Raw::I8 { rows, cols, q, scale } => {
                w.0.push(1);
                w.u32(*rows as u32);
                w.u32(*cols as u32);
                w.f32s(scale);
                w.0.extend(q.iter().map(|&v| v as u8));
            }
        }
    }
    let payload = w.0;
    let mut out = Vec::with_capacity(payload.len() + 20);
    out.extend_from_slice(QMAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out.extend_from_slice(&crc32(&payload).to_le_bytes());
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, out).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

fn load_quantized_bytes(data: &[u8]) -> Result<Engine, String> {
    if data.len() < 20 {
        return Err("file too small".into());
    }
    let len = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    if data.len() != 16 + len + 4 {
        return Err("file length mismatch (truncated or padded)".into());
    }
    let payload = &data[16..16 + len];
    if crc32(payload) != u32::from_le_bytes(data[16 + len..].try_into().unwrap()) {
        return Err("checksum mismatch: file is corrupt".into());
    }
    let mut r = Reader::new(payload);
    let cfg = read_config(&mut r)?;
    let n = r.u32()? as usize;
    let mut raws = Vec::new();
    for _ in 0..n {
        let kind = r.take(1)?[0];
        let (rows, cols) = (r.u32()? as usize, r.u32()? as usize);
        let count = rows.checked_mul(cols).ok_or("size overflow")?;
        raws.push(match kind {
            0 => Raw::F32 { rows, cols, d: r.f32s(count)? },
            1 => {
                let scale = r.f32s(cols)?;
                let q = r.take(count)?.iter().map(|&b| b as i8).collect();
                Raw::I8 { rows, cols, q, scale }
            }
            _ => return Err("unknown tensor kind".into()),
        });
    }
    Engine::assemble(cfg, raws)
}

/// Load either an f32 or an int8 model file into an inference engine.
pub fn load_engine(path: &Path) -> Result<Engine, String> {
    let head = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if head.len() >= 8 && &head[..8] == QMAGIC {
        load_quantized_bytes(&head)
    } else {
        let m = load_model(path)?;
        Ok(Engine::from_model(&m, false))
    }
}
