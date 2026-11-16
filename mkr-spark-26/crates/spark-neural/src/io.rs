//! `.model` file format (little-endian, integrity-checked):
//!   magic "MKRSPARK" | u32 version | u32 payload_len | payload | u32 crc32(payload)
//! payload = config fields, then for each parameter tensor: u32 rows, u32 cols, f32 data.
use crate::{config::Config, model::Model};
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"MKRSPARK";
pub const VERSION: u32 = 1;

pub fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for i in 0..256u32 {
        let mut c = i;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 };
        }
        table[i as usize] = c;
    }
    let mut crc = !0u32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

pub struct Writer(pub Vec<u8>);
impl Writer {
    pub fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn bytes(&mut self, b: &[u8]) {
        self.u64(b.len() as u64);
        self.0.extend_from_slice(b);
    }
    pub fn f32s(&mut self, v: &[f32]) {
        self.0.reserve(v.len() * 4);
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
}

pub struct Reader<'a> {
    pub b: &'a [u8],
    pub pos: usize,
}
impl<'a> Reader<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Reader { b, pos: 0 }
    }
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos.checked_add(n).map_or(true, |e| e > self.b.len()) {
            return Err("unexpected end of file (truncated or corrupt)".into());
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    pub fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn bytes(&mut self) -> Result<&'a [u8], String> {
        let n = self.u64()? as usize;
        self.take(n)
    }
    pub fn f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let raw = self.take(n.checked_mul(4).ok_or("size overflow")?)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

pub fn write_config(w: &mut Writer, c: &Config) {
    w.bytes(c.name.as_bytes());
    for v in [c.vocab, c.d_model, c.n_layers, c.n_heads, c.window, c.ffn_hidden, c.max_ctx] {
        w.u32(v as u32);
    }
    w.f32(c.rope_base);
    w.f32(c.norm_eps);
}

pub fn read_config(r: &mut Reader) -> Result<Config, String> {
    let name = String::from_utf8(r.bytes()?.to_vec()).map_err(|_| "bad config name")?;
    let mut v = [0usize; 7];
    for x in v.iter_mut() {
        *x = r.u32()? as usize;
    }
    let c = Config { name, vocab: v[0], d_model: v[1], n_layers: v[2], n_heads: v[3], window: v[4], ffn_hidden: v[5], max_ctx: v[6], rope_base: r.f32()?, norm_eps: r.f32()? };
    c.validate()?;
    if c.d_model > 1 << 16 || c.n_layers > 1024 || c.vocab > 1 << 20 {
        return Err("config values implausible; file is corrupt".into());
    }
    Ok(c)
}

/// Wrap a payload in the integrity-checked container.
pub fn wrap(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 20);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&crc32(payload).to_le_bytes());
    out
}

/// Validate magic/version/length/CRC and return the payload slice.
pub fn unwrap(data: &[u8]) -> Result<&[u8], String> {
    if data.len() < 20 || &data[..8] != MAGIC {
        return Err("not a MKR-SPARK file (bad magic)".into());
    }
    let ver = u32::from_le_bytes(data[8..12].try_into().unwrap());
    if ver != VERSION {
        return Err(format!("unsupported file version {ver}"));
    }
    let len = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    if data.len() != 16 + len + 4 {
        return Err("file length mismatch (truncated or padded)".into());
    }
    let payload = &data[16..16 + len];
    let stored = u32::from_le_bytes(data[16 + len..].try_into().unwrap());
    if crc32(payload) != stored {
        return Err("checksum mismatch: file is corrupt".into());
    }
    Ok(payload)
}

pub fn model_to_payload(m: &Model) -> Vec<u8> {
    let mut w = Writer(Vec::new());
    write_config(&mut w, &m.cfg);
    let ps = m.g.params();
    w.u32(ps.len() as u32);
    for p in ps {
        w.u32(p.rows as u32);
        w.u32(p.cols as u32);
        w.f32s(&p.data);
    }
    w.0
}

pub fn model_from_payload(payload: &[u8]) -> Result<Model, String> {
    let mut r = Reader::new(payload);
    let cfg = read_config(&mut r)?;
    let mut m = Model::new(cfg, 0)?;
    let n = r.u32()? as usize;
    if n != m.g.params().len() {
        return Err(format!("parameter count mismatch: file has {n}, architecture needs {}", m.g.params().len()));
    }
    for p in m.g.params_mut() {
        let (rows, cols) = (r.u32()? as usize, r.u32()? as usize);
        if rows != p.rows || cols != p.cols {
            return Err("tensor shape mismatch".into());
        }
        p.data = r.f32s(rows * cols)?;
    }
    if r.pos != payload.len() {
        return Err("trailing bytes in payload".into());
    }
    Ok(m)
}

pub fn save_model(m: &Model, path: &Path) -> Result<(), String> {
    let bytes = wrap(&model_to_payload(m));
    // atomic write: temp file then rename, so a crash never leaves a half-written model
    let tmp = path.with_extension("tmp");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

pub fn load_model(path: &Path) -> Result<Model, String> {
    let data = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    model_from_payload(unwrap(&data)?)
}
