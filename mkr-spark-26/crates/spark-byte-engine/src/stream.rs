//! Corpus access and batching. `Corpus` can be fully in memory or file-backed (streaming:
//! random windows are read with seeks, so RAM use is independent of corpus size).
use crate::{corpus::DOC_SEP, EOS, PAD};
use spark_tensor::Rng;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

enum Source {
    Memory(Vec<u8>),
    File { file: Mutex<File>, len: u64 },
}

/// A byte corpus with a train/validation split by contiguous region (no leakage between windows).
pub struct Corpus {
    src: Source,
    pub start: u64,
    pub end: u64,
}

impl Corpus {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let n = bytes.len() as u64;
        Corpus { src: Source::Memory(bytes), start: 0, end: n }
    }
    pub fn from_text(text: &str) -> Self {
        Self::from_bytes(text.as_bytes().to_vec())
    }
    /// Open a file for streaming access (not loaded into RAM).
    pub fn open_streaming(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Corpus { src: Source::File { file: Mutex::new(file), len }, start: 0, end: len })
    }
    pub fn len(&self) -> u64 {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Split into (train, validation) where validation is the final `val_fraction`.
    pub fn split(self, val_fraction: f32) -> (Corpus, Corpus) {
        let n = self.len();
        let val = ((n as f64) * val_fraction as f64) as u64;
        let cut = self.end - val;
        let (a, b) = match self.src {
            Source::Memory(v) => {
                let v2 = v.clone();
                (Source::Memory(v), Source::Memory(v2))
            }
            Source::File { file, len } => {
                let f2 = file.lock().unwrap().try_clone().expect("clone file handle");
                (Source::File { file, len }, Source::File { file: Mutex::new(f2), len })
            }
        };
        (Corpus { src: a, start: self.start, end: cut }, Corpus { src: b, start: cut, end: self.end })
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        match &self.src {
            Source::Memory(v) => {
                let o = offset as usize;
                buf.copy_from_slice(&v[o..o + buf.len()]);
                Ok(())
            }
            Source::File { file, .. } => {
                let mut f = file.lock().unwrap();
                f.seek(SeekFrom::Start(offset))?;
                f.read_exact(buf)
            }
        }
    }
    /// Read a window of `n` bytes at a relative offset.
    pub fn window(&self, rel: u64, n: usize) -> Vec<u8> {
        let mut buf = vec![0u8; n];
        self.read_at(self.start + rel, &mut buf).expect("corpus read");
        buf
    }
}

pub struct Batch {
    pub inputs: Vec<u32>,
    pub targets: Vec<i32>,
    pub bsz: usize,
    pub t: usize,
}

pub struct Batcher<'a> {
    corpus: &'a Corpus,
    pub bsz: usize,
    pub t: usize,
    rng: Rng,
}

impl<'a> Batcher<'a> {
    pub fn new(corpus: &'a Corpus, bsz: usize, t: usize, seed: u64) -> Self {
        Batcher { corpus, bsz, t, rng: Rng::new(seed) }
    }

    fn fill(&self, bytes: &[u8], inputs: &mut Vec<u32>, targets: &mut Vec<i32>, t: usize) {
        // bytes.len() == t+1 (or shorter -> padded). DOC_SEP maps to EOS.
        let map = |b: u8| if b == DOC_SEP { EOS } else { b as u32 };
        for i in 0..t {
            if i + 1 < bytes.len() {
                inputs.push(map(bytes[i]));
                targets.push(map(bytes[i + 1]) as i32);
            } else {
                inputs.push(PAD);
                targets.push(-1); // masked: padding contributes no loss
            }
        }
    }

    /// Random training batch: `bsz` windows of `t+1` bytes.
    pub fn next_random(&mut self) -> Batch {
        let (mut inputs, mut targets) = (Vec::new(), Vec::new());
        let n = self.corpus.len();
        for _ in 0..self.bsz {
            let want = (self.t + 1) as u64;
            let len = want.min(n);
            let max_start = n - len;
            let off = if max_start == 0 { 0 } else { self.rng.next_u64() % (max_start + 1) };
            let bytes = self.corpus.window(off, len as usize);
            self.fill(&bytes, &mut inputs, &mut targets, self.t);
        }
        Batch { inputs, targets, bsz: self.bsz, t: self.t }
    }

    /// Deterministic sequential validation batches covering the corpus (non-overlapping windows).
    pub fn sequential(&self, max_batches: usize) -> Vec<Batch> {
        let n = self.corpus.len();
        let step = self.t as u64;
        let mut offs = Vec::new();
        let mut o = 0u64;
        while o + 1 < n {
            offs.push(o);
            o += step;
        }
        let mut batches = Vec::new();
        for chunk in offs.chunks(self.bsz) {
            if batches.len() >= max_batches {
                break;
            }
            let (mut inputs, mut targets) = (Vec::new(), Vec::new());
            for &off in chunk {
                let len = ((self.t + 1) as u64).min(n - off) as usize;
                let bytes = self.corpus.window(off, len);
                self.fill(&bytes, &mut inputs, &mut targets, self.t);
            }
            batches.push(Batch { inputs, targets, bsz: chunk.len(), t: self.t });
        }
        batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_are_shifted_by_one() {
        let c = Corpus::from_text("abcdefghijklmnopqrstuvwxyz");
        let mut b = Batcher::new(&c, 3, 8, 1);
        let batch = b.next_random();
        assert_eq!(batch.inputs.len(), 24);
        for r in 0..3 {
            for i in 0..7 {
                assert_eq!(batch.targets[r * 8 + i], batch.inputs[r * 8 + i + 1] as i32);
            }
        }
    }

    #[test]
    fn padding_is_masked_and_sep_is_eos() {
        let mut data = b"hi".to_vec();
        data.push(DOC_SEP);
        let c = Corpus::from_bytes(data);
        let mut b = Batcher::new(&c, 1, 6, 1);
        let batch = b.next_random();
        assert_eq!(batch.inputs[..2], [b'h' as u32, b'i' as u32]);
        assert_eq!(batch.targets[1], EOS as i32); // the model must learn to predict end-of-document
        assert_eq!(batch.targets[2], -1); // beyond the data: masked
        assert_eq!(batch.inputs[2], PAD);
    }

    #[test]
    fn split_and_streaming_agree() {
        let dir = std::env::temp_dir().join(format!("mkr_stream_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("c.bin");
        let data: Vec<u8> = (0..1000u32).map(|i| (i % 200 + 1) as u8).collect();
        std::fs::write(&p, &data).unwrap();
        let mem = Corpus::from_bytes(data.clone());
        let strm = Corpus::open_streaming(&p).unwrap();
        let (tm, vm) = mem.split(0.2);
        let (ts, vs) = strm.split(0.2);
        assert_eq!(tm.len(), 800);
        assert_eq!(vm.len(), 200);
        assert_eq!(tm.window(10, 50), ts.window(10, 50));
        assert_eq!(vm.window(0, 20), vs.window(0, 20));
        assert_eq!(vm.window(0, 1)[0], data[800]);
        let vb = Batcher::new(&vm, 2, 32, 0).sequential(100);
        assert!(!vb.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
