//! SPARK Neural Architecture (SNA).
//!
//! Each block is pre-norm and has three sub-layers:
//!  1. SPARK-Scan: input-dependent gated linear recurrence (O(1) state per token at inference)
//!  2. Sliding-window causal attention with RoPE (optional; window = 0 disables it)
//!  3. SwiGLU feed-forward
//!
//! Established ingredients: RMSNorm, RoPE, SwiGLU, sliding-window attention, diagonal linear
//! recurrences (S4/LRU/Hawk/minGRU family), residual pre-norm blocks, AdamW.
//! Our own composition choices (not claimed as a breakthrough): the (1-a)-coupled gated scan with
//! multi-timescale decay initialisation feeding a silu output gate, placed *before* attention in the
//! same block so the window attention sees position/context-aware features, and a single shared
//! training/inference definition verified to match numerically (see tests).
pub mod config;
pub mod io;
pub mod model;

pub use config::Config;
pub use model::{BlockVars, Model};
