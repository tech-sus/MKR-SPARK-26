//! Forward-only CPU inference. Per token the engine touches each weight once (matrix-vector
//! products) and keeps a constant-size state: one recurrent vector per layer plus a ring buffer of
//! `window` keys/values. Memory therefore does not grow with generated length.
pub mod engine;
pub mod generate;
pub mod qio;
pub mod sampler;

pub use engine::{Engine, Mat, State};
pub use generate::{generate, GenConfig, GenResult};
pub use sampler::Sampler;
