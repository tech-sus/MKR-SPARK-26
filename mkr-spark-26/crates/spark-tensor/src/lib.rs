//! spark-tensor: dependency-light CPU tensor engine with reverse-mode autograd.
//! All tensors are dense row-major `f32` matrices (rows x cols). Sequence batches
//! are laid out as `[batch*time, features]`.
pub mod graph;
pub mod matmul;
pub mod rng;
pub mod rope;

pub use graph::{Graph, Tensor, Var};
pub use rng::Rng;
