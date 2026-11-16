//! Three separate learning mechanisms (never conflated):
//!  A. Knowledge learning  — verified facts + sources into the knowledge DB (`knowledge`)
//!  B. Memory learning     — user-approved personal facts (`memory`)
//!  C. Neural learning     — reviewed text fine-tunes the weights, guarded by validation,
//!                           replay of anchor data and automatic rollback (`finetune`)
//! Searching the web never changes the network. Weights change only through `finetune`,
//! only on text a human approved, and only if held-out checks pass.
pub mod finetune;
pub mod knowledge;
pub mod memory;
pub mod queue;

pub use finetune::{finetune, rollback, FineTuneConfig, FineTuneReport};
pub use knowledge::{learn_from_verdict, LearnOutcome, LearnPolicy};
pub use memory::{memory_proposals, parse_remember_command};
pub use queue::{Candidate, CandidateStatus, Queue};
