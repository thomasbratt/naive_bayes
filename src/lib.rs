#![doc = include_str!("../README.md")]

pub mod classifier;
pub mod learner;
pub mod results;

pub use self::classifier::Classifier;
pub use self::learner::Learner;
pub use self::results::Results;
