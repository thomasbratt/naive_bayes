#![doc = include_str!("../README.md")]

mod fixedclassifier;
mod fixedlearner;
mod results;
mod streamclassifier;
mod streamlearner;

mod likelihoods;
mod posteriors;

pub use fixedclassifier::FixedClassifier;
pub use fixedlearner::FixedLearner;
pub use results::Results;
pub use streamclassifier::StreamClassifier;
pub use streamlearner::StreamLearner;
