use crate::posteriors::posteriors;
use crate::results::Results;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

#[derive(Clone, Debug, PartialEq)]
pub struct StreamClassifier<D, H>
where
    D: Copy + Eq + Hash,
    H: Copy + Eq + Hash,
{
    log_priors: HashMap<H, f64>,
    // probability P is: P(D|H) * P(H)
    log_likelihoods: HashMap<D, Vec<(H, f64)>>,
}

impl<D: Copy + Eq + Hash, H: Copy + Eq + Hash> StreamClassifier<D, H> {
    /// Create a new Classifier.
    ///
    /// # Arguments
    ///
    /// * `log_priors` - probability of hypothesis
    /// * `log_likelihoods` - probability of data given hypothesis, for each input array position
    ///
    pub(crate) fn new(
        log_priors: HashMap<H, f64>,
        log_likelihoods: HashMap<D, Vec<(H, f64)>>,
    ) -> Self {
        StreamClassifier {
            log_priors,
            log_likelihoods,
        }
    }

    /// Classify an unknown input.
    ///
    /// # Arguments
    ///
    /// * `stream` - a stream of data to classify for a single hypothesis.
    ///
    /// # Return Value
    ///
    /// * `Results` type
    ///
    pub fn classify(&self, stream: &mut dyn Iterator<Item = &D>) -> Results<H> {
        const LOG2_PLACEHOLDER_PROBABILITY: f64 = -(f64::MANTISSA_DIGITS as f64);

        // Accumulate product of likelihoods, grouped by hypothesis.
        let mut log_likelihoods: HashMap<H, f64> = HashMap::default();
        let placeholder = Vec::new();
        let all: HashSet<&H> = self.log_priors.keys().collect();
        for d in stream {
            let mut missing = all.clone();
            let found = self.log_likelihoods.get(d).unwrap_or(&placeholder);
            for (h, p) in found {
                *log_likelihoods.entry(*h).or_insert(0.0) += p;
                missing.remove(h);
            }
            for h in missing {
                *log_likelihoods.entry(*h).or_insert(0.0) += LOG2_PLACEHOLDER_PROBABILITY;
            }
        }

        posteriors::<D, H>(&self.log_priors, &log_likelihoods)
    }
}
