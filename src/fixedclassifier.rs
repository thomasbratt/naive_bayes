use super::posteriors::posteriors;
use super::results::Results;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

#[derive(Clone, Debug, PartialEq)]
pub struct FixedClassifier<D, H, const DS: usize>
where
    D: Copy + Eq + Hash,
    H: Copy + Eq + Hash,
{
    log_priors: HashMap<H, f64>,
    // probability P is: P(D|H) * P(H)
    log_likelihoods: [HashMap<D, Vec<(H, f64)>>; DS],
}

impl<D: Copy + Eq + Hash, H: Copy + Eq + Hash, const DS: usize> FixedClassifier<D, H, DS> {
    /// Create a new Classifier.
    ///
    /// # Arguments
    ///
    /// * `log_priors` - probability of hypothesis
    /// * `log_likelihoods` - probability of data given hypothesis, for each input array position
    ///
    pub(crate) fn new(
        log_priors: HashMap<H, f64>,
        log_likelihoods: [HashMap<D, Vec<(H, f64)>>; DS],
    ) -> Self {
        FixedClassifier {
            log_priors,
            log_likelihoods,
        }
    }

    /// Classify an unknown input.
    ///
    /// # Arguments
    ///
    /// * `data` - array of input data to classify
    ///
    /// # Return Value
    ///
    /// * `Results` type
    ///
    pub fn classify(&self, data: &[D; DS]) -> Results<H> {
        const LOG2_PLACEHOLDER_PROBABILITY: f64 = -(f64::MANTISSA_DIGITS as f64);

        // Accumulate product of likelihoods, grouped by hypothesis.
        let mut log_likelihoods: HashMap<H, f64> = HashMap::default();
        let placeholder = Vec::new();
        let all: HashSet<&H> = self.log_priors.keys().collect();
        for (i, d) in data.iter().enumerate() {
            let mut missing = all.clone();
            let found = self.log_likelihoods[i].get(d).unwrap_or(&placeholder);
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
