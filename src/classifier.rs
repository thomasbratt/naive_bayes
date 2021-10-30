use crate::results::Results;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

#[derive(Clone, Debug, PartialEq)]
pub struct Classifier<D, H, const DS: usize>
where
    D: Copy + Eq + Hash,
    H: Copy + Eq + Hash,
{
    log_priors: HashMap<H, f64>,
    // probability P is: P(D|H) * P(H)
    log_likelihoods: [HashMap<D, Vec<(H, f64)>>; DS],
}

const LOG2_MANTISSA_F64: f64 = -53.0;
const LOG2_PLACEHOLDER_PROBABILITY: f64 = -53.0;

impl<D: Copy + Eq + Hash, H: Copy + Eq + Hash, const DS: usize> Classifier<D, H, DS> {
    /// Create a new Classifier.
    ///
    /// # Arguments
    ///
    /// * `log_priors` - probability of hypothesis
    /// * `log_likelihoods` - probability of data given hypothesis, for each input array position
    ///
    pub fn new(
        log_priors: HashMap<H, f64>,
        log_likelihoods: [HashMap<D, Vec<(H, f64)>>; DS],
    ) -> Self {
        Classifier {
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
        let mut accumulated_log_likelihoods: HashMap<H, f64> = HashMap::default();

        // Accumulate product of likelihoods, grouped by hypothesis.
        let placeholder = Vec::new();
        let all: HashSet<&H> = self.log_priors.keys().collect();
        for (i, d) in data.iter().enumerate() {
            let mut missing = all.clone();
            let found = self.log_likelihoods[i].get(d).unwrap_or(&placeholder);
            for (h, p) in found {
                *accumulated_log_likelihoods.entry(*h).or_insert(0.0) += p;
                missing.remove(h);
            }
            for h in missing {
                *accumulated_log_likelihoods.entry(*h).or_insert(0.0) +=
                    LOG2_PLACEHOLDER_PROBABILITY;
            }
        }

        // Resolution threshold for the number of hypotheses.
        let threshold = LOG2_MANTISSA_F64 - (self.log_priors.len() as f64).log2();

        // Max log probability.
        let max = accumulated_log_likelihoods
            .iter()
            .map(|(_, x)| *x)
            .into_iter()
            .reduce(f64::max)
            .unwrap_or(0.0);

        // Multiply each accumulated likelihood of h by the prior of h.

        let relative_probabilities: HashMap<H, f64> = if accumulated_log_likelihoods.is_empty() {
            HashMap::default()
        } else {
            accumulated_log_likelihoods
                .iter()
                .map(|(h, log_likelihood)| (*h, log_likelihood + *self.log_priors.get(h).unwrap()))
                .map(|(h, x)| (h, x - max))
                .filter(|(_, x)| *x > threshold)
                .collect()
        };

        // Sum relative probabilities.
        let sum: f64 = relative_probabilities
            .iter()
            .map(|(_, p)| (*p).exp2())
            .sum();

        // Normalise relative probabilities and add any missing hypotheses.
        let posteriors: HashMap<H, f64> = self
            .log_priors
            .iter()
            .map(|(h, _)| {
                (
                    *h,
                    relative_probabilities
                        .get(h)
                        .map_or(0.0, |x| x.exp2() / sum),
                )
            })
            .collect();

        Results::new(posteriors)
    }
}
