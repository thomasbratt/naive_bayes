use crate::results::Results;
use std::collections::HashMap;
use std::hash::Hash;
use std::slice::Iter;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Classifier<D, H>
where
    D: Copy + Eq + Hash,
    H: Copy + Eq + Hash,
{
    log_priors: HashMap<H, f64>,
    // probability P is: P(D|H) * P(H)
    log_likelihoods: HashMap<D, Vec<(H, f64)>>,
}

const LOG2_MANTISSA_F64: f64 = -53.0;

impl<D: Copy + Eq + Hash, H: Copy + Eq + Hash> Classifier<D, H> {
    pub fn new(log_priors: HashMap<H, f64>, log_likelihoods: HashMap<D, Vec<(H, f64)>>) -> Self {
        Classifier {
            log_priors,
            log_likelihoods,
        }
    }

    pub fn classify(&self, data: Iter<D>) -> Results<H> {
        let mut accumulated_log_likelihoods: HashMap<H, f64> = HashMap::default();

        // Accumulate product of likelihoods, grouped by hypothesis.
        for d in data {
            if let Some(k) = self.log_likelihoods.get(d) {
                for (h, p) in k {
                    *accumulated_log_likelihoods.entry(*h).or_insert(0.0) += p;
                }
            }
        }

        // Work out resolution threshold for the number of hypotheses.
        let threshold = LOG2_MANTISSA_F64 - (self.log_priors.len() as f64).log2();
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
