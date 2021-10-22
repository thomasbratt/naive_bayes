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
    hypotheses: Vec<H>,
    // probability P is: P(D|H) * P(H)
    probabilities: HashMap<D, Vec<(H, f64)>>,
}

impl<D: Copy + Eq + Hash, H: Copy + Eq + Hash> Classifier<D, H> {
    pub fn new(hypotheses: Vec<H>, probabilities: HashMap<D, Vec<(H, f64)>>) -> Self {
        Classifier {
            hypotheses,
            probabilities,
        }
    }

    pub fn classify(&self, data: Iter<D>) -> Results<H> {
        let mut results: HashMap<H, f64> = HashMap::default();

        // TODO: log sum exp
        for d in data {
            if let Some(hp) = self.probabilities.get(d) {
                for (h, p) in hp {
                    *results.entry(*h).or_insert(1.0) *= p;
                }
            }
        }

        // TODO: normalise relative probabilities

        // unstable:
        // results.try_insert(h, 0.0);
        for h in &self.hypotheses {
            results.entry(*h).or_insert(0.0);
        }

        Results::new(results)
    }
}
