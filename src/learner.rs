use crate::classifier::Classifier;
use std::collections::HashMap;
use std::hash::Hash;
use std::slice::Iter;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Learner<D, H>
where
    D: Copy + Eq + Hash,
    H: Copy + Eq + Hash,
{
    count_hypotheses: HashMap<H, f64>,
    count_likelihoods: HashMap<(D, H), f64>,
    count_total: f64,
}

impl<D: Copy + Eq + Hash, H: Copy + Eq + Hash> Learner<D, H> {
    pub fn update_online(&mut self, datum: D, hypothesis: H) -> &mut Self {
        *self.count_hypotheses.entry(hypothesis).or_insert(0.0) += 1.0;
        *self
            .count_likelihoods
            .entry((datum, hypothesis))
            .or_insert(0.0) += 1.0;
        self.count_total += 1.0;
        self
    }

    pub fn update_batch(&mut self, data: Iter<D>, hypothesis: H) -> &mut Self {
        let count_h = self.count_hypotheses.entry(hypothesis).or_insert(0.0);
        for d in data {
            *count_h += 1.0;
            self.count_total += 1.0;
            *self
                .count_likelihoods
                .entry((*d, hypothesis))
                .or_insert(0.0) += 1.0;
        }
        self
    }

    pub fn make_classifier(&mut self) -> Classifier<D, H> {
        let log_priors: HashMap<H, f64> = self
            .count_hypotheses
            .iter()
            .map(|(h, c)| (*h, (*c / self.count_total).log2()))
            .collect();

        let mut log_likelihoods_hashmap: HashMap<D, HashMap<H, f64>> = HashMap::default();
        for ((d, h), c) in &self.count_likelihoods {
            log_likelihoods_hashmap
                .entry(*d)
                .or_insert_with(HashMap::default)
                .insert(*h, (*c / self.count_total).log2());
        }

        let log_likelihoods: HashMap<D, Vec<(H, f64)>> = log_likelihoods_hashmap
            .iter()
            .map(|(d, k)| {
                let v = k.iter().map(|x| (*x.0, *x.1)).collect();
                (*d, v)
            })
            .collect();

        Classifier::new(log_priors, log_likelihoods)
    }
}
