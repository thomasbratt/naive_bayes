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
    count_hypothesis: HashMap<H, f64>,
    count_likelihood: HashMap<(D, H), f64>,
    count_total: f64,
}

impl<D: Copy + Eq + Hash, H: Copy + Eq + Hash> Learner<D, H> {
    pub fn update_online(&mut self, datum: D, hypothesis: H) -> &mut Self {
        *self.count_hypothesis.entry(hypothesis).or_insert(0.0) += 1.0;
        *self
            .count_likelihood
            .entry((datum, hypothesis))
            .or_insert(0.0) += 1.0;
        self.count_total += 1.0;
        self
    }

    pub fn update_batch(&mut self, data: Iter<D>, hypothesis: H) -> &mut Self {
        for d in data {
            self.update_online(*d, hypothesis);
        }
        self
    }

    pub fn make_classifier(&mut self) -> Classifier<D, H> {
        let priors: HashMap<H, f64> = self
            .count_hypothesis
            .iter()
            .map(|(h, c)| (*h, (*c / self.count_total).log2()))
            .collect();

        // TODO: product_of_conditionals
        let mut probabilities_as_map: HashMap<D, HashMap<H, f64>> = HashMap::default();
        for ((datum, hypothesis), count) in &self.count_likelihood {
            // TODO: remove prior_h
            let prior_h = *priors.get(hypothesis).unwrap();
            let p: f64 = prior_h + (*count / self.count_total).log2();
            probabilities_as_map
                .entry(*datum)
                .or_insert_with(HashMap::default)
                .insert(*hypothesis, p);
        }

        // TODO: product_of_conditionals
        // TODO: add prior_h back here
        let probabilities: HashMap<D, Vec<(H, f64)>> = probabilities_as_map
            .iter()
            .map(|(datum, hp)| {

                let v = hp.iter().map(|x| (*x.0, *x.1)).collect();
                (*datum, v)
            })
            .collect();

        let hypotheses: Vec<H> = self.count_hypothesis.keys().copied().collect();
        Classifier::new(hypotheses, probabilities)
    }
}
