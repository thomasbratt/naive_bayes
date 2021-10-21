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
    count_hypothesis: HashMap<H, u64>,
    count_likelihood: HashMap<(D, H), u64>,
    count_total: u64,
}

impl<D: Copy + Eq + Hash, H: Copy + Eq + Hash> Learner<D, H> {
    // TODO: map.entry(key).or_insert(default)
    pub fn update_online(&mut self, datum: D, hypothesis: H) -> &mut Self {
        match self.count_hypothesis.get_mut(&hypothesis) {
            Some(x) => {
                *x += 1;
            }
            None => {
                self.count_hypothesis.insert(hypothesis, 0);
            }
        }
        let key = (datum, hypothesis);
        match self.count_likelihood.get_mut(&key) {
            Some(x) => {
                *x += 1;
            }
            None => {
                self.count_likelihood.insert(key, 0);
            }
        }
        self.count_total += 1;
        self
    }

    pub fn update_batch(&mut self, data: Iter<D>, hypothesis: H) -> &mut Self {
        for d in data {
            self.update_online(*d, hypothesis);
        }
        self
    }

    pub fn make_classifier(&mut self) -> Classifier<D, H> {
        let mut probabilities: HashMap<D, HashMap<H, f64>> = HashMap::default();

        // TODO: try .map(/*to tuple*/).collect()
        let mut priors: HashMap<H, f64> = HashMap::default();
        for (h, c) in &self.count_hypothesis {
            priors.insert(*h, *c as f64 / self.count_total as f64);
        }

        // TODO: try .map(/*to tuple*/).collect()

        for ((datum, hypothesis), count) in &self.count_likelihood {
            let prior_h = *priors.get(&hypothesis).unwrap_or(&0.0);

            let p: f64 = prior_h * (*count as f64 / self.count_total as f64);

            // : HashMap<H, P>
            let hp = &mut *probabilities.entry(*datum).or_insert(HashMap::default());
            hp.insert(*hypothesis, p);
        }

        let mut results: HashMap<D, Vec<(H, f64)>> = HashMap::default();
        for (datum, hp) in probabilities {
            let v = hp.iter().map(|x| (*x.0, *x.1)).collect();
            results.insert(datum, v);
        }

        let h: Vec<H> = self.count_hypothesis.keys().map(|x| x.clone()).collect();
        Classifier::new(h, results)
    }
}
