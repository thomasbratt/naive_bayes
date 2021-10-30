use crate::classifier::Classifier;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::Iterator;
use std::prelude::rust_2021::TryInto;

#[derive(Clone, Debug, PartialEq)]
pub struct Learner<D, H, const DS: usize>
where
    D: Copy + Debug + Eq + Hash,
    H: Copy + Debug + Eq + Hash,
{
    count_hypotheses: HashMap<H, f64>,
    count_joint: [HashMap<(D, H), f64>; DS],
    count_total: f64,
}

impl<D: Copy + Debug + Eq + Hash, H: Copy + Debug + Eq + Hash, const DS: usize> Default
    for Learner<D, H, DS>
{
    fn default() -> Self {
        Learner {
            count_hypotheses: HashMap::default(),
            count_joint: [(); DS].map(|_| HashMap::<(D, H), f64>::default()),
            count_total: 0.0,
        }
    }
}

impl<D: Copy + Debug + Eq + Hash, H: Copy + Debug + Eq + Hash, const DS: usize> Learner<D, H, DS> {
    /// Update the Learner with a single instance of training data for a single hypothesis.
    ///
    /// # Arguments
    ///
    /// * `data` - an array representing a single instance of training data
    /// * `hypothesis` - the target hypothesis/label/category/classification for the data.
    ///
    pub fn update(&mut self, data: &[D; DS], hypothesis: H) -> &mut Self {
        for (i, d) in data.iter().enumerate() {
            *self.count_joint[i].entry((*d, hypothesis)).or_insert(0.0) += 1.0;
        }
        *self.count_hypotheses.entry(hypothesis).or_insert(0.0) += 1.0;
        self.count_total += 1.0;
        self
    }

    /// Update the Learner with multiple instances of training data for a single hypothesis.
    ///
    /// # Arguments
    ///
    /// * `data` - an Iterator over arrays that each represent a single instance of training data
    /// * `hypothesis` - the single target hypothesis/label/category/classification for the data
    ///
    pub fn update_batch(
        &mut self,
        data: &mut dyn Iterator<Item = &[D; DS]>,
        hypothesis: H,
    ) -> &mut Self {
        let mut count = 0.0;
        for item in data {
            for (i, d) in item.iter().enumerate() {
                *self.count_joint[i].entry((*d, hypothesis)).or_insert(0.0) += 1.0;
            }
            // Retrieving the length can be expensive, for example when the data is being streamed.
            // This counter is cheap to maintain and can be processed without additional latency.
            count += 1.0;
        }
        // Do this lookup once instead of for each item in data.
        *self.count_hypotheses.entry(hypothesis).or_insert(0.0) += count;
        self.count_total += count;
        self
    }

    /// Make a classifier based on a snapshot of the current Learner's training.
    ///
    /// # Arguments
    ///
    /// * `data` - an Iterator over arrays that each represent a single instance of training data
    /// * `hypothesis` - the single target hypothesis/label/category/classification for the data
    ///
    /// # Return Value
    ///
    /// * `Classififer` type
    ///
    pub fn make_classifier(&mut self) -> Classifier<D, H, DS> {
        let log_priors: HashMap<H, f64> = self
            .count_hypotheses
            .iter()
            .map(|(h, c)| (*h, (*c / self.count_total).log2()))
            .collect();

        let log_likelihoods: [HashMap<D, Vec<(H, f64)>>; DS] = self
            .count_joint
            // Process each position in the input array separately.
            //
            // The 'naive' assumption of 'naive bayes' assumes that the probability at each position
            // is independent of the other positions.
            // Under this assumption, the classifier can estimate the probability of a specific
            // combination of data values at each array position by multiplying together the
            // probabilities at each individual position.
            .iter()
            .map(|dhc| {
                // Determine p(d|h) given count of |(d,h)| and count of |h|:
                //
                //      p(d|h) = |(d,h)| / |h|
                //
                // This probability is conditional on h, and log2 of this value is stored
                // precomputed for later use by the Classifier.
                //
                // To reduce storage, only store values log2(p(d|h)) that occur in the training
                // data. Zero values can be inferred later on, rather than stored.
                // Multiplication by zero values should also be avoided, as this will always result
                // in a final estimate of zero.
                // For this reason, the Classifier will assume that missing values of p(d|h) have
                // some small but non-zero estimate of the probability.
                //
                // To speed up classification, the mapping:
                //
                //      h -> p(d|h)
                //
                // is stored in another mapping that uses d as the key:
                //
                //      d -> h -> p(d|h)
                //
                // This requires only O(|i|) lookups, where |i| is the number of positions in the
                // input array.
                dhc.iter()
                    .fold(
                        HashMap::default(),
                        |mut acc: HashMap<D, HashMap<H, f64>>, ((d, h), c)| {
                            acc.entry(*d)
                                .or_insert_with(HashMap::default)
                                .insert(*h, (*c / self.count_hypotheses.get(h).unwrap()).log2());
                            acc
                        },
                    )
                    // To reduce storage again, convert the hashmap to a more compact Vec<>.
                    .iter()
                    .map(|(d, hp)| {
                        let v = hp.iter().map(|x| (*x.0, *x.1)).collect();
                        (*d, v)
                    })
                    .collect()
            })
            .collect::<Vec<HashMap<D, Vec<(H, f64)>>>>()
            .try_into()
            .unwrap();

        Classifier::new(log_priors, log_likelihoods)
    }
}
