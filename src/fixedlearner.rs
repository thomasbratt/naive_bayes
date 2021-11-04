use crate::fixedclassifier::FixedClassifier;
use crate::likelihoods::likelihoods;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::Iterator;
use std::prelude::rust_2021::TryInto;

#[derive(Clone, Debug, PartialEq)]
pub struct FixedLearner<D, H, const DS: usize>
where
    D: Copy + Debug + Eq + Hash,
    H: Copy + Debug + Eq + Hash,
{
    count_hypotheses: HashMap<H, f64>,
    count_joint: [HashMap<(D, H), f64>; DS],
    count_total: f64,
}

impl<D: Copy + Debug + Eq + Hash, H: Copy + Debug + Eq + Hash, const DS: usize> Default
    for FixedLearner<D, H, DS>
{
    fn default() -> Self {
        FixedLearner {
            count_hypotheses: HashMap::default(),
            count_joint: [(); DS].map(|_| HashMap::<(D, H), f64>::default()),
            count_total: 0.0,
        }
    }
}

impl<D: Copy + Debug + Eq + Hash, H: Copy + Debug + Eq + Hash, const DS: usize>
    FixedLearner<D, H, DS>
{
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
    /// * `Classifier` type
    ///
    pub fn make_classifier(&mut self) -> FixedClassifier<D, H, DS> {
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
            // combination of data values in the array by multiplying together the probabilities at
            // each individual position.
            .iter()
            .map(|dhc| likelihoods(&self.count_hypotheses, dhc))
            .collect::<Vec<HashMap<D, Vec<(H, f64)>>>>()
            .try_into()
            .unwrap();

        FixedClassifier::new(log_priors, log_likelihoods)
    }
}
