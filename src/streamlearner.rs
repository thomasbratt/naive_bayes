use crate::likelihoods::likelihoods;
use crate::streamclassifier::StreamClassifier;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::Iterator;

#[derive(Clone, Debug, PartialEq)]
pub struct StreamLearner<D, H>
where
    D: Copy + Debug + Eq + Hash,
    H: Copy + Debug + Eq + Hash,
{
    count_hypotheses: HashMap<H, f64>,
    count_joint: HashMap<(D, H), f64>,
    count_total: f64,
}

impl<D: Copy + Debug + Eq + Hash, H: Copy + Debug + Eq + Hash> Default for StreamLearner<D, H> {
    fn default() -> Self {
        StreamLearner {
            count_hypotheses: HashMap::default(),
            count_joint: HashMap::default(),
            count_total: 0.0,
        }
    }
}

impl<D: Copy + Debug + Eq + Hash, H: Copy + Debug + Eq + Hash> StreamLearner<D, H> {
    /// Update the Learner with a stream of data for a single hypothesis.
    ///
    /// # Arguments
    ///
    /// * `stream` - a stream of training data for a single hypothesis.
    /// * `hypothesis` - the target hypothesis/label/category/classification for the data.
    ///
    pub fn update(&mut self, stream: &mut dyn Iterator<Item = &D>, hypothesis: H) -> &mut Self {
        let mut count = 0.0;
        for d in stream {
            *self.count_joint.entry((*d, hypothesis)).or_insert(0.0) += 1.0;
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
    /// * `data` - a stream of data representing a single hypothesis.
    /// * `hypothesis` - the single target hypothesis/label/category/classification for the data
    ///
    /// # Return Value
    ///
    /// * `StreamClassifier` type
    ///
    pub fn make_classifier(&mut self) -> StreamClassifier<D, H> {
        let log_priors: HashMap<H, f64> = self
            .count_hypotheses
            .iter()
            .map(|(h, c)| (*h, (*c / self.count_total).log2()))
            .collect();
        let log_likelihoods: HashMap<D, Vec<(H, f64)>> =
            likelihoods(&self.count_hypotheses, &self.count_joint);
        StreamClassifier::new(log_priors, log_likelihoods)
    }
}
