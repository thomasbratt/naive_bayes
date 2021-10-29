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
    pub fn update(&mut self, data: &[D; DS], hypothesis: H) -> &mut Self {
        for (i, d) in data.iter().enumerate() {
            *self.count_joint[i].entry((*d, hypothesis)).or_insert(0.0) += 1.0;
        }
        *self.count_hypotheses.entry(hypothesis).or_insert(0.0) += 1.0;
        self.count_total += 1.0;
        self
    }

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
        // This is an expensive lookup, so do this once instead of once for each item in data.
        *self.count_hypotheses.entry(hypothesis).or_insert(0.0) += count;
        self.count_total += count;
        self
    }

    pub fn make_classifier(&mut self) -> Classifier<D, H, DS> {
        let log_priors: HashMap<H, f64> = self
            .count_hypotheses
            .iter()
            .map(|(h, c)| (*h, (*c / self.count_total).log2()))
            .collect();

        let log_likelihoods: [HashMap<D, Vec<(H, f64)>>; DS] = self
            .count_joint
            .iter()
            .map(|dhc| {
                dhc.iter()
                    // from  {(d,h): c}, |h|
                    // to    {d:{h:log2(c/|h|)}}
                    .fold(
                        HashMap::default(),
                        |mut acc: HashMap<D, HashMap<H, f64>>, ((d, h), c)| {
                            acc.entry(*d)
                                .or_insert_with(HashMap::default)
                                .insert(*h, (*c / self.count_hypotheses.get(h).unwrap()).log2());
                            acc
                        },
                    )
                    // Same as above but convert the HashMap<> to a more compact Vec<>
                    // from  {d:{h:p}}
                    // to    {d:[(h,p)]}
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
