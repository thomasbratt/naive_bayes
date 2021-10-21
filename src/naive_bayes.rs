use crate::classification_results::ClassificationResults;
use linear_algebra::matrix::Matrix;
use linear_algebra::vector::{dot, Vector};
use num_traits::{Float, NumAssign, One, Zero};
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::slice::Iter;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NaiveBayesClassifier<C, D, H, P>
where
    C: NumAssign + One + Zero,
    D: Copy + Eq + Hash,
    H: Copy + Eq + Hash,
    P: Float,
{
    count_data: HashMap<D, C>,
    count_categories: HashMap<H, C>,
    _p: PhantomData<P>,
}

impl<C: NumAssign + One + Zero, D: Copy + Eq + Hash, H: Copy + Eq + Hash, P: Float>
    NaiveBayesClassifier<C, D, H, P>
{
    pub fn classify(&self, data: Iter<D>) -> ClassificationResults<H, P> {
        unimplemented!("classify");
        ClassificationResults::new(HashMap::default())
    }

    pub fn learn(&mut self, datum: D, hypothesis: H) -> &mut Self {
        match self.count_data.get_mut(&datum) {
            Some(x) => {
                *x += <C as One>::one();
            }
            None => {
                self.count_data.insert(datum, <C as Zero>::zero());
            }
        }
        match self.count_categories.get_mut(&hypothesis) {
            Some(x) => {
                *x += <C as One>::one();
            }
            None => {
                self.count_categories
                    .insert(hypothesis, <C as Zero>::zero());
            }
        }
        self
    }

    pub fn learn_batch(&mut self, data: Iter<D>, hypothesis: H) -> &mut Self {
        for d in data {
            self.learn(*d, hypothesis);
        }
        self
    }
}
