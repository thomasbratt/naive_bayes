use num_traits::Float;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Clone, Debug, PartialEq)]
pub struct ClassificationResults<H, P>
where
    H: Copy + Eq + Hash,
    P: Float,
{
    values: HashMap<H, P>,
}

impl<H: Copy + Eq + Hash, P: Float> ClassificationResults<H, P> {
    pub fn new(values: HashMap<H, P>) -> Self {
        Self { values }
    }

    pub fn best(&self) -> Option<(H, P)> {
        self.values
            .iter()
            .max_by(|&lhs, &rhs| {
                if lhs.1 < rhs.1 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .map(|o| (*o.0, *o.1))
    }
}

impl<H: Copy + Eq + Hash, P: Float> IntoIterator for ClassificationResults<H, P> {
    type Item = (H, P);

    type IntoIter = std::collections::hash_map::IntoIter<H, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}
