use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Clone, Debug, PartialEq)]
pub struct Results<H>
where
    H: Copy + Eq + Hash,
{
    values: HashMap<H, f64>,
}

impl<H: Copy + Eq + Hash> Results<H> {
    /// Create a struct to hold classification results.
    ///
    /// # Arguments
    ///
    /// * `values` - posterior probability of each hypothesis
    ///
    pub(crate) fn new(values: HashMap<H, f64>) -> Self {
        Self { values }
    }

    /// Return the hypothesis with the highest posterior probability.
    ///
    /// # Return Value
    ///
    /// * `Option::Empty() - no results.
    /// * `Option::Some((H, f64))` - the hypothesis and its posterior probability.
    ///
    pub fn best(&self) -> Option<(H, f64)> {
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

impl<H: Copy + Eq + Hash> IntoIterator for Results<H> {
    type Item = (H, f64);

    type IntoIter = std::collections::hash_map::IntoIter<H, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}
