use std::collections::HashMap;
use std::hash::Hash;

// Determine p(d|h) given count of |(d,h)| and count of |h|:
//
//      p(d|h) = |(d,h)| / |h|
//
// This probability is conditional on h, and log2 of this value is precomputed and
// stored for later use by the Classifier.
//
// To reduce storage, only store non-zero values log2(p(d|h)) that are seen in the
// training data.
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
pub(crate) fn likelihoods<D: Copy + Eq + Hash, H: Copy + Eq + Hash>(
    count_hypotheses: &HashMap<H, f64>,
    count_joint: &HashMap<(D, H), f64>,
) -> HashMap<D, Vec<(H, f64)>> {
    count_joint
        .iter()
        .fold(
            HashMap::default(),
            |mut acc: HashMap<D, HashMap<H, f64>>, ((d, h), c)| {
                acc.entry(*d)
                    .or_insert_with(HashMap::default)
                    .insert(*h, (*c / count_hypotheses.get(h).unwrap()).log2());
                acc
            },
        )
        // Reduce storage again by converting the HashMap to a more compact Vec.
        .iter()
        .map(|(d, hp)| {
            let v = hp.iter().map(|x| (*x.0, *x.1)).collect();
            (*d, v)
        })
        .collect()
}
