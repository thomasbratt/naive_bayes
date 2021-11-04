use crate::Results;
use std::collections::HashMap;
use std::hash::Hash;

/// Returns posterior probabilities given log prirors and likelihoods, grouped by hypotheses.
///
/// # Arguments
///
/// * `log_priors`:
/// * `log_likelihoods`:
///
/// returns: Results<H>
pub(crate) fn posteriors<D: Copy + Eq + Hash, H: Copy + Eq + Hash>(
    log_priors: &HashMap<H, f64>,
    log_likelihoods: &HashMap<H, f64>,
) -> Results<H> {
    const LOG2_MANTISSA_F64: f64 = -(f64::MANTISSA_DIGITS as f64);

    // Discard any values that would result in a sum that would underflow a double precision
    // floating point representation when exponentiated.
    //
    // As there are |H| log probabilities added together, each one must be greater than the
    // minimum representable value divided by |H| to guarantee that the sum does not underflow.
    let threshold = LOG2_MANTISSA_F64 - (log_priors.len() as f64).log2();

    // Max log probability.
    let max = log_likelihoods
        .iter()
        .map(|(_, x)| *x)
        .into_iter()
        .reduce(f64::max)
        .unwrap_or(0.0);

    // Multiply each accumulated likelihood of h by the prior of h.
    let relative_probabilities: HashMap<H, f64> = if log_likelihoods.is_empty() {
        HashMap::default()
    } else {
        log_likelihoods
            .iter()
            .map(|(h, log_likelihood)| (*h, log_likelihood + *log_priors.get(h).unwrap()))
            .map(|(h, x)| (h, x - max))
            .filter(|(_, x)| *x > threshold)
            .collect()
    };

    // Sum relative probabilities.
    let sum: f64 = relative_probabilities
        .iter()
        .map(|(_, p)| (*p).exp2())
        .sum();

    // Normalise relative probabilities and add any missing hypotheses.
    let posteriors: HashMap<H, f64> = log_priors
        .iter()
        .map(|(h, _)| {
            (
                *h,
                relative_probabilities
                    .get(h)
                    .map_or(0.0, |x| x.exp2() / sum),
            )
        })
        .collect();

    Results::new(posteriors)
}
