#[cfg(test)]
mod tests {
    use crate::naive_bayes::NaiveBayesClassifier;
    use assert_approx_eq::assert_approx_eq;
    use linear_algebra::matrix::Matrix;
    use linear_algebra::vector::Vector;

    #[test]
    fn adhoc() {
        let mut bayes: NaiveBayesClassifier<u64, char, &'static str, f64> =
            NaiveBayesClassifier::default();
        bayes
            .learn_batch(['a', 'b', 'c', 'a', 'a'].iter(), "01")
            .learn_batch(['b', 'b', 'c', 'b', 'a'].iter(), "02")
            .learn_batch(['b', 'c', 'c', 'b', 'c'].iter(), "03");

        let actual = bayes.classify(['a'].iter());

        assert_eq!(actual.best().unwrap().0, "01");
    }
}
