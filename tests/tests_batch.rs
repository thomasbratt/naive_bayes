#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use crate::classifier::Classifier;
    use crate::learner::Learner;

    #[test]
    fn classifies_test_case_01() {
        let classifier = learn_batch();

        let actual = classifier.classify(&['a']);

        assert_eq!(actual.best().unwrap().0, "01");
        assert_approx_eq!(actual.best().unwrap().1, 0.75, 0.1);
    }

    #[test]
    fn classifies_test_case_02() {
        let classifier = learn_batch();

        let actual = classifier.classify(&['b']);

        assert_eq!(actual.best().unwrap().0, "02");
        assert_approx_eq!(actual.best().unwrap().1, 0.5, 0.1);
    }

    #[test]
    fn classifies_test_case_03() {
        let classifier = learn_batch();

        let actual = classifier.classify(&['c']);

        assert_eq!(actual.best().unwrap().0, "03");
        assert_approx_eq!(actual.best().unwrap().1, 0.6, 0.1);
    }

    fn learn_batch() -> Classifier<char, &'static str, 1> {
        Learner::default()
            .update_batch(&mut [['a'], ['b'], ['c'], ['a'], ['a']].iter(), "01")
            .update_batch(&mut [['b'], ['b'], ['c'], ['b'], ['a']].iter(), "02")
            .update_batch(&mut [['b'], ['c'], ['c'], ['b'], ['c']].iter(), "03")
            .make_classifier()
    }
}
