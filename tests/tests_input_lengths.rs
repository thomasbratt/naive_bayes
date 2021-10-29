#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use naive_bayes::classifier::Classifier;
    use naive_bayes::learner::Learner;

    #[test]
    fn classifies_test_case_01() {
        let classifier = learn();

        let actual = classifier.classify(&['a', 'a', 'c']);

        assert_eq!(actual.best().unwrap().0, "01");
        assert_approx_eq!(actual.best().unwrap().1, 0.9, 0.1);
    }

    #[test]
    fn classifies_test_case_02() {
        let classifier = learn();

        let actual = classifier.classify(&['b', 'b', 'd']);

        assert_eq!(actual.best().unwrap().0, "02");
        assert_approx_eq!(actual.best().unwrap().1, 0.64, 0.1);
    }

    #[test]
    fn classifies_test_case_03() {
        let classifier = learn();

        let actual = classifier.classify(&['c', 'b', 'c']);

        assert_eq!(actual.best().unwrap().0, "03");
        assert_approx_eq!(actual.best().unwrap().1, 0.8, 0.1);
    }

    #[test]
    fn handles_empty_training_set() {
        let classifier: Classifier<char, &'static str, 3> = Learner::default().make_classifier();

        let actual = classifier.classify(&['a', 'a', 'c']);

        assert_eq!(actual.best(), Option::None);
        assert_eq!(actual.into_iter().count(), 0);
    }

    fn learn() -> Classifier<char, &'static str, 3> {
        Learner::default()
            .update(&['a', 'a', 'a'], "01")
            .update(&['b', 'b', 'b'], "01")
            .update(&['c', 'c', 'c'], "01")
            .update(&['a', 'a', 'a'], "01")
            .update(&['a', 'a', 'a'], "01")
            .update(&['b', 'b', 'b'], "02")
            .update(&['b', 'b', 'b'], "02")
            .update(&['c', 'c', 'c'], "02")
            .update(&['b', 'b', 'b'], "02")
            .update(&['a', 'a', 'a'], "02")
            .update(&['b', 'b', 'b'], "03")
            .update(&['c', 'c', 'c'], "03")
            .update(&['c', 'c', 'c'], "03")
            .update(&['b', 'b', 'b'], "03")
            .update(&['c', 'c', 'c'], "03")
            .make_classifier()
    }
}
