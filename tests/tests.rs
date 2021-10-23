#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use naive_bayes::classifier::Classifier;
    use naive_bayes::learner::Learner;

    #[test]
    fn adhoc() {
        let mut learner = Learner::default();
        learner
            .update_batch(['a', 'b', 'c', 'a', 'a'].iter(), "01")
            .update_batch(['b', 'b', 'c', 'b', 'a'].iter(), "02")
            .update_batch(['b', 'c', 'c', 'b', 'c'].iter(), "03");
        let classifier = learner.make_classifier();

        let actual = classifier.classify(['a'].iter());

        println!("learner   : {:?}", learner);
        println!("classifier: {:?}", classifier);
        println!("results   : {:?}", actual);

        assert_eq!(actual.best().unwrap().0, "01");
    }

    #[test]
    fn classifies_test_case_01() {
        let classifier = learn();

        let actual = classifier.classify(['a'].iter());

        assert_eq!(actual.best().unwrap().0, "01");
        assert_approx_eq!(actual.best().unwrap().1, 0.75, 0.1);
    }

    #[test]
    fn classifies_test_case_02() {
        let classifier = learn();

        let actual = classifier.classify(['b'].iter());

        assert_eq!(actual.best().unwrap().0, "02");
        assert_approx_eq!(actual.best().unwrap().1, 0.5, 0.1);
    }

    #[test]
    fn classifies_test_case_03() {
        let classifier = learn();

        let actual = classifier.classify(['c'].iter());

        assert_eq!(actual.best().unwrap().0, "03");
        assert_approx_eq!(actual.best().unwrap().1, 0.6, 0.1);
    }

    fn learn() -> Classifier<char, &'static str> {
        Learner::default()
            .update_batch(['a', 'b', 'c', 'a', 'a'].iter(), "01")
            .update_batch(['b', 'b', 'c', 'b', 'a'].iter(), "02")
            .update_batch(['b', 'c', 'c', 'b', 'c'].iter(), "03")
            .make_classifier()
    }
}
