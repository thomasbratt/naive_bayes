#[cfg(test)]
mod tests {
    use crate::classifier::Classifier;
    use crate::learner::Learner;

    #[test]
    fn adhoc() {
        let mut learner = Learner::default();
        learner
            .update_batch(&mut [['a'], ['b'], ['c'], ['a'], ['a']].iter(), "01")
            .update_batch(&mut [['b'], ['b'], ['c'], ['b'], ['a']].iter(), "02")
            .update_batch(&mut [['b'], ['c'], ['c'], ['b'], ['c']].iter(), "03");
        let classifier = learner.make_classifier();

        let actual = classifier.classify(&['a']);

        println!("learner   : {:?}", learner);
        println!("classifier: {:?}", classifier);
        println!("results   : {:?}", actual);

        assert_eq!(actual.best().unwrap().0, "01");
    }

    #[test]
    fn handles_empty_training_set() {
        let classifier: Classifier<char, &'static str, 1> = Learner::default().make_classifier();

        let actual = classifier.classify(&['a']);

        assert_eq!(actual.best(), Option::None);
        assert_eq!(actual.into_iter().count(), 0);
    }
}
