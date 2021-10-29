#[cfg(test)]
mod tests {
    use naive_bayes::learner::Learner;

    #[test]
    fn adhoc() {
        let mut learner = Learner::default();
        learner
            .update_batch(&mut [['a', 'a', 'c'], ['a', 'b', 'c']].iter(), "01")
            .update_batch(&mut [['b', 'b', 'c'], ['b', 'b', 'c']].iter(), "02")
            .update_batch(&mut [['c', 'c', 'c'], ['c', 'b', 'c']].iter(), "03");
        let classifier = learner.make_classifier();

        let actual = classifier.classify(&['a', 'a', 'c']);

        println!("learner   : {:?}", learner);
        println!("classifier: {:?}", classifier);
        println!("results   : {:?}", actual);

        assert_eq!(actual.best().unwrap().0, "01");
    }
}
