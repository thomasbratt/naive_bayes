#[cfg(test)]
mod tests {
    use naive_bayes::learner::Learner;

    #[test]
    fn adhoc() {
        let mut learner: Learner<char, &'static str, 1> = learn();
        let classifier = learner.make_classifier();

        let actual = classifier.classify(&['a']);

        println!("learner   : {:?}", learner);
        println!("classifier: {:?}", classifier);
        println!("results   : {:?}", actual);

        assert_eq!(actual.best().unwrap().0, "01");
    }

    fn learn() -> Learner<char, &'static str, 1> {
        let mut learner = Learner::default();

        learner
            .update(&['a'], "01")
            .update(&['a'], "01")
            .update(&['a'], "01")
            .update(&['b'], "01")
            .update(&['c'], "01")

            .update(&['b'], "02")
            .update(&['b'], "02")

            .update(&['b'], "03")
            .update(&['c'], "03")
            .update(&['c'], "03");

        learner
    }
}
