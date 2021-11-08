use naive_bayes::FixedLearner;

#[test]
fn adhoc() {
    let mut learner: FixedLearner<char, &'static str, 1> = learn();
    let classifier = learner.make_classifier();

    let actual = classifier.classify(&['b']);

    println!("learner   : {:?}", learner);
    println!("classifier: {:?}", classifier);
    println!("results   : {:?}", actual);

    // assert_eq!(actual.best().unwrap().0, "01");
}

fn learn() -> FixedLearner<char, &'static str, 1> {
    let mut learner = FixedLearner::default();
    learner
        .update(&['a'], "01")
        .update(&['b'], "01")
        .update(&['c'], "01")
        .update(&['a'], "01")
        .update(&['a'], "01")
        .update(&['b'], "02")
        .update(&['b'], "02")
        .update(&['c'], "02")
        .update(&['b'], "02")
        .update(&['a'], "02")
        .update(&['b'], "03")
        .update(&['c'], "03")
        .update(&['c'], "03")
        .update(&['b'], "03")
        .update(&['c'], "03");
    learner
}
