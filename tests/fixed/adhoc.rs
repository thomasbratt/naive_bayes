use naive_bayes::StreamLearner;

#[test]
fn adhoc() {
    let mut learner: StreamLearner<char, &'static str> = learn();
    let classifier = learner.make_classifier();

    let actual = classifier.classify(&mut ['b'].iter());

    println!("learner   : {:?}", learner);
    println!("classifier: {:?}", classifier);
    println!("results   : {:?}", actual);

    // assert_eq!(actual.best().unwrap().0, "01");
}

fn learn() -> StreamLearner<char, &'static str> {
    let mut learner = StreamLearner::default();
    learner
        .update(&mut ['a'].iter(), "01")
        .update(&mut ['a'].iter(), "01")
        .update(&mut ['a'].iter(), "01")
        .update(&mut ['b'].iter(), "01")
        .update(&mut ['c'].iter(), "01")
        .update(&mut ['b'].iter(), "02")
        .update(&mut ['b'].iter(), "02")
        .update(&mut ['b'].iter(), "03")
        .update(&mut ['c'].iter(), "03")
        .update(&mut ['c'].iter(), "03");
    learner
}
