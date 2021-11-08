use naive_bayes::StreamLearner;

#[test]
fn adhoc() {
    let mut learner: StreamLearner<char, &'static str> = learn();
    let classifier = learner.make_classifier();

    let actual = classifier.classify(&mut ['b'].into_iter());

    println!("learner   : {:?}", learner);
    println!("classifier: {:?}", classifier);
    println!("results   : {:?}", actual);

    // assert_eq!(actual.best().unwrap().0, "01");
}

fn learn() -> StreamLearner<char, &'static str> {
    let mut learner: StreamLearner<char, &'static str> = StreamLearner::default();
    learner
        .update(&mut ['a'].into_iter(), "01")
        .update(&mut ['a'].into_iter(), "01")
        .update(&mut ['a'].into_iter(), "01")
        .update(&mut ['b'].into_iter(), "01")
        .update(&mut ['c'].into_iter(), "01")
        .update(&mut ['b'].into_iter(), "02")
        .update(&mut ['b'].into_iter(), "02")
        .update(&mut ['b'].into_iter(), "03")
        .update(&mut ['c'].into_iter(), "03")
        .update(&mut ['c'].into_iter(), "03");
    learner
}
