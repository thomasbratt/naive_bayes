use assert_approx_eq::assert_approx_eq;
use naive_bayes::StreamClassifier;
use naive_bayes::StreamLearner;

#[test]
fn classifies_test_case_01() {
    let classifier = learn();

    let actual = classifier.classify(&mut ['a'].iter());

    assert_eq!(actual.best().unwrap().0, "01");
    assert_approx_eq!(actual.best().unwrap().1, 0.75, 0.1);
}

#[test]
fn classifies_test_case_02() {
    let classifier = learn();

    let actual = classifier.classify(&mut ['b'].iter());

    assert_eq!(actual.best().unwrap().0, "02");
    assert_approx_eq!(actual.best().unwrap().1, 0.5, 0.1);
}

#[test]
fn classifies_test_case_03() {
    let classifier = learn();

    let actual = classifier.classify(&mut ['c'].iter());

    assert_eq!(actual.best().unwrap().0, "03");
    assert_approx_eq!(actual.best().unwrap().1, 0.6, 0.1);
}

fn learn() -> StreamClassifier<&'static char, &'static str> {
    StreamLearner::default()
        .update(&mut ['a', 'b', 'c', 'a', 'a'].iter(), "01")
        .update(&mut ['b', 'b', 'c', 'b', 'a'].iter(), "02")
        .update(&mut ['b', 'c', 'c', 'b', 'c'].iter(), "03")
        .make_classifier()
}
