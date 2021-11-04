use assert_approx_eq::assert_approx_eq;
use naive_bayes::FixedClassifier;
use naive_bayes::FixedLearner;

#[test]
fn classifies_test_case_01() {
    let classifier = learn();

    let actual = classifier.classify(&['a']);

    assert_eq!(actual.best().unwrap().0, "01");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_test_case_02() {
    let classifier = learn();

    let actual = classifier.classify(&['b']);

    assert_eq!(actual.best().unwrap().0, "02");
    assert_approx_eq!(actual.best().unwrap().1, 0.5, 0.1);
}

#[test]
fn classifies_test_case_03() {
    let classifier = learn();

    let actual = classifier.classify(&['c']);

    assert_eq!(actual.best().unwrap().0, "03");
    assert_approx_eq!(actual.best().unwrap().1, 0.6, 0.1);
}

fn learn() -> FixedClassifier<char, &'static str, 1> {
    FixedLearner::default()
        .update(&['a'], "01")
        .update(&['a'], "01")
        .update(&['a'], "01")
        .update(&['b'], "01")
        .update(&['c'], "01")
        .update(&['b'], "02")
        .update(&['b'], "02")
        .update(&['b'], "03")
        .update(&['c'], "03")
        .update(&['c'], "03")
        .make_classifier()
}
