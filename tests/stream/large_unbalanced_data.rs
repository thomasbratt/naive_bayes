use assert_approx_eq::assert_approx_eq;
use naive_bayes::StreamClassifier;
use naive_bayes::StreamLearner;

#[test]
fn classifies_english_training_data_01() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(include_str!(
        "data/english/train_en_01.txt"
    )));

    assert_eq!(actual.best().unwrap().0, "english");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_english_training_data_02() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(include_str!(
        "data/english/train_en_02.txt"
    )));

    assert_eq!(actual.best().unwrap().0, "english");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_english_test_data_01() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(include_str!(
        "data/english/test_en_01.txt"
    )));

    assert_eq!(actual.best().unwrap().0, "english");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_english_test_data_02() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(include_str!(
        "data/english/test_en_02.txt"
    )));

    assert_eq!(actual.best().unwrap().0, "english");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_polish_training_data_01() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(include_str!(
        "data/polish/train_pl_01.txt"
    )));

    assert_eq!(actual.best().unwrap().0, "polish");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_polish_training_data_02() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(include_str!(
        "data/polish/train_pl_02.txt"
    )));

    assert_eq!(actual.best().unwrap().0, "polish");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_polish_test_data_01() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(include_str!(
        "data/polish/test_pl_01.txt"
    )));

    assert_eq!(actual.best().unwrap().0, "polish");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_polish_test_data_02() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(include_str!(
        "data/polish/test_pl_02.txt"
    )));

    assert_eq!(actual.best().unwrap().0, "polish");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_spanish_training_data_01() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(include_str!(
        "data/spanish/train_es_01.txt"
    )));

    assert_eq!(actual.best().unwrap().0, "spanish");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_spanish_training_data_02() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(include_str!(
        "data/spanish/train_es_02.txt"
    )));

    assert_eq!(actual.best().unwrap().0, "spanish");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_spanish_test_data_01() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(include_str!(
        "data/spanish/test_es_01.txt"
    )));

    assert_eq!(actual.best().unwrap().0, "spanish");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_spanish_test_data_02() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(include_str!(
        "data/spanish/test_es_02.txt"
    )));

    assert_eq!(actual.best().unwrap().0, "spanish");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

fn learn() -> StreamClassifier<char, &'static str> {
    let mut learner = StreamLearner::<char, &'static str>::default();
    learner
        .update(
            &mut to_ascii(include_str!("data/english/train_en_01.txt")),
            "english",
        )
        .update(
            &mut to_ascii(include_str!("data/english/train_en_02.txt")),
            "english",
        )
        .update(
            &mut to_ascii(include_str!("data/polish/train_pl_01.txt")),
            "polish",
        )
        .update(
            &mut to_ascii(include_str!("data/polish/train_pl_02.txt")),
            "polish",
        )
        .update(
            &mut to_ascii(include_str!("data/spanish/train_es_01.txt")),
            "spanish",
        )
        .update(
            &mut to_ascii(include_str!("data/spanish/train_es_02.txt")),
            "spanish",
        );

    learner.make_classifier()
}

fn to_ascii(original: &'static str) -> impl Iterator<Item = char> {
    original.chars().into_iter().filter(|x| x.is_ascii_alphabetic())
}
