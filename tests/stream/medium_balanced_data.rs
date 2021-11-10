use assert_approx_eq::assert_approx_eq;
use naive_bayes::StreamClassifier;
use naive_bayes::StreamLearner;

#[test]
fn classifies_english_training_data_01() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(TRAIN_EN_01));

    assert_eq!(actual.best().unwrap().0, "english");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_english_test_data_01() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(TEST_EN_01));

    assert_eq!(actual.best().unwrap().0, "english");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_polish_training_data_01() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(TRAIN_PL_01));

    assert_eq!(actual.best().unwrap().0, "polish");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_polish_test_data_01() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(TEST_PL_01));

    assert_eq!(actual.best().unwrap().0, "polish");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_spanish_training_data_01() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(TRAIN_ES_01));

    assert_eq!(actual.best().unwrap().0, "spanish");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

#[test]
fn classifies_spanish_test_data_01() {
    let classifier = learn();

    let actual = classifier.classify(&mut to_ascii(TEST_ES_01));

    assert_eq!(actual.best().unwrap().0, "spanish");
    assert_approx_eq!(actual.best().unwrap().1, 1.0, 0.1);
}

const TRAIN_EN_01: &'static str = "Article 2
    Everyone is entitled to all the rights and freedoms set forth in this Declaration,
    without distinction of any kind, such as race, colour, sex, language, religion, political or
    other opinion, national or social origin, property, birth or other status. Furthermore, no
    distinction shall be made on the basis of the political, jurisdictional or international status
    of the country or territory to which a person belongs, whether it be independent, trust,
    non-self-governing or under any other limitation of sovereignty.";
const TRAIN_PL_01: &'static str = "Artykuł 2
    Każdy człowiek posiada wszystkie prawa i wolności zawarte w niniejszej Deklaracji
    bez względu na jakiekolwiek różnice rasy, koloru, płci, języka, wyznania, poglądów politycznych i
    innych, narodowości, pochodzenia społecznego, majątku, urodzenia lub jakiegokolwiek innego stanu.
    Nie wolno ponadto czynić żadnej różnicy w zależności od sytuacji politycznej, prawnej lub
    międzynarodowej kraju lub obszaru, do którego dana osoba przynależy, bez względu na to, czy
    dany kraj lub obszar jest niepodległy, czy też podlega systemowi powiernictwa, nie rządzi
    się samodzielnie lub jest w jakikolwiek sposób ograniczony w swej niepodległości.";
const TRAIN_ES_01: &'static str = "Artículo 2
    Toda persona tiene los derechos y libertades proclamados en esta Declaración,
    sin distinción alguna de raza, color, sexo, idioma, religión, opinión política
    o de cualquier otra índole, origen nacional o social, posición económica,
    nacimiento o cualquier otra condición.
    Además, no se hará distinción alguna fundada en la condición política, jurídica
    o internacional del país o territorio de cuya jurisdicción dependa una persona,
    tanto si se trata de un país independiente, como de un territorio bajo
    administración fiduciaria, no autónomo o sometido a cualquier otra limitación
    de soberanía.";

const TEST_EN_01: &'static str = "Article 1
    All human beings are born free and equal in dignity and rights. They are endowed with reason and
    conscience and should act towards one another in a spirit of brotherhood.";
const TEST_PL_01: &'static str = "Artykuł 1
    Wszyscy ludzie rodzą się wolni i równi pod względem swej godności i swych praw. Są oni obdarzeni
    rozumem i sumieniem i powinni postępować wobec innych w duchu braterstwa.";
const TEST_ES_01: &'static str = "Artículo 1
    Todos los seres humanos nacen libres e iguales en dignidad y derechos y, dotados como están de
    razón y conciencia, deben comportarse fraternalmente los unos con los otros.";

fn learn() -> StreamClassifier<char, &'static str> {
    let mut learner = StreamLearner::<char, &'static str>::default();
    learner
        .update(
            &mut to_ascii(TRAIN_EN_01),
            "english",
        )
        .update(
            &mut to_ascii(TRAIN_PL_01),
            "polish",
        )
        .update(
            &mut to_ascii(TRAIN_ES_01),
            "spanish",
        );
    learner.make_classifier()
}

fn to_ascii(original: &'static str) -> impl Iterator<Item = char> {
    original.chars().into_iter().filter(|x| x.is_ascii_alphabetic())
}
