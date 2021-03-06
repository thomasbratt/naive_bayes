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
const TRAIN_PL_01: &'static str = "Artyku?? 2
    Ka??dy cz??owiek posiada wszystkie prawa i wolno??ci zawarte w niniejszej Deklaracji
    bez wzgl??du na jakiekolwiek r????nice rasy, koloru, p??ci, j??zyka, wyznania, pogl??d??w politycznych i
    innych, narodowo??ci, pochodzenia spo??ecznego, maj??tku, urodzenia lub jakiegokolwiek innego stanu.
    Nie wolno ponadto czyni?? ??adnej r????nicy w zale??no??ci od sytuacji politycznej, prawnej lub
    mi??dzynarodowej kraju lub obszaru, do kt??rego dana osoba przynale??y, bez wzgl??du na to, czy
    dany kraj lub obszar jest niepodleg??y, czy te?? podlega systemowi powiernictwa, nie rz??dzi
    si?? samodzielnie lub jest w jakikolwiek spos??b ograniczony w swej niepodleg??o??ci.";
const TRAIN_ES_01: &'static str = "Art??culo 2
    Toda persona tiene los derechos y libertades proclamados en esta Declaraci??n,
    sin distinci??n alguna de raza, color, sexo, idioma, religi??n, opini??n pol??tica
    o de cualquier otra ??ndole, origen nacional o social, posici??n econ??mica,
    nacimiento o cualquier otra condici??n.
    Adem??s, no se har?? distinci??n alguna fundada en la condici??n pol??tica, jur??dica
    o internacional del pa??s o territorio de cuya jurisdicci??n dependa una persona,
    tanto si se trata de un pa??s independiente, como de un territorio bajo
    administraci??n fiduciaria, no aut??nomo o sometido a cualquier otra limitaci??n
    de soberan??a.";

const TEST_EN_01: &'static str = "Article 1
    All human beings are born free and equal in dignity and rights. They are endowed with reason and
    conscience and should act towards one another in a spirit of brotherhood.";
const TEST_PL_01: &'static str = "Artyku?? 1
    Wszyscy ludzie rodz?? si?? wolni i r??wni pod wzgl??dem swej godno??ci i swych praw. S?? oni obdarzeni
    rozumem i sumieniem i powinni post??powa?? wobec innych w duchu braterstwa.";
const TEST_ES_01: &'static str = "Art??culo 1
    Todos los seres humanos nacen libres e iguales en dignidad y derechos y, dotados como est??n de
    raz??n y conciencia, deben comportarse fraternalmente los unos con los otros.";

fn learn() -> StreamClassifier<char, &'static str> {
    let mut learner = StreamLearner::<char, &'static str>::default();
    learner
        .update(&mut to_ascii(TRAIN_EN_01), "english")
        .update(&mut to_ascii(TRAIN_PL_01), "polish")
        .update(&mut to_ascii(TRAIN_ES_01), "spanish");
    learner.make_classifier()
}

fn to_ascii(original: &'static str) -> impl Iterator<Item = char> {
    original
        .chars()
        .into_iter()
        .filter(|x| x.is_ascii_alphabetic())
}
