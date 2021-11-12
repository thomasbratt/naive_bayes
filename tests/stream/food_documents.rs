use naive_bayes::StreamLearner;

// Test from: https://github.com/jackm321/Rust_Classifier/blob/master/tests/foods.rs
#[test]
fn food() {
    let mut learner: StreamLearner<&'static str, &'static str> = learn();
    let classifier = learner.make_classifier();
    let text = "salami pancetta beef ribs";

    let actual = classifier.classify(&mut text.split(' ').into_iter());

    assert_eq!(actual.best().unwrap().0, "meat");
}

fn learn() -> StreamLearner<&'static str, &'static str> {
    let mut learner: StreamLearner<&'static str, &'static str> = StreamLearner::default();
    learner
        .update(&mut "beetroot water spinach okra water chestnut ricebean pea catsear courgette summer purslane. water spinach arugula pea tatsoi aubergine spring onion bush tomato kale radicchio turnip chicory salsify pea sprouts fava bean. dandelion zucchini burdock yarrow chickpea dandelion sorrel courgette turnip greens tigernut soybean radish artichoke wattle seed endive groundnut broccoli arugula."
                    .split(' ').into_iter(),
                "veggie")
        .update(&mut "sirloin meatloaf ham hock sausage meatball tongue prosciutto picanha turkey ball tip pastrami. ribeye chicken sausage, ham hock landjaeger pork belly pancetta ball tip tenderloin leberkas shank shankle rump. cupim short ribs ground round biltong tenderloin ribeye drumstick landjaeger short loin doner chicken shoulder spare ribs fatback boudin. pork chop shank shoulder, t-bone beef ribs drumstick landjaeger meatball."
                    .split(' ').into_iter(),
                "meat")
        .update(&mut "pea horseradish azuki bean lettuce avocado asparagus okra. kohlrabi radish okra azuki bean corn fava bean mustard tigernut jã­cama green bean celtuce collard greens avocado quandong fennel gumbo black-eyed pea. grape silver beet watercress potato tigernut corn groundnut. chickweed okra pea winter purslane coriander yarrow sweet pepper radish garlic brussels sprout groundnut summer purslane earthnut pea tomato spring onion azuki bean gourd. gumbo kakadu plum komatsuna black-eyed pea green bean zucchini gourd winter purslane silver beet rock melon radish asparagus spinach."
                    .split(' ').into_iter(),
                "veggie")
        .update(&mut "sirloin porchetta drumstick, pastrami bresaola landjaeger turducken kevin ham capicola corned beef. pork cow capicola, pancetta turkey tri-tip doner ball tip salami. fatback pastrami rump pancetta landjaeger. doner porchetta meatloaf short ribs cow chuck jerky pork chop landjaeger picanha tail."
                    .split(' ').into_iter(),
                "meat")
        ;
    learner
}
