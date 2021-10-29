# naive_bayes

A Naive Bayes machine learning classifier.

Features
-------

* Developed as an investigation into capabilities and implementation characteristics
* Not recommended for production
* Written in the Rust programming language
* Supports sparse data that do not fit well into a vector (ie words)
* Has some mitigation for very small probabilities (floating point values)

Usage
-------

* Clone the repository (see below)
* Install Rust using rustup (https://rustup.rs/)
* Run `cargo test` or `cargo build`

TODO
-------

* test with input arrays with size greater than 1
* test with larger data volumnes
* better handling for missing entries
* write up documentation, both google docs and README.md

* refactor into sparse matrix format
* refactor into normal versus log probability operations
* support stream data (where size of input is unknown)


Implementation
-------

* The statistics of the training data associated with each class/hypothesis/label is used to predict
which class/hypothesis/label some new data of interest belongs to
* Bayesian statistics
* 'Naive' refers to independence simplification
* Sparse data
* Problem of underflow when using floating point multiplication. Log probabilities as a mitigation
* Normalising log probabilities
* Problem of probability values of zero

Alternatives
-------

Naive Bayes is a very well known technique.
There are many alternatives for Rust available through cargo: https://crates.io/search?q=naive%20bayes

License
-------

MIT permissive license. See MIT-LICENSE.txt for full license details.

Source Code Repository
----------------------

https://github.com/thomasbratt/naive_bayes
