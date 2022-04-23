# naive_bayes

A Naive Bayes machine learning classifier.

[![CircleCI](https://circleci.com/gh/thomasbratt/naive_bayes/tree/main.svg?style=svg)](https://circleci.com/gh/thomasbratt/naive_bayes/tree/main)

## Features

* Developed as an investigation into capabilities and implementation characteristics
* Written in the Rust programming language
* Supports sparse data that do not fit well into a vector (ie words in a document)
* Has mitigation for very small probabilities (floating point values)

## Usage

* Not recommended for production use. There is no package for this on crates.io
* Install Rust using rustup <https://rustup.rs/>
* Clone the repository (see below)
* Run `cargo test` or `cargo build`

## Example

Classify single characters as belonging to 1 of 3 categories.

```rust
    use assert_approx_eq::assert_approx_eq;
    use naive_bayes::FixedClassifier;
    use naive_bayes::FixedLearner;

    let classifier = FixedLearner::default()
        .update_batch(&mut [['a'], ['b'], ['c'], ['a'], ['a']].iter(), "01")
        .update_batch(&mut [['b'], ['b'], ['c'], ['b'], ['a']].iter(), "02")
        .update_batch(&mut [['b'], ['c'], ['c'], ['b'], ['c']].iter(), "03")
        .make_classifier();

    let actual = classifier.classify(&['a']);

    assert_eq!(actual.best().unwrap().0, "01");
    assert_approx_eq!(actual.best().unwrap().1, 0.75, 0.1);
```

## API

## Type Parameters

* `D: Copy + Debug + Eq + Hash` - type of data being classified, for example str or i64
* `const DS: usize` - the size of the array of data in a training or classification instance
* `H: Copy + Debug + Eq + Hash`  - the target hypothesis/label/category/classification of the data.
 
### Learner

*update* - updates the learner with a single training instance (an instance is a fixed length array)

```text
    pub fn update(&mut self, data: &[D; DS], hypothesis: H) -> &mut Self
```
*update_batch* - updates the learner with a batch of training instances  (an instance is a fixed length array)
associated with a single hypothesis.  

```text
    pub fn update_batch(&mut self, data: &mut dyn Iterator<Item = &[D; DS]>, hypothesis: H) -> &mut Self
```
### Classifier

*classify* - classifies a single data instance (one instance is a fixed length array of data values)

```text
    pub fn classify(&self, data: &[D; DS]) -> Results<H>
```
## Implementation

### Training

The Learner records the following counts, which are later converted into probabilities.

A count of the number of training instances for a given hypothesis/label/class:

```text
    count_hypotheses: HashMap<H, f64>
```
Each instance that is learnt or classified is represented as a fixed size array. So, for each separate array index in each training instance, a count is kept of the number of times each distinct data value occurred with a given hypothesis/label/class.

```text
    count_joint: [HashMap<(D, H), f64>; DS]
```
A count is kept of the number of training instances:

```text
    count_total: f64
```
### Classification

When a Classifier is required, the counts maintained by the Learner are converted into probabilities and a snapshot in time of these values is taken. Due to numerical issues, logarithms of these probabilities are used (see later sections for details).

The log probability of a specific hypothesis/label/class occurring in the training data:

```text
    log_priors : HashMap<H, f64>
```
For each position in each instance, the log probability of each a specific hypothesis/label/class occurring in the training data when a specific data values is also seen

```text
    log_likelihoods: [HashMap<D, Vec<(H, f64)>>; DS]
```
To classify an instance containing data D, find the probability of each H by multiplying the values associated with D and H. These can be found by lookups in the data structures log_priors and log_likelihoods. As discussed below, adding logarithms is used instead of directly multiplying values.

### Probability Issues

The 'naive' assumption of 'naive bayes' assumes that the probability at each position in the input array is
independent of the other positions. This means that the **Learner** processes each position in the input array
separately.
Also under this assumption, the **Classifier** can estimate the probability of a specific combination of data values
at each array position by multiplying together the probabilities at each individual position.

It is important to avoid multiplication by probability estimates of zero, as this always results in a combined
estimate of zero.
For this reason, the **Classifier** will assume that missing values of **p(d|h)** have some small but non-zero
estimate of the probability.
The **Learner** determines **p(d|h)** given a count of **|(d,h)|** and count of **|h|**:
```text
      p(d|h) = |(d,h)| / |h|
```
### Performance

* The probability above is conditional on **h** and **log2** of this value is stored precomputed for later use by the
  **Classifier**.
* To reduce storage, only **p(d|h)** that occur in the training data are stored.
  For this reason, and as noted above, the **Classifier** will assume that missing values of **p(d|h)** have some small
  but non-zero estimate of the probability.
* To speed up classification, the mapping **h->p(d|h)** is stored in another mapping that uses **d** as the key.
  This requires only **O(|i|)** lookups, where **|i|** is the number of positions in the input array.
  The result is:
```text
      d -> h -> p(d|h)

      HashMap<D, HashMap<H, f64>>
```
* To reduce storage again, the **Hashmap** is converted to a more compact **Vec<>**.
  The result is:
```text
      d -> h -> p(d|h)

      HashMap<D, Vec<(H, f64)>>>
```
* **update_batch** looks up and updates the count of **h** once per batch.

### Numerical Issues

#### Why multiplying many small floating point numbers is a problem ####

When a unique data value does not often occur in a much larger training data set, its corresponding probability will be very small. This can lead to numerical underflow when using floating point multiplication.

Taking each small number to be the same value v, then the value decreases exponentially with each number (say n times) included in the product:
```text
    value = v^n
```
The example below shows n=3 but a production system would have a much larger value for n and so would run out of floating point precision very quickly.
```text
    step1 = 0.001 * 0.001
    step1 = 0.000001
    
    step2 = 0.000001 * 0.001
    step2 = 0.000000001
    
    step3 = 0.000000001 * 0.001
    step3 = 0.000000000001
```
#### How to safely multiply many small floating point numbers ####

There is a homeomorphism between multiplying real numbers and adding their logarithms. Adding logarithms does not lead to numerical underflow in the same way.

In other words, converting floating point values to their logarithms and adding them, then converting back the result using exponentiation will preserve more precision.

#### How to normalise small floating point numbers ####

Normalisation in this context means multiplying and then dividing by their sum.
This is implemented as follows:

Work out the maximum precision for an IEEE double precision floating point value - this is 53 binary digits
mantissa = 2^-53.

Using logarithms again, divide the maximum precision by the number of values we are multiplying, this will give a threshold
```text
    For 3 values:
    |V|=3

    threshold = log⁡2(mantissa) - log⁡2(|V|)
    threshold = -53 -1.58496250072
    threshold = -54.58
```
Take logarithms of each value v in V, giving logv.
Discard any value logv that is less than threshold
Exponentiate the remaining values logv
Divide each exponentiated value by their sum, as normal.

#### How to find the precision of a floating point representation ####

Floating-point numbers have three components:
```text
    a sign bit
    an exponent
    a fraction (mantissa)
```
So that the final number is:
```text
    sign * 2^exponent * (1 + fraction)
```
Floating point numbers are stored with one less bit of fraction than they can use, because zeroes are represented
specially and all non-zero numbers have at least one non-zero binary bit.

Combining this, the digits of precision for a floating point number is:
```text
    32-bit float has 24 bits of precision (fraction) or approximately 7.22 decimal digits  
    64-bit double has 53 bits of precision (fraction) or approximately 15.95 decimal digits
```
TODO
-------

* test with larger data volumes
* document stream data (where size of input is unknown)
* document core algorithm, in google docs
* refactor into sparse matrix format
* refactor into normal versus log probability operations
* determine better value for LOG2_PLACEHOLDER_PROBABILITY

# References

<https://stackoverflow.com/questions/13542944/how-many-significant-digits-do-floats-and-doubles-have-in-java>

<https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability/66621#66621>

<https://www.cs.rhodes.edu/~kirlinp/courses/ai/f18/projects/proj3/naive-bayes-log-probs.pdf>

<https://en.wikipedia.org/wiki/Double-precision_floating-point_format>

# Alternatives


Naive Bayes is a very well known technique.
There are many alternatives for Rust available through cargo: <https://crates.io/search?q=naive%20bayes>

License
-------

MIT permissive license. See LICENSE.txt for full license details.

Source Code Repository
----------------------

<https://github.com/thomasbratt/naive_bayes>

