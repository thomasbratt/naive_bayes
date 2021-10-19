#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use linear_algebra::matrix::Matrix;
    use linear_algebra::vector::Vector;

    #[test]
    fn adhoc() {
        let data = Matrix::from([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
        ]);
        let target = Vector::from([1.0, 3.0, 5.0, 7.0, 9.0, 11.0]);
        // let settings = Settings {
        //     iterations: 20,
        //     rate: 0.05,
        //     target_mean_error: 0.01,
        // };
        // let actual = regression::learn(data, target, settings, progress_silent);
        // assert_approx_eq!(actual[0], 1.0, 0.1);
        // assert_approx_eq!(actual[1], 2.0, 0.1);
    }
}