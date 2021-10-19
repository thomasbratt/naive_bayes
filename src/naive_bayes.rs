use linear_algebra::matrix::Matrix;
use linear_algebra::vector::{dot, Vector};
use num_traits::{NumAssign, Zero};

pub fn learn<F, const COLUMNS: usize, const ROWS: usize>(
    data: Matrix<f64, COLUMNS, ROWS>,
    target: Vector<f64, ROWS>,
    progress: F,
) -> Vector<f64, COLUMNS>
    where
        F: Fn(i32) -> (),
{
    let mut weights = Vector::fill(1.0);
    // for i in 0..settings.iterations {
    //     let mut error_total = 0.0;
    //     for r in 0..ROWS {
    //
    //     }
    //
    //     progress(i);
    // }
    weights
}