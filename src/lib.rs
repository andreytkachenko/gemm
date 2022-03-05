mod aligned_alloc;
pub mod dim;
pub mod executor;
pub mod gemm;
pub mod kernel;
pub mod matrix;
mod sgemm;

#[cfg(test)]
extern crate blas;
#[cfg(test)]
extern crate openblas;
#[cfg(test)]
mod test;

pub use crate::sgemm::sgemm;
