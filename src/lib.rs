pub mod matrix;
pub mod gemm;
mod sgemm;
mod aligned_alloc;
pub mod kernel;
pub mod dim;
pub mod executor;

#[cfg(test)]
extern crate blas;
#[cfg(test)]
extern crate openblas;
#[cfg(test)]
mod test;


pub use crate::sgemm::sgemm;
