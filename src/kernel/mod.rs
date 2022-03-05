#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub mod avx;
pub mod generic;
pub mod sse;

use crate::dim::Dim;
use crate::matrix::{Matrix, MatrixMut, MutMatrix, Number};

pub mod params {
    pub mod single {
        pub const MC: usize = 128;
        pub const KC: usize = 256;
        pub const NC: usize = 8 * 1024;
        pub const MR: usize = 16;
        pub const NR: usize = 5;
    }
    pub mod double {
        pub const MC: usize = 256;
        pub const KC: usize = 512;
        pub const NC: usize = 4096;
        pub const MR: usize = 8;
        pub const NR: usize = 4;
    }
}

// +----------------------+
// |TL  :    :    :    |TR|
// |    :    :    :    |  |
// + - - - - - - - - - ---+
// |    :    :    :    |  |
// |    :    :    :    |  |
// + - - - - - - - - - ---+
// |    :    :    :    |  |
// |    :    :    :    |  |
// +----------------------+
// |BL  |    |    |    |BR|
// +----------------------+

pub trait GemmKernelSup<F: Number> {
    unsafe fn sup_br<A: Matrix<F>, B: Matrix<F>, C: MatrixMut<F>>(
        k: usize,
        alpha: F,
        a: A,
        b: B,
        beta: F,
        c: C,
    );
}

pub trait GemmKernelSupMr<F: Number, MR: Dim> {
    unsafe fn sup_bl<B: Matrix<F>, C: MatrixMut<F>>(
        alpha: F,
        pa: MutMatrix<F>,
        b: B,
        beta: F,
        c: C,
    );
}

pub trait GemmKernelSupNr<F: Number, NR: Dim> {
    unsafe fn sup_tr<A: Matrix<F>, C: MatrixMut<F>>(
        alpha: F,
        a: A,
        pb: MutMatrix<F>,
        beta: F,
        c: C,
    );
}

pub trait GemmKernel<F: Number, MR: Dim, NR: Dim>:
    GemmKernelSupMr<F, MR> + GemmKernelSupNr<F, NR> + GemmKernelSup<F>
{
    unsafe fn pack_row_a<A: Matrix<F>>(a: A, pa: MutMatrix<F>);
    unsafe fn pack_row_b<B: Matrix<F>>(b: B, pb: MutMatrix<F>);

    unsafe fn main_tl<C: MatrixMut<F>>(alpha: F, pa: MutMatrix<F>, pb: MutMatrix<F>, beta: F, c: C);
}
