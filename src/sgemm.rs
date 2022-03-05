use crate::dim::*;
use crate::executor::Executor;
use crate::kernel::avx::AvxKernel;
use crate::kernel::generic::GenericKernel;

pub unsafe fn sgemm<E: Executor>(
    e: &E,
    transa: bool,
    transb: bool,
    transc: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    crate::gemm::gemm::<E, f32, AvxKernel<f32, GenericKernel>, A16, A5>(
        e, transa, transb, transc, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    );
}
