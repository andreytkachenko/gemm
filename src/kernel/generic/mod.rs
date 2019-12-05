pub mod l3d;
pub mod l3s;

use crate::matrix::{Matrix, MatrixMut, MutMatrix};
use crate::kernel::{GemmKernel, GemmKernelSupNr, GemmKernelSupMr, GemmKernelSup};
use crate::dim::*;

pub struct GenericKernel;

impl GemmKernelSupNr<f32, A5> for GenericKernel {
    #[inline]
    unsafe fn sup_tr<A: Matrix<f32>, C: MatrixMut<f32>>(
        alpha: f32,
        a: A,
        pb: MutMatrix<f32>,
        beta: f32,
        c: C,
    ) {
        self::l3s::sgemm_sup_1x8(
            pb.stride,
            alpha,
            a,
            pb,
            beta,
            c)
    }
} 

impl GemmKernelSupMr<f32, A16> for GenericKernel {
    #[inline]
    unsafe fn sup_bl<B: Matrix<f32>, C: MatrixMut<f32>>(
        alpha: f32,
        pa: MutMatrix<f32>,
        b: B,
        beta: f32,
        c: C,
    ) {
        unimplemented!()
    }
}

impl GemmKernelSup<f32> for GenericKernel {
    #[inline]
    unsafe fn sup_br<A: Matrix<f32>, B: Matrix<f32>, C: MatrixMut<f32>>(
        k: usize,
        alpha: f32,
        a: A,
        b: B,
        beta: f32,
        c: C,
    ) {
        let mut elem = 0.0;

        for p in 0..k {
            elem += *a.row(p) * *b.col(p);
        }

        elem *= alpha;

        if beta != 0.0 {
            elem += beta * *c.ptr();
        }

        *c.ptr_mut() += elem;
    }
}

impl GemmKernel<f32, A16, A5> for GenericKernel {

    #[inline]
    unsafe fn pack_row_a<A: Matrix<f32>>(a: A, pa: MutMatrix<f32>) {
        if  a.is_transposed() {
            self::l3s::sgemm_pa_t(pa.stride, a.ptr(), a.stride(), pa.ptr_mut());
        } else {
            unimplemented!()
        }
    }

    #[inline]
    unsafe fn pack_row_b<B: Matrix<f32>>(b: B, pb: MutMatrix<f32>) {
        if  b.is_transposed() {
            self::l3s::sgemm_pb_t(pb.stride, b.ptr(), b.stride(), pb.ptr_mut());
        } else {
            self::l3s::sgemm_pb_x8(pb.stride, b.ptr(), b.stride(), pb.ptr_mut());
        }
    }

    #[inline]
    unsafe fn main_tl<C: MatrixMut<f32>>(
        alpha: f32,
        pa: MutMatrix<f32>,
        pb: MutMatrix<f32>,
        beta: f32,
        c: C,
    ) {
        unimplemented!()
    }
}