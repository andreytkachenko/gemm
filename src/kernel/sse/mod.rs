mod fma;
// mod hsum;
mod intrinsics;
// pub mod l1d;
// pub mod l1s;
// pub mod l3d;
pub mod l3s;

use core::marker::PhantomData;
use crate::matrix::{Number, MutMatrix, Matrix, MatrixMut};
use crate::kernel::{GemmKernel, GemmKernelSupNr, GemmKernelSupMr, GemmKernelSup};
use crate::dim::*;


pub struct SseKernel<F: Number, I>(PhantomData<fn(F, I)>);

impl<I> GemmKernelSupNr<f32, A5> for SseKernel<f32, I> 
    where I: GemmKernelSupNr<f32, A5>
{
    #[inline]
    unsafe fn sup_tr<A: Matrix<f32>, C: MatrixMut<f32>>(
        alpha: f32,
        a: A,
        pb: MutMatrix<f32>,
        beta: f32,
        c: C,
    ) {
        I::sup_tr(alpha, a, pb, beta, c);
    }
} 

impl<I> GemmKernelSupMr<f32, A16> for SseKernel<f32, I> 
    where I: GemmKernelSupMr<f32, A16>
{
    #[inline]
    unsafe fn sup_bl<B: Matrix<f32>, C: MatrixMut<f32>>(
        alpha: f32,
        pa: MutMatrix<f32>,
        b: B,
        beta: f32,
        c: C,
    ) {
        I::sup_bl(alpha, pa, b, beta, c);
    }
}

impl<I> GemmKernelSup<f32> for SseKernel<f32, I> 
    where I: GemmKernelSup<f32>
{
    #[inline]
    unsafe fn sup_br<A: Matrix<f32>, B: Matrix<f32>, C: MatrixMut<f32>>(
        k: usize,
        alpha: f32,
        a: A,
        b: B,
        beta: f32,
        c: C,
    ) {
        I::sup_br(k, alpha, a, b, beta, c);
    }
}

impl<I> GemmKernel<f32, A16, A5> for SseKernel<f32, I> 
    where I: GemmKernel<f32, A16, A5>
{
    #[inline]
    unsafe fn pack_row_a<A: Matrix<f32>>(a: A, pa: MutMatrix<f32>, i: usize) {
        I::pack_row_a(a, pa, i);
    }

    #[inline]
    unsafe fn pack_row_b<B: Matrix<f32>>(b: B, pb: MutMatrix<f32>, j: usize) {
        I::pack_row_b(b, pb, j);
    }

    #[inline]
    unsafe fn main_tl<C: MatrixMut<f32>>(
        alpha: f32,
        pa: MutMatrix<f32>,
        pb: MutMatrix<f32>,
        beta: f32,
        c: C,
    ) {
        I::main_tl(alpha, pa, pb, beta, c);
    }
}