#![allow(dead_code)]

use super::fma::fmadd_ps;
use super::intrinsics::*;
use crate::kernel::params::single::NR;
use crate::matrix::{Matrix, MatrixMut, MutMatrix};

pub(crate) unsafe fn sgemm_sup_1x8<A: Matrix<f32>, C: MatrixMut<f32>>(
    k: usize,
    _alpha: f32,
    a: A,
    pb: MutMatrix<f32>,
    _beta: f32,
    c: C,
) {
    let mut c0_3 = _mm_setzero_ps();
    let mut c4 = 0.0f32;
    // let mut c5 = 0.0f32;
    // let mut c6 = 0.0f32;
    // let mut c7 = 0.0f32;

    let mut a = a;
    let mut pb = pb;

    for _ in 0..k {
        let a0 = *a.ptr();
        let a0_simd = _mm_broadcast_ss(&*a.ptr());

        c0_3 = fmadd_ps(_mm_loadu_ps(pb.ptr()), a0_simd, c0_3);
        c4 += *pb.col(4) * a0;
        // c5 += *pb.add(5) * a0;
        // c6 += *pb.add(6) * a0;
        // c7 += *pb.add(7) * a0;

        a.inc_row();
        pb.shift_col(NR);
    }

    // c0 *= alpha;
    // c1 *= alpha;
    // c2 *= alpha;
    // c3 *= alpha;
    // c4 *= alpha;
    // c5 *= alpha;
    // c6 *= alpha;
    // c7 *= alpha;

    let ccol0_3 = c.ptr_mut();
    let ccol4 = c.row_mut(4);
    // let ccol5 = c.add(ldc * 5);
    // let ccol6 = c.add(ldc * 6);
    // let ccol7 = c.add(ldc * 7);

    // if beta != 0.0 {
    //     c0 += beta * *ccol0;
    //     c1 += beta * *ccol1;
    //     c2 += beta * *ccol2;
    //     c3 += beta * *ccol3;
    //     c4 += beta * *ccol4;
    //     // c5 += beta * *ccol5;
    //     // c6 += beta * *ccol6;
    //     // c7 += beta * *ccol7;
    // }

    _mm_storeu_ps(ccol0_3, c0_3);
    *ccol4 = c4;
    // *ccol5 = c5;
    // *ccol6 = c6;
    // *ccol7 = c7;
}

pub(crate) unsafe fn sgemm_sup_1x8_t(
    k: usize,
    _alpha: f32,
    a: *const f32,
    _lda: usize,
    pb: *const f32,
    _beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    let mut c03 = _mm_setzero_ps();
    let mut c4 = 0.0f32;
    // let mut c5 = 0.0f32;
    // let mut c6 = 0.0f32;
    // let mut c7 = 0.0f32;

    let mut a = a;
    let mut pb = pb;

    for _ in 0..k {
        let a0 = *a;
        let a0_simd = _mm_set_ps1(a0);

        c03 = fmadd_ps(_mm_loadu_ps(pb), a0_simd, c03);
        c4 += *pb.add(4) * a0;
        // c5 += *pb.add(5) * a0;
        // c6 += *pb.add(6) * a0;
        // c7 += *pb.add(7) * a0;

        a = a.add(1);
        pb = pb.add(NR);
    }

    // c0 *= alpha;
    // c1 *= alpha;
    // c2 *= alpha;
    // c3 *= alpha;
    // c4 *= alpha;
    // c5 *= alpha;
    // c6 *= alpha;
    // c7 *= alpha;

    // let ccol0 = c;
    // let ccol1 = c.add(ldc);
    // let ccol2 = c.add(ldc * 2);
    // let ccol3 = c.add(ldc * 3);
    let ccol4 = c.add(ldc * 4);
    // let ccol5 = c.add(ldc * 5);
    // let ccol6 = c.add(ldc * 6);
    // let ccol7 = c.add(ldc * 7);

    // if beta != 0.0 {
    //     c0 += beta * *ccol0;
    //     c1 += beta * *ccol1;
    //     c2 += beta * *ccol2;
    //     c3 += beta * *ccol3;
    //     c4 += beta * *ccol4;
    //     // c5 += beta * *ccol5;
    //     // c6 += beta * *ccol6;
    //     // c7 += beta * *ccol7;
    // }

    _mm_storeu_ps(c, c03);
    // *ccol0 = c0;
    // *ccol1 = c1;
    // *ccol2 = c2;
    // *ccol3 = c3;
    *ccol4 = c4;
    // *ccol5 = c5;
    // *ccol6 = c6;
    // *ccol7 = c7;
}

pub(crate) unsafe fn sgemm_pb_x8(k: usize, b: *const f32, ldb: usize, pb: *mut f32) {
    let mut bcol0 = b;
    let mut bcol1 = b.add(ldb);
    let mut bcol2 = b.add(ldb * 2);
    let mut bcol3 = b.add(ldb * 3);
    let mut bcol4 = b.add(ldb * 4);
    // let mut bcol5 = b.add(ldb * 5);
    // let mut bcol6 = b.add(ldb * 6);
    // let mut bcol7 = b.add(ldb * 7);

    let mut pb = pb;

    for _ in 0..k {
        _mm_storeu_ps(pb, _mm_set_ps(*bcol3, *bcol2, *bcol1, *bcol0));

        // *pb = *bcol0;
        // *pb.add(1) = *bcol1;
        // *pb.add(2) = *bcol2;
        // *pb.add(3) = *bcol3;
        *pb.add(4) = *bcol4;
        // *pb.add(5) = *bcol5;
        // *pb.add(6) = *bcol6;
        // *pb.add(7) = *bcol7;

        bcol0 = bcol0.add(1);
        bcol1 = bcol1.add(1);
        bcol2 = bcol2.add(1);
        bcol3 = bcol3.add(1);
        bcol4 = bcol4.add(1);
        // bcol5 = bcol5.add(1);
        // bcol6 = bcol6.add(1);
        // bcol7 = bcol7.add(1);

        pb = pb.add(NR);
    }
}
