pub(crate) unsafe fn sgemm_sup_1x4(
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    pb: *const f32,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    let mut c0 = 0.0;
    let mut c1 = 0.0;
    let mut c2 = 0.0;
    let mut c3 = 0.0;

    let mut a = a;
    let mut pb = pb;

    for _ in 0..k {
        let a0 = *a;

        c0 += *pb * a0;
        c1 += *pb.add(1) * a0;
        c2 += *pb.add(2) * a0;
        c3 += *pb.add(3) * a0;

        a = a.add(lda);
        pb = pb.add(4);
    }

    c0 *= alpha;
    c1 *= alpha;
    c2 *= alpha;
    c3 *= alpha;

    let ccol0 = c;
    let ccol1 = c.add(ldc);
    let ccol2 = c.add(ldc * 2);
    let ccol3 = c.add(ldc * 3);

    if beta != 0.0 {
        c0 += beta * *ccol0;
        c1 += beta * *ccol1;
        c2 += beta * *ccol2;
        c3 += beta * *ccol3;
    }

    *ccol0 = c0;
    *ccol1 = c1;
    *ccol2 = c2;
    *ccol3 = c3;
}

pub(crate) unsafe fn sgemm_sup1_t(
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    pb: *const f32,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    let mut c0 = 0.0;
    let mut c1 = 0.0;
    let mut c2 = 0.0;
    let mut c3 = 0.0;

    let mut a = a;
    let mut pb = pb;

    for _ in 0..k {
        let a0 = *a;

        c0 += *pb * a0;
        c1 += *pb.add(1) * a0;
        c2 += *pb.add(2) * a0;
        c3 += *pb.add(3) * a0;

        a = a.add(1);
        pb = pb.add(4);
    }

    c0 *= alpha;
    c1 *= alpha;
    c2 *= alpha;
    c3 *= alpha;

    let ccol0 = c;
    let ccol1 = c.add(ldc);
    let ccol2 = c.add(ldc * 2);
    let ccol3 = c.add(ldc * 3);

    if beta != 0.0 {
        c0 += beta * *ccol0;
        c1 += beta * *ccol1;
        c2 += beta * *ccol2;
        c3 += beta * *ccol3;
    }

    *ccol0 = c0;
    *ccol1 = c1;
    *ccol2 = c2;
    *ccol3 = c3;
}

pub(crate) unsafe fn sgemm_pb_x4(k: usize, b: *const f32, ldb: usize, pb: *mut f32) {
    let mut bcol0 = b;
    let mut bcol1 = b.add(ldb);
    let mut bcol2 = b.add(ldb * 2);
    let mut bcol3 = b.add(ldb * 3);

    let mut pb = pb;

    for _ in 0..k {
        *pb = *bcol0;
        *pb.add(1) = *bcol1;
        *pb.add(2) = *bcol2;
        *pb.add(3) = *bcol3;

        bcol0 = bcol0.add(1);
        bcol1 = bcol1.add(1);
        bcol2 = bcol2.add(1);
        bcol3 = bcol3.add(1);
        pb = pb.add(4);
    }
}

// pub(crate) unsafe fn sgemm_pa_n(k: usize, a: *const f32, lda: usize, pa: *mut f32) {
//     use crate::kernel::params::single::MR;
//     let mut a = a;
//     let mut pa = pa;

//     for p in 0..k {
//         for j in 0..MR {
//             *pa.add(j * k + p) = *a.add(p * lda + j);
//         }
//     }
// }

pub(crate) unsafe fn sgemm_pb_t(k: usize, b: *const f32, ldb: usize, pb: *mut f32) {
    use crate::kernel::params::single::NR;
    let mut b = b;
    let mut pb = pb;

    for _ in 0..k {
        for j in 0..NR {
            *pb.add(j) = *b.add(j);
        }

        pb = pb.add(NR);
        b = b.add(ldb);
    }
}

pub(crate) unsafe fn sgemm_pa_t(k: usize, a: *const f32, lda: usize, pa: *mut f32) {
    use crate::kernel::params::single::MR;
    let mut a = a;
    let mut pa = pa;

    for _ in 0..k {
        for j in 0..MR {
            *pa.add(j) = *a.add(j * lda);
        }

        a = a.add(1);
        pa = pa.add(MR);
    }
}