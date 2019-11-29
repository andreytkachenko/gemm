mod matrix;
mod sgemm;
mod aligned_alloc;
mod kernel;

extern crate blas;
extern crate openblas;

use self::matrix::{ConstMatrix, MutMatrix, ConstTransposedMatrix, MutTransposedMatrix};

pub unsafe fn sgemm(
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
    match (transa, transb, transc) {
        (false, false, false) => sgemm::sgemm(
            m, n, k, alpha, 
            ConstMatrix::new(a, lda),
            ConstMatrix::new(b, ldb),
            beta,
            MutMatrix::new(c, ldc)), 

        (false, false, true) => sgemm::sgemm(
            m, n, k, alpha, 
            ConstMatrix::new(a, lda),
            ConstMatrix::new(b, ldb),
            beta,
            MutTransposedMatrix::new(c, ldc)), 
        
        (false, true, false) => sgemm::sgemm(
            m, n, k, alpha, 
            ConstMatrix::new(a, lda),
            ConstTransposedMatrix::new(b, ldb),
            beta,
            MutMatrix::new(c, ldc)), 

        (false, true, true) => sgemm::sgemm(
            m, n, k, alpha, 
            ConstMatrix::new(a, lda),
            ConstTransposedMatrix::new(b, ldb),
            beta,
            MutTransposedMatrix::new(c, ldc)), 

        (true, false, false) => sgemm::sgemm(
            m, n, k, alpha, 
            ConstTransposedMatrix::new(a, lda),
            ConstMatrix::new(b, ldb),
            beta,
            MutMatrix::new(c, ldc)), 
        
        (true, false, true) => sgemm::sgemm(
            m, n, k, alpha, 
            ConstTransposedMatrix::new(a, lda),
            ConstMatrix::new(b, ldb),
            beta,
            MutTransposedMatrix::new(c, ldc)), 

                    
        (true, true, false) => sgemm::sgemm(
            m, n, k, alpha, 
            ConstTransposedMatrix::new(a, lda),
            ConstTransposedMatrix::new(b, ldb),
            beta,
            MutMatrix::new(c, ldc)), 

                    
        (true, true, true) => sgemm::sgemm(
            m, n, k, alpha, 
            ConstTransposedMatrix::new(a, lda),
            ConstTransposedMatrix::new(b, ldb),
            beta,
            MutTransposedMatrix::new(c, ldc)), 
    }
}





unsafe fn sgemm_ref_nn(
    m: usize,
    n: usize,
    k: usize,
    _alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    _beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    for j in 0..n {
        for i in 0..m {
            let mut ci = *c.add(i + j * ldc);
            for p in 0..k {
                ci += *a.add(i + p * lda) * *b.add(p + j * ldb);
            }
            *c.add(i + j * ldc) = ci;
        }
    }
}

unsafe fn sgemm_ref_nt(
    m: usize,
    n: usize,
    k: usize,
    _alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    _beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    for j in 0..n {
        for i in 0..m {
            let mut ci = *c.add(i + j * ldc);

            for p in 0..k {
                ci += *a.add(i + p * lda) * *b.add(j + p * ldb);
            }

            *c.add(i + j * ldc) = ci;
        }
    }
}

#[inline(never)]
pub fn gemm_nn(m: usize, n: usize, k: usize, alpha: f32, 
    a: &[f32], lda: usize, 
    b: &[f32], ldb: usize,
    _bata: f32, 
    c: &mut [f32], ldc: usize) {

    let mk = m * k;
    let nk = n * k;
    let mn = m * n;
    let a = &a[0..mk];
    let b = &b[0..nk];
    let c = &mut c[0..mn];

    for i_m in 0..m {
        for i_k in 0..k {
            let a_part = alpha * a[i_m * lda + i_k];
            
            for i_n in 0..n {
                c[i_m * ldc + i_n] += a_part * b[i_k * ldb + i_n];
            }
        }
    }
}


fn main2() {
    use rand::Rng;

    let mut rng = rand::thread_rng();

    const LEN: usize = 511;
    let (m, n, k) = (LEN, LEN, LEN);

    let mut a = vec![0.5; m * k];
    let mut a_t = vec![0.5; m * k];
    let mut b = vec![0.5; n * k];
    let mut b_t = vec![0.5; n * k];
    let mut c_nn = vec![0.0; m * n];
    let mut c_nt = vec![0.0; m * n];
    let mut c_tn = vec![0.0; m * n];
    let mut c_tt = vec![0.0; m * n];
    let mut cref = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..k {
            let v = rng.gen();
            a[i + j * m] = v;
            a_t[j + i * m] = v;
        }
    }

    for i in 0..n {
        for j in 0..k {
            let v = rng.gen();
            b[i + j * n] = v;
            b_t[j + i * n] = v;
        }
    }

    // let time = std::time::Instant::now();
    // unsafe {
    //     gemm_nn(
    //         LEN,
    //         LEN,
    //         LEN,
    //         1.0,
    //         a.as_slice(),
    //         LEN,
    //         b.as_slice(),
    //         LEN,
    //         1.0,
    //         cref1.as_mut_slice(),
    //         LEN,
    //     )
    // }

    // println!("Naive (mine) {}", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    unsafe {
        sgemm_ref_nn(
            m,
            n,
            k,

            1.0,
            a.as_ptr(),
            m,

            b.as_ptr(),
            k,

            1.0,
            cref.as_mut_ptr(),
            m,
        )
    }
    
    println!("Naive {}", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    unsafe {
        sgemm(
            false,
            false,
            false,
            m,
            n,
            k,

            1.0,
            a.as_ptr(),
            m,

            b.as_ptr(),
            k,

            1.0,
            c_nn.as_mut_ptr(),
            m,
        );
    }
    println!("[NN] Optimized {}", time.elapsed().as_millis());

    for i in 0..LEN {
        for j in 0..LEN {
            let (a, b) = (c_nn[i + j * LEN], cref[i + j * LEN]);
            assert!(feq(a, b), "a != b, a[{}]={}, b[{}]={}", i, a, j, b);
        }
    }

    let time = std::time::Instant::now();
    unsafe {
        sgemm(
            false,
            true,
            false,
            m,
            n,
            k,

            1.0,
            a.as_ptr(),
            m,

            b_t.as_ptr(),
            n,

            1.0,
            c_nt.as_mut_ptr(),
            m,
        );
    }

    println!("[NT] Optimized {}", time.elapsed().as_millis());

    for i in 0..LEN {
        for j in 0..LEN {
            let (a, b) = (c_nt[i + j * LEN], cref[i + j * LEN]);
            assert!(feq(a, b), "a != b, a[{}]={}, b[{}]={}", i, a, j, b);
        }
    }

    let time = std::time::Instant::now();
    unsafe {
        sgemm(
            true,
            false,
            false,
            m,
            n,
            k,

            1.0,
            a_t.as_ptr(),
            k,

            b.as_ptr(),
            k,

            1.0,
            c_tn.as_mut_ptr(),
            m,
        );
    }

    println!("[TN] Optimized {}", time.elapsed().as_millis());

    for i in 0..LEN {
        for j in 0..LEN {
            let (a, b) = (c_tn[i + j * LEN], cref[i + j * LEN]);
            assert!(feq(a, b), "a != b, a[{}]={}, b[{}]={}", i, a, j, b);
        }
    }

    let time = std::time::Instant::now();
    unsafe {
        sgemm(
            true,
            true,
            false,
            m,
            n,
            k,

            1.0,
            a_t.as_ptr(),
            k,

            b_t.as_ptr(),
            n,

            1.0,
            c_tt.as_mut_ptr(),
            m,
        );
    }
    println!("[TT] Optimized {}", time.elapsed().as_millis());


    for i in 0..LEN {
        for j in 0..LEN {
            let (a, b) = (c_tt[i + j * LEN], cref[i + j * LEN]);
            assert!(feq(a, b), "a != b, a[{}]={}, b[{}]={}", i, a, j, b);
        }
    }
}


fn main() {
    use rand::Rng;

    let mut rng = rand::thread_rng();

    const LEN: usize = 8192;
    let (m, n, k) = (LEN, LEN, LEN);

    let mut a = vec![0.5; m * k];
    let mut b = vec![0.5; n * k];
    let mut c = vec![0.0; m * n];
    let mut cref1 = vec![0.0; m * n];
    let mut cref = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..k {
            a[i + j * m] = rng.gen();
        }
    }

    for i in 0..n {
        for j in 0..k {
            b[i + j * n] = rng.gen();
        }
    }



    let time = std::time::Instant::now();
    unsafe {
        // blas::sgemm(
        //     b'N',
        //     b'N',
        //     m as i32,
        //     n as i32,
        //     k as i32,
        //     1.0,
        //     a.as_slice(),
        //     m as i32,
        //     b.as_slice(),
        //     k as i32,
        //     1.0,
        //     cref1.as_mut_slice(),
        //     m as i32,
        // );
        // gemm_nn(
        //     m,
        //     n,
        //     k,
        //     1.0,
        //     a.as_slice(),
        //     m,
        //     b.as_slice(),
        //     m,
        //     1.0,
        //     cref1.as_mut_slice(),
        //     m,
        // )
    }
    println!("Matrixmultiply (mine) {}", time.elapsed().as_millis());
    // println!("Naive (mine) {}", time.elapsed().as_millis());

    // let time = std::time::Instant::now();
    // unsafe {
    //     sgemm_ref_nn(
    //         m,
    //         n,
    //         k,

    //         1.0,
    //         a.as_ptr(),
    //         m,

    //         b.as_ptr(),
    //         k,

    //         1.0,
    //         c.as_mut_ptr(),
    //         m,
    //     )
    // }
    
    // println!("Naive {}", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    unsafe {
        sgemm(
            false,
            false,
            false,
            m,
            n,
            k,

            1.0,
            a.as_ptr(),
            m,

            b.as_ptr(),
            k,

            1.0,
            cref.as_mut_ptr(),
            m,
        );
    }

    println!("Optimized {}", time.elapsed().as_millis());


    // for i in 0..LEN {
    //     for j in 0..LEN {
    //         let (a, b) = (c[i + j * LEN], cref[i + j * LEN]);
    //         assert!(feq(a, b), "a != b, a[{}]={}, b[{}]={}", i, a, j, b);
    //     }
    // }
}

fn feq(a: f32, b: f32) -> bool {
    if a == b {
        true
    } else if a == 0.0 || b == 0.0 || (a.abs() + b.abs() < std::f32::MIN_POSITIVE) {
        (a - b).abs() < std::f32::EPSILON * 10.0 * std::f32::MIN_POSITIVE
    } else {
        (a - b).abs() / (a.abs() + b.abs()) < std::f32::EPSILON * 10.0
    }
}
