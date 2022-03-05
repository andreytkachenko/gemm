const M_LEN: usize = 17;
const N_LEN: usize = 6;
const K_LEN: usize = 4;

fn make_matrices() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let _rng = rand::thread_rng();

    let (m, n, k) = (M_LEN, N_LEN, K_LEN);

    let mut a = vec![0.0; m * k];
    let mut a_t = vec![0.0; m * k];

    let mut b = vec![0.0; n * k];
    let mut b_t = vec![0.0; n * k];

    let mut counter = 0;
    for row in 0..k {
        for col in 0..m {
            let v = counter as f32;
            a[row * m + col] = v;
            a_t[col * k + row] = v;
            counter += 1;
        }
    }

    let mut counter = 0;
    for row in 0..n {
        for col in 0..k {
            let v = counter as f32;
            b[row * k + col] = v;
            b_t[col * n + row] = v;
            counter += 1;
        }
    }

    (a, a_t, b, b_t)
}

unsafe fn sgemm_ref(
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
    for i_m in 0..m {
        for i_k in 0..k {
            let a_part = *a.add(i_m * lda + i_k);

            for i_n in 0..n {
                *c.add(i_m * ldc + i_n) += a_part * *b.add(i_k * ldb + i_n);
            }
        }
    }

    // for j in 0..n {
    //     for i in 0..m {
    //         let mut ci = *c.add(i + j * ldc);

    //         for p in 0..k {
    //             ci += *a.add(i + p * lda) * *b.add(p + j * ldb);
    //         }

    //         *c.add(i + j * ldc) = ci;
    //     }
    // }
}

#[test]
fn test_sgemm_nn() {
    let (m, n, k) = (M_LEN, N_LEN, K_LEN);
    let (a, _, b, _) = make_matrices();

    let mut c = vec![0.0; m * n];
    let mut cref = vec![0.0; m * n];

    for row in 0..N_LEN {
        for col in 0..K_LEN {
            let index = row * K_LEN + col;

            print!("{:>5} ", b[index]);
        }
        println!();
    }

    println!();

    unsafe {
        blas::sgemm(
            b'N',
            b'N',
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_slice(),
            m as i32,
            b.as_slice(),
            k as i32,
            0.0,
            cref.as_mut_slice(),
            m as i32,
        )
    }

    unsafe {
        crate::sgemm(
            &crate::executor::DefaultExecutor,
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
            0.0,
            c.as_mut_ptr(),
            m,
        );
    }

    for row in 0..N_LEN {
        for col in 0..M_LEN {
            let index = row * M_LEN + col;

            print!("{:>5} ", c[index]);
        }
        println!();
    }

    println!();

    for row in 0..N_LEN {
        for col in 0..M_LEN {
            let index = row * M_LEN + col;

            print!("{:>5} ", cref[index]);
        }
        println!();
    }

    for row in 0..N_LEN {
        for col in 0..M_LEN {
            let index = row * M_LEN + col;

            let (a, b) = (c[index], cref[index]);

            assert!(
                feq(a, b),
                "a != b, a[{}, {}]={}, b[{}, {}]={}",
                row,
                col,
                a,
                row,
                col,
                b
            );
        }
    }
}

#[test]
fn test_sgemm_nt() {
    let (m, n, k) = (M_LEN, N_LEN, K_LEN);
    let (a, _, _b, b_t) = make_matrices();

    let mut c = vec![0.0; m * n];
    let mut cref = vec![0.0; m * n];

    unsafe {
        blas::sgemm(
            b'N',
            b'T',
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_slice(),
            m as i32,
            b_t.as_slice(),
            n as i32,
            0.0,
            cref.as_mut_slice(),
            m as i32,
        )
    }

    unsafe {
        crate::sgemm(
            &crate::executor::DefaultExecutor,
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
            0.0,
            c.as_mut_ptr(),
            m,
        );
    }

    for row in 0..N_LEN {
        for col in 0..M_LEN {
            let index = row * M_LEN + col;
            let (a, b) = (c[index], cref[index]);
            assert!(
                feq(a, b),
                "a != b, a[{}, {}]={}, b[{}, {}]={}",
                row,
                col,
                a,
                row,
                col,
                b
            );
        }
    }
}

#[test]
fn test_sgemm_tn() {
    let (m, n, k) = (M_LEN, N_LEN, K_LEN);
    let (_a, a_t, b, _) = make_matrices();

    let mut c = vec![0.0; m * n];
    let mut cref = vec![0.0; m * n];

    unsafe {
        blas::sgemm(
            b'T',
            b'N',
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a_t.as_slice(),
            k as i32,
            b.as_slice(),
            k as i32,
            0.0,
            cref.as_mut_slice(),
            m as i32,
        )
    }

    unsafe {
        crate::sgemm(
            &crate::executor::DefaultExecutor,
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
            0.0,
            c.as_mut_ptr(),
            m,
        );
    }

    for row in 0..N_LEN {
        for col in 0..M_LEN {
            let index = row * M_LEN + col;
            let (a, b) = (c[index], cref[index]);
            assert!(
                feq(a, b),
                "a != b, a[{}, {}]={}, b[{}, {}]={}",
                row,
                col,
                a,
                row,
                col,
                b
            );
        }
    }
}

#[test]
fn test_sgemm_tt() {
    let (m, n, k) = (M_LEN, N_LEN, K_LEN);
    let (_a, a_t, _b, b_t) = make_matrices();

    let mut c = vec![0.0; m * n];
    let mut cref = vec![0.0; m * n];

    unsafe {
        blas::sgemm(
            b'T',
            b'T',
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a_t.as_slice(),
            k as i32,
            b_t.as_slice(),
            n as i32,
            0.0,
            cref.as_mut_slice(),
            m as i32,
        )
    }

    unsafe {
        crate::sgemm(
            &crate::executor::DefaultExecutor,
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
            0.0,
            c.as_mut_ptr(),
            m,
        );
    }

    for row in 0..N_LEN {
        for col in 0..M_LEN {
            let index = row * M_LEN + col;
            let (a, b) = (c[index], cref[index]);
            assert!(
                feq(a, b),
                "a != b, a[{}, {}]={}, b[{}, {}]={}",
                row,
                col,
                a,
                row,
                col,
                b
            );
        }
    }
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
