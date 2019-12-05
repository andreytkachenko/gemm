mod matrix;
mod gemm;
mod sgemm;
mod aligned_alloc;
mod kernel;
mod dim;
mod executor;


extern crate blas;
extern crate openblas;

use crate::sgemm::sgemm;

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
            1.0,
            cref1.as_mut_slice(),
            m as i32,
        );

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
            &executor::RayonExecutor,
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
}

