use crate::aligned_alloc;
use crate::matrix::{Number, Matrix, MatrixMut, MutMatrix, ConstMatrix, ConstTransposedMatrix, MutTransposedMatrix};
use crate::kernel::params::single::*;
use crate::kernel;
use crate::kernel::GemmKernel;
use crate::dim::Dim;
use crate::executor::Executor;

pub unsafe fn gemm<E, F, K, MR, NR>(
    e: &E,
    transa: bool,
    transb: bool,
    transc: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: F,
    a: *const F,
    lda: usize,
    b: *const F,
    ldb: usize,
    beta: F,
    c: *mut F,
    ldc: usize,
) 
where E: Executor,
      F: Number,
      MR: Dim, NR: Dim,
      K: GemmKernel<F, MR, NR>,
{
    match (transa, transb, transc) {
        (false, false, false) => gemm_template::<E, F, K, MR, NR, _, _, _>(
            e, m, n, k, alpha, 
            ConstMatrix::new(a, lda),
            ConstMatrix::new(b, ldb),
            beta,
            MutMatrix::new(c, ldc)), 

        (false, false, true) => gemm_template::<E, F, K, MR, NR, _, _, _>(
            e, m, n, k, alpha, 
            ConstMatrix::new(a, lda),
            ConstMatrix::new(b, ldb),
            beta,
            MutTransposedMatrix::new(c, ldc)), 
        
        (false, true, false) => gemm_template::<E, F, K, MR, NR, _, _, _>(
            e, m, n, k, alpha, 
            ConstMatrix::new(a, lda),
            ConstTransposedMatrix::new(b, ldb),
            beta,
            MutMatrix::new(c, ldc)), 

        (false, true, true) => gemm_template::<E, F, K, MR, NR, _, _, _>(
            e, m, n, k, alpha, 
            ConstMatrix::new(a, lda),
            ConstTransposedMatrix::new(b, ldb),
            beta,
            MutTransposedMatrix::new(c, ldc)), 

        (true, false, false) => gemm_template::<E, F, K, MR, NR, _, _, _>(
            e, m, n, k, alpha, 
            ConstTransposedMatrix::new(a, lda),
            ConstMatrix::new(b, ldb),
            beta,
            MutMatrix::new(c, ldc)), 
        
        (true, false, true) => gemm_template::<E, F, K, MR, NR, _, _, _>(
            e, m, n, k, alpha, 
            ConstTransposedMatrix::new(a, lda),
            ConstMatrix::new(b, ldb),
            beta,
            MutTransposedMatrix::new(c, ldc)), 

                    
        (true, true, false) => gemm_template::<E, F, K, MR, NR, _, _, _>(
            e, m, n, k, alpha, 
            ConstTransposedMatrix::new(a, lda),
            ConstTransposedMatrix::new(b, ldb),
            beta,
            MutMatrix::new(c, ldc)), 

                    
        (true, true, true) => gemm_template::<E, F, K, MR, NR, _, _, _>(
            e, m, n, k, alpha, 
            ConstTransposedMatrix::new(a, lda),
            ConstTransposedMatrix::new(b, ldb),
            beta,
            MutTransposedMatrix::new(c, ldc)), 
    }
}

unsafe fn gemm_template<E, F, K, MR, NR, A, B, C>(
    e: &E,
    m: usize,
    n: usize,
    k: usize,
    alpha: F,
    a: A,
    b: B,
    beta: F,
    c: C
) 
where E: Executor,
      F: Number,
      MR: Dim, NR: Dim,
      K: GemmKernel<F, MR, NR>,
      A: Matrix<F>,
      B: Matrix<F>,
      C: MatrixMut<F>,
{
    let packed_a = aligned_alloc::Alloc::new(MC * KC * std::mem::size_of::<F>());
    let packed_b = aligned_alloc::Alloc::new(KC * NC * std::mem::size_of::<F>());

    for j in (0..n).step_by(NC) {
        let j_b = std::cmp::min(n - j, NC);
        for p in (0..k).step_by(KC) {
            let p_b = std::cmp::min(k - p, KC);
            for i in (0..m).step_by(MC) {
                let i_b = std::cmp::min(m - i, MC);

                let pa = MutMatrix::new(packed_a.ptr::<F>(), p_b);
                let pb = MutMatrix::new(packed_b.ptr::<F>(), p_b);

                inner_kernel::<E, F, K, MR, NR, _, _, _>(
                    e, 
                    i_b,
                    j_b,
                    p_b,
                    alpha,
                    a.sub(p, i),
                    b.sub(j, p),
                    beta,
                    c.sub(j, i),
                    pa,
                    pb,
                    i == 0
                );
            }
        }
    }
}

//      | MR |   
// +----------------------+
// |TL  :    :    :    |TR|
// |    :    :    :    |  |
// + - - - - - - - - - ---+----
// |    :    :    :    |  |  NR
// |    :    :    :    |  |
// + - - - - - - - - - ---+----
// |    :    :    :    |  |
// |    :    :    :    |  |
// +----------------------+
// |BL  |    |    |    |BR|
// +----------------------+
unsafe fn inner_kernel<E, F, K, MR, NR, A, B, C>(
    e: &E,
    m: usize,
    n: usize,
    k: usize,
    alpha: F,
    a: A,
    b: B,
    beta: F,
    c: C,
    pa: MutMatrix<F>,
    pb: MutMatrix<F>,
    first_time: bool,
)
    where E: Executor,
          F: Number,
          MR: Dim, 
          NR: Dim,
          K: kernel::GemmKernel<F, MR, NR>,
          A: Matrix<F>,
          B: Matrix<F>,
          C: MatrixMut<F>,

{
    let n_left = n % NR;
    let n_main = n - n_left;

    let m_left = m % MR;
    let m_main = m - m_left;

    if first_time {
        e.execute(0, n_main, NR, move |j| 
            K::pack_row_b(b, pb, j));
    }

    e.execute(0, m_main, MR, move |i| 
        K::pack_row_a(a, pa, i));

    e.synchronize();
    
    e.execute(0, n_main, NR, move |j| {
        // Section TL
        for i in (0..m_main).step_by(MR) {
            K::main_tl(alpha,
                pa.sub_row(i),
                pb.sub_row(j),
                beta,
                c.sub(j, i));
        }

        // Section TR
        for i in m_main..m {
            K::sup_tr(
                alpha,
                a.sub_col(i),
                pb.sub_row(j),
                beta,
                c.sub(j, i));
        }
    });

    e.execute(n_main, n, 1, move |j| {
        // Section BL
        for i in (0..m_main).step_by(MR) {
            K::sup_bl(
                alpha,
                pa.sub_row(i),
                b.sub_row(j),
                beta,
                c.sub(j, i)
            );
        }

        // Section BR
        for i in m_main..m {
            K::sup_br(
                k,
                alpha,
                a.sub_col(i),
                b.sub_row(j),
                beta,
                c.sub(j, i))
        }
    });

    e.synchronize();
}