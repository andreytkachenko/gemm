use crate::aligned_alloc;
use crate::matrix::{Matrix, MatrixMut, MutMatrix};
use crate::kernel::params::single::*;
use crate::kernel;


pub(crate) unsafe fn sgemm<A, B, C>(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: A,
    b: B,
    beta: f32,
    c: C
) 
where A: Matrix,
      B: Matrix,
      C: MatrixMut,
{
    let packed_a = aligned_alloc::Alloc::new(MC * KC * std::mem::size_of::<f32>());
    let packed_b = aligned_alloc::Alloc::new(KC * NC * std::mem::size_of::<f32>());

    for j in (0..n).step_by(NC) {
        let j_b = std::cmp::min(n - j, NC);
        for p in (0..k).step_by(KC) {
            let p_b = std::cmp::min(k - p, KC);
            for i in (0..m).step_by(MC) {
                let i_b = std::cmp::min(m - i, MC);

                let pa = MutMatrix::new(packed_a.ptr_f32(), p_b);
                let pb = MutMatrix::new(packed_b.ptr_f32(), p_b);

                inner_kernel(
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


// +----------------------+
// |A   :    :    :    |B |
// |    :    :    :    |  |
// + - - - - - - - - - ---+
// |    :    :    :    |  |
// |    :    :    :    |  |
// + - - - - - - - - - ---+
// |    :    :    :    |  |
// |    :    :    :    |  |
// +----------------------+
// |C   |    |    |    |D |
// +----------------------+


#[inline]
unsafe fn sgemm_pa<A: Matrix>(a: A, pa: MutMatrix, i: usize) {
    if  a.is_transposed() {
        kernel::sgemm_pa_t(pa.stride, a.col(i), a.stride(), pa.row_mut(i));
    } else {
        kernel::sgemm_pa_n(pa.stride, a.col(i), a.stride(), pa.row_mut(i));
    }
}

#[inline]
unsafe fn sgemm_pb<B: Matrix>(b: B, pb: MutMatrix, j: usize) {
    if  b.is_transposed() {
        kernel::sgemm_pb_t(pb.stride, b.row(j), b.stride(), pb.row_mut(j));
    } else {
        kernel::sgemm_pb_n(pb.stride, b.row(j), b.stride(), pb.row_mut(j));
    }
}

#[inline]
unsafe fn sgemm_ukr<C: MatrixMut>(
    i: usize, j: usize,
    alpha: f32,
    pa: MutMatrix,
    pb: MutMatrix,
    beta: f32,
    c: C) 
{
    if  c.is_transposed() {
        unimplemented!()
    } else {
        kernel::sgemm_ukr(
            pa.stride,
            alpha,
            pa.row(i),
            pb.row(j),
            beta,
            c.index_mut(j, i),
            c.stride())
    }
}

#[inline]
unsafe fn sgemm_sup1<A: Matrix, C: MatrixMut>(
    i: usize, j: usize,
    alpha: f32,
    a: A,
    pb: MutMatrix,
    beta: f32,
    c: C) 
{
    if  c.is_transposed() {
        unimplemented!()
    } else {
        if a.is_transposed() {
            kernel::sgemm_sup1_t(
                pb.stride,
                alpha,
                a.col(i),
                a.stride(),
                pb.row(j),
                beta,
                c.index_mut(j, i),
                c.stride())
        } else {
            kernel::sgemm_sup1(
                pb.stride,
                alpha,
                a.col(i),
                a.stride(),
                pb.row(j),
                beta,
                c.index_mut(j, i),
                c.stride())
        }
    }
}

#[inline]
unsafe fn sgemm_sup0<B: Matrix, C: MatrixMut>(
    i: usize, j: usize,
    alpha: f32,
    pa: MutMatrix,
    b: B,
    beta: f32,
    c: C) 
{
    if  c.is_transposed() {
        unimplemented!()
    } else {
        if b.is_transposed() {
            kernel::sgemm_sup0_t(
                pa.stride,
                alpha,
                pa.row(i),
                b.row(j),
                b.stride(),
                beta,
                c.index_mut(j, i))
        } else {
            kernel::sgemm_sup0(
                pa.stride,
                alpha,
                pa.row(i),
                b.row(j),
                beta,
                c.index_mut(j, i))
        }
    }
}

unsafe fn inner_kernel<A, B, C>(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: A,
    b: B,
    beta: f32,
    c: C,
    pa: MutMatrix,
    pb: MutMatrix,
    first_time: bool,
)
    where A: Matrix,
          B: Matrix,
          C: MatrixMut,

{
    let n_left = n % NR;
    let n_main = n - n_left;

    let m_left = m % MR;
    let m_main = m - m_left;

    if first_time {
        for j in (0..n_main).step_by(NR) {
            sgemm_pb(b, pb, j);
        }
    }

    for i in (0..m_main).step_by(MR) {
        sgemm_pa(a, pa, i);
    };
    
    
    for j in (0..n_main).step_by(NR) {
        
        // Section A
        for i in (0..m_main).step_by(MR) {
            sgemm_ukr(
                i,
                j,
                alpha,
                pa,
                pb,
                beta,
                c,
            );
        }

        // Section B
        for i in m_main..m {
            sgemm_sup1(
                i,
                j,
                alpha,
                a,
                pb,
                beta,
                c,
            );
        }
    }   

    for j in n_main..n {
        
        // Section C
        for i in (0..m_main).step_by(MR) {
            sgemm_sup0(
                i,
                j,
                alpha,
                pa,
                b,
                beta,
                c,
            );
        }

        // Section D
        for i in m_main..m {
            let mut elem = 0.0;

            for p in 0..k {
                elem += a.get(p, i) * b.get(j, p);
            }

            elem *= alpha;

            if beta != 0.0 {
                elem += beta * c.get(j, i);
            }

            c.set(j, i, elem);
        }
    }
}