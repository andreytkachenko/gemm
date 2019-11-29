#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
mod avx;

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub use avx::{l1d::*, l1s::*};

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub(crate) use avx::l3s::{
    sgemm_pa_16x as sgemm_pa_n, sgemm_sup_16x1 as sgemm_sup0, sgemm_ukr_16x4 as sgemm_ukr, sgemm_sup0_t,
};

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub(crate) use avx::l3d::{
    dgemm_pa_8x as dgemm_pa, dgemm_sup_8x1 as dgemm_sup0, dgemm_ukr_8x4 as dgemm_ukr,
};

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub(crate) use generic::l3s::{sgemm_pb_x4 as sgemm_pb_n, sgemm_pb_t, sgemm_pa_t, sgemm_sup_1x4 as sgemm_sup1, sgemm_sup1_t};

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub(crate) use generic::l3d::{dgemm_pb_x4 as dgemm_pb, dgemm_sup_1x4 as dgemm_sup1};


mod generic;


pub mod params {
    pub mod single {
        pub const MC: usize = 256;
        pub const KC: usize = 128;
        pub const NC: usize = 1024;
        pub const MR: usize = 16;
        pub const NR: usize = 4;
    }
    pub mod double {
        pub const MC: usize = 256;
        pub const KC: usize = 512;
        pub const NC: usize = 4096;
        pub const MR: usize = 8;
        pub const NR: usize = 4;
    }
}