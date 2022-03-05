#![allow(dead_code)]

use super::intrinsics::*;

#[cfg(target_feature = "fma")]
#[inline(always)]
pub unsafe fn fmadd_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    _mm_fmadd_ps(a, b, c)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub unsafe fn fmadd_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    _mm_add_ps(_mm_mul_ps(a, b), c)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub unsafe fn fmsub_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    _mm_fmsub_ps(a, b, c)
}

#[cfg(not(target_feature = "fma"))]
pub unsafe fn fmsub_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    _mm_sub_ps(_mm_mul_ps(a, b), c)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub unsafe fn fmadd_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    _mm_fmadd_pd(a, b, c)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub unsafe fn fmadd_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    _mm_add_pd(_mm_mul_pd(a, b), c)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub unsafe fn fmsub_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    _mm_fmsub_pd(a, b, c)
}

#[cfg(not(target_feature = "fma"))]
pub unsafe fn fmsub_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    _mm_sub_pd(_mm_mul_pd(a, b), c)
}
