//! AArch64 intrinsics.
//!
//! The reference for NEON is [ARM's NEON Intrinsics Reference][arm_ref]. The
//! [ARM's NEON Intrinsics Online Database][arm_dat] is also useful.
//!
//! [arm_ref]: http://infocenter.arm.com/help/topic/com.arm.doc.ihi0073a/IHI0073A_arm_neon_intrinsics_ref.pdf
//! [arm_dat]: https://developer.arm.com/technologies/neon/intrinsics

mod v8;
pub use self::v8::*;

mod neon;
pub use self::neon::*;

mod crypto;
pub use self::crypto::*;

mod crc;
pub use self::crc::*;

pub use super::acle::*;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Generates the trap instruction `BRK 1`
#[cfg_attr(test, assert_instr(brk))]
#[inline]
pub unsafe fn brk() -> ! {
    crate::intrinsics::abort()
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::core_arch::simd::*;
    use std::mem::transmute;
    use stdarch_test::simd_test;

    macro_rules! verify_fn {
        ($f: ident, $a:expr, $b:expr, $e: expr, $t: ident) => {
            let r : $t = transmute($f(transmute($a), transmute($b)));
            assert_eq!(r, transmute($e));
        };
    }

    macro_rules! mku64x1 {
        ($n: expr) => {
            u64x1::new($n as u64)
        };
    }

    macro_rules! mku64x2 {
        ($a: expr, $b: expr) => {
            u64x2::new($a as u64, $b as u64)
        };
    }

    macro_rules! mks64x1 {
        ($n: expr) => {
            i64x1::new($n as i64)
        };
    }

    macro_rules! mks64x2 {
        ($a: expr, $b: expr) => {
            i64x2::new($a as i64, $b as i64)
        };
    }

    macro_rules! mkf64x1 {
        ($n: expr) => {
            $n as f64
        };
    }

    macro_rules! mkf64x2 {
        ($a: expr, $b: expr) => {
            f64x2::new($a as f64, $b as f64)
        };
    }

    const ZERO : i128  = 0x00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00i128;
    const ONE : i128   = 0x00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01i128;
    const N_ONE : u128 = 0xFF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FFu128;
    const EVEN_BYTES : u128 = 0xFF_00_FF_00_FF_00_FF_00_FF_00_FF_00_FF_00_FF_00u128;
    const ODD_BYTES : u128  = EVEN_BYTES >> 8;

    const FLT_N_ONE : f64 = -1.0;
    const FLT_ZERO : f64 = 0.0;
    const FLT_ONE : f64 = 1.0;

    const FLT_1_2: f64 = 1.2f64;
    const FLT_0_2: f64 = FLT_1_2 - FLT_ONE;
    const FLT_2_2: f64 = FLT_1_2 + FLT_ONE;
    const FLT_2 : f64 = 2.0f64;
    const FLT_N_1_2: f64 = -1.2f64;
    const FLT_3_4: f64 = 3.4f64;

    const FLT_MAX : f64 = std::f64::MAX;
    const FLT_MIN : f64 = std::f64::MIN;
    const FLT_INF : f64 = std::f64::INFINITY;
    const FLT_N_INF : f64 = std::f64::NEG_INFINITY;
    const FLT_NAN : f64 = std::f64::NAN;

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u64() {
        verify_fn!(vceq_u64, mku64x1!(ZERO), mku64x1!(ZERO), mku64x1!(N_ONE), u64);
        verify_fn!(vceq_u64, mku64x1!(N_ONE), mku64x1!(N_ONE), mku64x1!(N_ONE), u64);
        verify_fn!(vceq_u64, mku64x1!(N_ONE), mku64x1!(ZERO), mku64x1!(ZERO), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u64() {
        verify_fn!(vceqq_u64, mku64x2!(ZERO, ZERO), mku64x2!(ZERO, ZERO), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vceqq_u64, mku64x2!(N_ONE, ZERO), mku64x2!(N_ONE, ZERO), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vceqq_u64, mku64x2!(ZERO, N_ONE), mku64x2!(N_ONE, ZERO), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vceqq_u64, mku64x2!(N_ONE, N_ONE), mku64x2!(N_ONE, N_ONE), mku64x2!(N_ONE, N_ONE), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s64() {
        verify_fn!(vceq_s64, mks64x1!(ZERO), mks64x1!(ZERO), mks64x1!(N_ONE), i64);
        verify_fn!(vceq_s64, mks64x1!(N_ONE), mks64x1!(N_ONE), mks64x1!(N_ONE), i64);
        verify_fn!(vceq_s64, mks64x1!(N_ONE), mks64x1!(ZERO), mks64x1!(ZERO), i64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s64() {
        verify_fn!(vceqq_s64, mks64x2!(ZERO, ZERO), mks64x2!(ZERO, ZERO), mks64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vceqq_s64, mks64x2!(N_ONE, ZERO), mks64x2!(N_ONE, ZERO), mks64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vceqq_s64, mks64x2!(ZERO, N_ONE), mks64x2!(N_ONE, ZERO), mks64x2!(ZERO, ZERO), i128);
        verify_fn!(vceqq_s64, mks64x2!(N_ONE, N_ONE), mks64x2!(N_ONE, N_ONE), mks64x2!(N_ONE, N_ONE), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_p64() {
        verify_fn!(vceq_p64, mku64x1!(ZERO), mku64x1!(ZERO), mku64x1!(N_ONE), u64);
        verify_fn!(vceq_p64, mku64x1!(N_ONE), mku64x1!(N_ONE), mku64x1!(N_ONE), u64);
        verify_fn!(vceq_p64, mku64x1!(N_ONE), mku64x1!(ZERO), mku64x1!(ZERO), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_p64() {
        verify_fn!(vceqq_p64, mku64x2!(ZERO, ZERO), mku64x2!(ZERO, ZERO), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vceqq_p64, mku64x2!(N_ONE, ZERO), mku64x2!(N_ONE, ZERO), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vceqq_p64, mku64x2!(ZERO, N_ONE), mku64x2!(N_ONE, ZERO), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vceqq_p64, mku64x2!(N_ONE, N_ONE), mku64x2!(N_ONE, N_ONE), mku64x2!(N_ONE, N_ONE), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_f64() {
        verify_fn!(vceq_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_1_2), mku64x1!(N_ONE), u64);
        verify_fn!(vceq_f64, mkf64x1!(FLT_3_4), mkf64x1!(FLT_3_4), mku64x1!(N_ONE), u64);
        verify_fn!(vceq_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_3_4), mku64x1!(ZERO), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_f64() {
        verify_fn!(vceqq_f64, mkf64x2!(FLT_1_2, FLT_1_2), mkf64x2!(FLT_1_2, FLT_1_2), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vceqq_f64, mkf64x2!(FLT_3_4, FLT_1_2), mkf64x2!(FLT_3_4, FLT_1_2), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vceqq_f64, mkf64x2!(FLT_1_2, FLT_3_4), mkf64x2!(FLT_3_4, FLT_1_2), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vceqq_f64, mkf64x2!(FLT_1_2, FLT_3_4), mkf64x2!(FLT_3_4, FLT_1_2), mku64x2!(ZERO, ZERO), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s64() {
        verify_fn!(vcgt_s64, mks64x1!(ZERO), mks64x1!(ZERO), mks64x1!(ZERO), i64);
        verify_fn!(vcgt_s64, mks64x1!(ZERO), mks64x1!(N_ONE), mks64x1!(N_ONE), i64);
        verify_fn!(vcgt_s64, mks64x1!(N_ONE), mks64x1!(N_ONE), mks64x1!(ZERO), i64);
        verify_fn!(vcgt_s64, mks64x1!(ODD_BYTES), mks64x1!(EVEN_BYTES), mks64x1!(N_ONE), i64);
        verify_fn!(vcgt_s64, mks64x1!(EVEN_BYTES), mks64x1!(ODD_BYTES), mks64x1!(ZERO), i64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s64() {
        verify_fn!(vcgtq_s64, mks64x2!(ZERO, ZERO), mks64x2!(ZERO, ZERO), mks64x2!(ZERO, ZERO), i128);
        verify_fn!(vcgtq_s64, mks64x2!(ZERO, N_ONE), mks64x2!(N_ONE, ZERO), mks64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcgtq_s64, mks64x2!(N_ONE, ZERO), mks64x2!(ZERO, N_ONE), mks64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcgtq_s64, mks64x2!(ZERO, ZERO), mks64x2!(N_ONE, N_ONE), mks64x2!(N_ONE, N_ONE), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u64() {
        verify_fn!(vcgt_u64, mku64x1!(ZERO), mku64x1!(ZERO), mku64x1!(ZERO), u64);
        verify_fn!(vcgt_u64, mku64x1!(N_ONE), mku64x1!(ZERO), mku64x1!(N_ONE), u64);
        verify_fn!(vcgt_u64, mku64x1!(N_ONE), mku64x1!(N_ONE), mku64x1!(ZERO), u64);
        verify_fn!(vcgt_u64, mku64x1!(ODD_BYTES), mku64x1!(EVEN_BYTES), mku64x1!(ZERO), u64);
        verify_fn!(vcgt_u64, mku64x1!(EVEN_BYTES), mku64x1!(ODD_BYTES), mku64x1!(N_ONE), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u64() {
        verify_fn!(vcgtq_u64, mku64x2!(ZERO, ZERO), mku64x2!(ZERO, ZERO), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vcgtq_u64, mku64x2!(ZERO, N_ONE), mku64x2!(N_ONE, ZERO), mku64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcgtq_u64, mku64x2!(N_ONE, ZERO), mku64x2!(ZERO, N_ONE), mku64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcgtq_u64, mku64x2!(N_ONE, N_ONE), mku64x2!(ZERO, ZERO), mku64x2!(N_ONE, N_ONE), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_f64() {
        verify_fn!(vcgt_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_1_2), mku64x1!(ZERO), u64);
        verify_fn!(vcgt_f64, mkf64x1!(FLT_3_4), mkf64x1!(FLT_3_4), mku64x1!(ZERO), u64);
        verify_fn!(vcgt_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_3_4), mku64x1!(ZERO), u64);
        verify_fn!(vcgt_f64, mkf64x1!(FLT_3_4), mkf64x1!(FLT_1_2), mku64x1!(N_ONE), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_f64() {
        verify_fn!(vcgtq_f64, mkf64x2!(FLT_1_2, FLT_1_2), mkf64x2!(FLT_1_2, FLT_1_2), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vcgtq_f64, mkf64x2!(FLT_3_4, FLT_1_2), mkf64x2!(FLT_3_4, FLT_1_2), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vcgtq_f64, mkf64x2!(FLT_1_2, FLT_3_4), mkf64x2!(FLT_3_4, FLT_1_2), mku64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcgtq_f64, mkf64x2!(FLT_3_4, FLT_1_2), mkf64x2!(FLT_1_2, FLT_3_4), mku64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcgtq_f64, mkf64x2!(FLT_3_4, FLT_3_4), mkf64x2!(FLT_1_2, FLT_1_2), mku64x2!(N_ONE, N_ONE), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s64() {
        verify_fn!(vclt_s64, mks64x1!(ZERO), mks64x1!(ZERO), mks64x1!(ZERO), i64);
        verify_fn!(vclt_s64, mks64x1!(N_ONE), mks64x1!(ZERO), mks64x1!(N_ONE), i64);
        verify_fn!(vclt_s64, mks64x1!(N_ONE), mks64x1!(N_ONE), mks64x1!(ZERO), i64);
        verify_fn!(vclt_s64, mks64x1!(EVEN_BYTES), mks64x1!(ODD_BYTES), mks64x1!(N_ONE), i64);
        verify_fn!(vclt_s64, mks64x1!(ODD_BYTES), mks64x1!(EVEN_BYTES), mks64x1!(ZERO), i64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s64() {
        verify_fn!(vcltq_s64, mks64x2!(ZERO, ZERO), mks64x2!(ZERO, ZERO), mks64x2!(ZERO, ZERO), i128);
        verify_fn!(vcltq_s64, mks64x2!(N_ONE, ZERO), mks64x2!(ZERO, N_ONE), mks64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcltq_s64, mks64x2!(ZERO, N_ONE), mks64x2!(N_ONE, ZERO), mks64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcltq_s64, mks64x2!(N_ONE, N_ONE), mks64x2!(ZERO, ZERO), mks64x2!(N_ONE, N_ONE), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u64() {
        verify_fn!(vclt_u64, mku64x1!(ZERO), mku64x1!(ZERO), mku64x1!(ZERO), u64);
        verify_fn!(vclt_u64, mku64x1!(ZERO), mku64x1!(N_ONE), mku64x1!(N_ONE), u64);
        verify_fn!(vclt_u64, mku64x1!(N_ONE), mku64x1!(N_ONE), mku64x1!(ZERO), u64);
        verify_fn!(vclt_u64, mku64x1!(ODD_BYTES), mku64x1!(EVEN_BYTES), mku64x1!(N_ONE), u64);
        verify_fn!(vclt_u64, mku64x1!(EVEN_BYTES), mku64x1!(ODD_BYTES), mku64x1!(ZERO), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u64() {
        verify_fn!(vcltq_u64, mku64x2!(ZERO, ZERO), mku64x2!(ZERO, ZERO), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vcltq_u64, mku64x2!(ZERO, N_ONE), mku64x2!(N_ONE, ZERO), mku64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcltq_u64, mku64x2!(N_ONE, ZERO), mku64x2!(ZERO, N_ONE), mku64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcltq_u64, mku64x2!(ZERO, ZERO), mku64x2!(N_ONE, N_ONE), mku64x2!(N_ONE, N_ONE), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_f64() {
        verify_fn!(vclt_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_1_2), mku64x1!(ZERO), u64);
        verify_fn!(vclt_f64, mkf64x1!(FLT_3_4), mkf64x1!(FLT_3_4), mku64x1!(ZERO), u64);
        verify_fn!(vclt_f64, mkf64x1!(FLT_3_4), mkf64x1!(FLT_1_2), mku64x1!(ZERO), u64);
        verify_fn!(vclt_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_3_4), mku64x1!(N_ONE), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_f64() {
        verify_fn!(vcltq_f64, mkf64x2!(FLT_1_2, FLT_1_2), mkf64x2!(FLT_1_2, FLT_1_2), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vcltq_f64, mkf64x2!(FLT_1_2, FLT_3_4), mkf64x2!(FLT_1_2, FLT_3_4), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vcltq_f64, mkf64x2!(FLT_1_2, FLT_3_4), mkf64x2!(FLT_3_4, FLT_1_2), mku64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcltq_f64, mkf64x2!(FLT_3_4, FLT_1_2), mkf64x2!(FLT_1_2, FLT_3_4), mku64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcltq_f64, mkf64x2!(FLT_1_2, FLT_1_2), mkf64x2!(FLT_3_4, FLT_3_4), mku64x2!(N_ONE, N_ONE), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s64() {
        verify_fn!(vcle_s64, mks64x1!(ZERO), mks64x1!(ZERO), mks64x1!(N_ONE), i64);
        verify_fn!(vcle_s64, mks64x1!(ZERO), mks64x1!(N_ONE), mks64x1!(ZERO), i64);
        verify_fn!(vcle_s64, mks64x1!(N_ONE), mks64x1!(N_ONE), mks64x1!(N_ONE), i64);
        verify_fn!(vcle_s64, mks64x1!(ODD_BYTES), mks64x1!(EVEN_BYTES), mks64x1!(ZERO), i64);
        verify_fn!(vcle_s64, mks64x1!(EVEN_BYTES), mks64x1!(ODD_BYTES), mks64x1!(N_ONE), i64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s64() {
        verify_fn!(vcleq_s64, mks64x2!(ZERO, ZERO), mks64x2!(ZERO, ZERO), mks64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vcleq_s64, mks64x2!(ZERO, N_ONE), mks64x2!(N_ONE, ZERO), mks64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcleq_s64, mks64x2!(N_ONE, ZERO), mks64x2!(ZERO, N_ONE), mks64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcleq_s64, mks64x2!(ZERO, ZERO), mks64x2!(N_ONE, N_ONE), mks64x2!(ZERO, ZERO), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u64() {
        verify_fn!(vcle_u64, mku64x1!(ZERO), mku64x1!(ZERO), mku64x1!(N_ONE), u64);
        verify_fn!(vcle_u64, mku64x1!(N_ONE), mku64x1!(ZERO), mku64x1!(ZERO), u64);
        verify_fn!(vcle_u64, mku64x1!(N_ONE), mku64x1!(N_ONE), mku64x1!(N_ONE), u64);
        verify_fn!(vcle_u64, mku64x1!(ODD_BYTES), mku64x1!(EVEN_BYTES), mku64x1!(N_ONE), u64);
        verify_fn!(vcle_u64, mku64x1!(EVEN_BYTES), mku64x1!(ODD_BYTES), mku64x1!(ZERO), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u64() {
        verify_fn!(vcleq_u64, mku64x2!(ZERO, ZERO), mku64x2!(ZERO, ZERO), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vcleq_u64, mku64x2!(ZERO, N_ONE), mku64x2!(N_ONE, ZERO), mku64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcleq_u64, mku64x2!(N_ONE, ZERO), mku64x2!(ZERO, N_ONE), mku64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcleq_u64, mku64x2!(N_ONE, N_ONE), mku64x2!(ZERO, ZERO), mku64x2!(ZERO, ZERO), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_f64() {
        verify_fn!(vcle_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_1_2), mku64x1!(N_ONE), u64);
        verify_fn!(vcle_f64, mkf64x1!(FLT_3_4), mkf64x1!(FLT_3_4), mku64x1!(N_ONE), u64);
        verify_fn!(vcle_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_3_4), mku64x1!(N_ONE), u64);
        verify_fn!(vcle_f64, mkf64x1!(FLT_3_4), mkf64x1!(FLT_1_2), mku64x1!(ZERO), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_f64() {
        verify_fn!(vcleq_f64, mkf64x2!(FLT_1_2, FLT_1_2), mkf64x2!(FLT_1_2, FLT_1_2), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vcleq_f64, mkf64x2!(FLT_3_4, FLT_1_2), mkf64x2!(FLT_3_4, FLT_1_2), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vcleq_f64, mkf64x2!(FLT_1_2, FLT_3_4), mkf64x2!(FLT_3_4, FLT_1_2), mku64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcleq_f64, mkf64x2!(FLT_3_4, FLT_1_2), mkf64x2!(FLT_1_2, FLT_3_4), mku64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcleq_f64, mkf64x2!(FLT_3_4, FLT_3_4), mkf64x2!(FLT_1_2, FLT_1_2), mku64x2!(ZERO, ZERO), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s64() {
        verify_fn!(vcge_s64, mks64x1!(ZERO), mks64x1!(ZERO), mks64x1!(N_ONE), i64);
        verify_fn!(vcge_s64, mks64x1!(N_ONE), mks64x1!(ZERO), mks64x1!(ZERO), i64);
        verify_fn!(vcge_s64, mks64x1!(N_ONE), mks64x1!(N_ONE), mks64x1!(N_ONE), i64);
        verify_fn!(vcge_s64, mks64x1!(EVEN_BYTES), mks64x1!(ODD_BYTES), mks64x1!(ZERO), i64);
        verify_fn!(vcge_s64, mks64x1!(ODD_BYTES), mks64x1!(EVEN_BYTES), mks64x1!(N_ONE), i64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s64() {
        verify_fn!(vcgeq_s64, mks64x2!(ZERO, ZERO), mks64x2!(ZERO, ZERO), mks64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vcgeq_s64, mks64x2!(N_ONE, ZERO), mks64x2!(ZERO, N_ONE), mks64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcgeq_s64, mks64x2!(ZERO, N_ONE), mks64x2!(N_ONE, ZERO), mks64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcgeq_s64, mks64x2!(N_ONE, N_ONE), mks64x2!(ZERO, ZERO), mks64x2!(ZERO, ZERO), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u64() {
        verify_fn!(vcge_u64, mku64x1!(ZERO), mku64x1!(ZERO), mku64x1!(N_ONE), u64);
        verify_fn!(vcge_u64, mku64x1!(ZERO), mku64x1!(N_ONE), mku64x1!(ZERO), u64);
        verify_fn!(vcge_u64, mku64x1!(N_ONE), mku64x1!(N_ONE), mku64x1!(N_ONE), u64);
        verify_fn!(vcge_u64, mku64x1!(ODD_BYTES), mku64x1!(EVEN_BYTES), mku64x1!(ZERO), u64);
        verify_fn!(vcge_u64, mku64x1!(EVEN_BYTES), mku64x1!(ODD_BYTES), mku64x1!(N_ONE), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u64() {
        verify_fn!(vcgeq_u64, mku64x2!(ZERO, ZERO), mku64x2!(ZERO, ZERO), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vcgeq_u64, mku64x2!(ZERO, N_ONE), mku64x2!(N_ONE, ZERO), mku64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcgeq_u64, mku64x2!(N_ONE, ZERO), mku64x2!(ZERO, N_ONE), mku64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcgeq_u64, mku64x2!(ZERO, ZERO), mku64x2!(N_ONE, N_ONE), mku64x2!(ZERO, ZERO), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_f64() {
        verify_fn!(vcge_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_1_2), mku64x1!(N_ONE), u64);
        verify_fn!(vcge_f64, mkf64x1!(FLT_3_4), mkf64x1!(FLT_3_4), mku64x1!(N_ONE), u64);
        verify_fn!(vcge_f64, mkf64x1!(FLT_3_4), mkf64x1!(FLT_1_2), mku64x1!(N_ONE), u64);
        verify_fn!(vcge_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_3_4), mku64x1!(ZERO), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_f64() {
        verify_fn!(vcgeq_f64, mkf64x2!(FLT_1_2, FLT_1_2), mkf64x2!(FLT_1_2, FLT_1_2), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vcgeq_f64, mkf64x2!(FLT_1_2, FLT_3_4), mkf64x2!(FLT_1_2, FLT_3_4), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vcgeq_f64, mkf64x2!(FLT_1_2, FLT_3_4), mkf64x2!(FLT_3_4, FLT_1_2), mku64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcgeq_f64, mkf64x2!(FLT_3_4, FLT_1_2), mkf64x2!(FLT_1_2, FLT_3_4), mku64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcgeq_f64, mkf64x2!(FLT_1_2, FLT_1_2), mkf64x2!(FLT_3_4, FLT_3_4), mku64x2!(ZERO, ZERO), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_f64() {
        verify_fn!(vmul_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_ZERO), mkf64x1!(ZERO), u64);
        verify_fn!(vmul_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_ONE), mkf64x1!(FLT_1_2), u64);
        verify_fn!(vmul_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_N_ONE), mkf64x1!(FLT_N_1_2), u64);
        verify_fn!(vmul_f64, mkf64x1!(FLT_ONE), mkf64x1!(FLT_N_ONE), mkf64x1!(FLT_N_ONE), u64);
        verify_fn!(vmul_f64, mkf64x1!(FLT_INF), mkf64x1!(FLT_INF), mkf64x1!(FLT_INF), u64);
        verify_fn!(vmul_f64, mkf64x1!(FLT_INF), mkf64x1!(FLT_N_ONE), mkf64x1!(FLT_N_INF), u64);
        verify_fn!(vmul_f64, mkf64x1!(FLT_MIN), mkf64x1!(FLT_MIN), mkf64x1!(FLT_INF), u64);
        verify_fn!(vmul_f64, mkf64x1!(FLT_MAX), mkf64x1!(FLT_MAX), mkf64x1!(FLT_INF), u64);
        verify_fn!(vmul_f64, mkf64x1!(FLT_MAX), mkf64x1!(FLT_MIN), mkf64x1!(FLT_N_INF), u64);
        verify_fn!(vmul_f64, mkf64x1!(FLT_NAN), mkf64x1!(FLT_ZERO), mkf64x1!(FLT_NAN), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_f64() {
        verify_fn!(vmulq_f64, mkf64x2!(FLT_1_2, FLT_1_2), mkf64x2!(FLT_ZERO, FLT_ONE), mkf64x2!(ZERO, FLT_1_2), u128);
        verify_fn!(vmulq_f64, mkf64x2!(FLT_1_2, FLT_ONE), mkf64x2!(FLT_N_ONE, FLT_N_ONE), mkf64x2!(FLT_N_1_2, FLT_N_ONE), u128);
        verify_fn!(vmulq_f64, mkf64x2!(FLT_INF, FLT_INF), mkf64x2!(FLT_INF, FLT_N_ONE), mkf64x2!(FLT_INF, FLT_N_INF), u128);
        verify_fn!(vmulq_f64, mkf64x2!(FLT_MIN, FLT_MAX), mkf64x2!(FLT_MIN, FLT_MAX), mkf64x2!(FLT_INF, FLT_INF), u128);
        verify_fn!(vmulq_f64, mkf64x2!(FLT_MAX, FLT_NAN), mkf64x2!(FLT_MIN, FLT_ZERO), mkf64x2!(FLT_N_INF, FLT_NAN), u128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_f64() {
        verify_fn!(vsub_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_ZERO), mkf64x1!(FLT_1_2), u64);
        verify_fn!(vsub_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_ONE), mkf64x1!(FLT_0_2), f64);
        verify_fn!(vsub_f64, mkf64x1!(FLT_1_2), mkf64x1!(FLT_N_ONE), mkf64x1!(FLT_2_2), u64);
        verify_fn!(vsub_f64, mkf64x1!(FLT_ONE), mkf64x1!(FLT_N_ONE), mkf64x1!(FLT_2), u64);
        verify_fn!(vsub_f64, mkf64x1!(FLT_INF), mkf64x1!(FLT_INF), mkf64x1!(FLT_NAN), u64);
        verify_fn!(vsub_f64, mkf64x1!(FLT_INF), mkf64x1!(FLT_N_ONE), mkf64x1!(FLT_INF), u64);
        verify_fn!(vsub_f64, mkf64x1!(FLT_MIN), mkf64x1!(FLT_MIN), mkf64x1!(FLT_ZERO), u64);
        verify_fn!(vsub_f64, mkf64x1!(FLT_MAX), mkf64x1!(FLT_MAX), mkf64x1!(FLT_ZERO), u64);
        verify_fn!(vsub_f64, mkf64x1!(FLT_MAX), mkf64x1!(FLT_MIN), mkf64x1!(FLT_INF), u64);
        verify_fn!(vsub_f64, mkf64x1!(FLT_NAN), mkf64x1!(FLT_ZERO), mkf64x1!(FLT_NAN), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_f64() {
        verify_fn!(vsubq_f64, mkf64x2!(FLT_1_2, FLT_1_2), mkf64x2!(FLT_ZERO, FLT_ONE), mkf64x2!(FLT_1_2, FLT_0_2), f64x2);
        verify_fn!(vsubq_f64, mkf64x2!(FLT_1_2, FLT_ONE), mkf64x2!(FLT_N_ONE, FLT_N_ONE), mkf64x2!(FLT_2_2, FLT_2), u128);
        verify_fn!(vsubq_f64, mkf64x2!(FLT_INF, FLT_INF), mkf64x2!(FLT_INF, FLT_N_ONE), mkf64x2!(FLT_NAN, FLT_INF), u128);
        verify_fn!(vsubq_f64, mkf64x2!(FLT_MIN, FLT_MAX), mkf64x2!(FLT_MIN, FLT_MAX), mkf64x2!(FLT_ZERO, FLT_ZERO), u128);
        verify_fn!(vsubq_f64, mkf64x2!(FLT_MAX, FLT_NAN), mkf64x2!(FLT_MIN, FLT_ZERO), mkf64x2!(FLT_INF, FLT_NAN), u128);
    }
}
