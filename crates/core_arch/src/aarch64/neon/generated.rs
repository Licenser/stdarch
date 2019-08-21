use super::*;
#[cfg(test)]
use stdarch_test::assert_instr;

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceq_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceq_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceq_p64(a: poly64x1_t, b: poly64x1_t) -> uint64x1_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqq_p64(a: poly64x2_t, b: poly64x2_t) -> uint64x2_t {
    simd_eq(a, b)
}

/// Floating-point compare equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
pub unsafe fn vceq_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_eq(a, b)
}

/// Floating-point compare equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
pub unsafe fn vceqq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_eq(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcgt_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcgtq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhi))]
pub unsafe fn vcgt_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhi))]
pub unsafe fn vcgtq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_gt(a, b)
}

/// Floating-point compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub unsafe fn vcgt_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_gt(a, b)
}

/// Floating-point compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub unsafe fn vcgtq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_gt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vclt_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcltq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhi))]
pub unsafe fn vclt_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhi))]
pub unsafe fn vcltq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_lt(a, b)
}

/// Floating-point compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub unsafe fn vclt_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_lt(a, b)
}

/// Floating-point compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub unsafe fn vcltq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_lt(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcle_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcleq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhs))]
pub unsafe fn vcle_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhs))]
pub unsafe fn vcleq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_le(a, b)
}

/// Floating-point compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
pub unsafe fn vcle_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_le(a, b)
}

/// Floating-point compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
pub unsafe fn vcleq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_le(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcge_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcgeq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhs))]
pub unsafe fn vcge_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhs))]
pub unsafe fn vcgeq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_ge(a, b)
}

/// Floating-point compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
pub unsafe fn vcge_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_ge(a, b)
}

/// Floating-point compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
pub unsafe fn vcgeq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_ge(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mul))]
pub unsafe fn vmul_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mul))]
pub unsafe fn vmulq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_mul(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sub))]
pub unsafe fn vsub_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sub))]
pub unsafe fn vsubq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_sub(a, b)
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

    const FLT_A : f64 = 1.2f64;
    const FLT_B : f64 = 3.4f64;

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
        verify_fn!(vceq_f64, mkf64x1!(FLT_A), mkf64x1!(FLT_A), mku64x1!(N_ONE), u64);
        verify_fn!(vceq_f64, mkf64x1!(FLT_B), mkf64x1!(FLT_B), mku64x1!(N_ONE), u64);
        verify_fn!(vceq_f64, mkf64x1!(FLT_A), mkf64x1!(FLT_B), mku64x1!(ZERO), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_f64() {
        verify_fn!(vceqq_f64, mkf64x2!(FLT_A, FLT_A), mkf64x2!(FLT_A, FLT_A), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vceqq_f64, mkf64x2!(FLT_B, FLT_A), mkf64x2!(FLT_B, FLT_A), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vceqq_f64, mkf64x2!(FLT_A, FLT_B), mkf64x2!(FLT_B, FLT_A), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vceqq_f64, mkf64x2!(FLT_A, FLT_B), mkf64x2!(FLT_B, FLT_A), mku64x2!(ZERO, ZERO), i128);
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
        verify_fn!(vcgt_f64, mkf64x1!(FLT_A), mkf64x1!(FLT_A), mku64x1!(ZERO), u64);
        verify_fn!(vcgt_f64, mkf64x1!(FLT_B), mkf64x1!(FLT_B), mku64x1!(ZERO), u64);
        verify_fn!(vcgt_f64, mkf64x1!(FLT_A), mkf64x1!(FLT_B), mku64x1!(ZERO), u64);
        verify_fn!(vcgt_f64, mkf64x1!(FLT_B), mkf64x1!(FLT_A), mku64x1!(N_ONE), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_f64() {
        verify_fn!(vcgtq_f64, mkf64x2!(FLT_A, FLT_A), mkf64x2!(FLT_A, FLT_A), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vcgtq_f64, mkf64x2!(FLT_B, FLT_A), mkf64x2!(FLT_B, FLT_A), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vcgtq_f64, mkf64x2!(FLT_A, FLT_B), mkf64x2!(FLT_B, FLT_A), mku64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcgtq_f64, mkf64x2!(FLT_B, FLT_A), mkf64x2!(FLT_A, FLT_B), mku64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcgtq_f64, mkf64x2!(FLT_B, FLT_B), mkf64x2!(FLT_A, FLT_A), mku64x2!(N_ONE, N_ONE), i128);
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
        verify_fn!(vclt_f64, mkf64x1!(FLT_A), mkf64x1!(FLT_A), mku64x1!(ZERO), u64);
        verify_fn!(vclt_f64, mkf64x1!(FLT_B), mkf64x1!(FLT_B), mku64x1!(ZERO), u64);
        verify_fn!(vclt_f64, mkf64x1!(FLT_B), mkf64x1!(FLT_A), mku64x1!(ZERO), u64);
        verify_fn!(vclt_f64, mkf64x1!(FLT_A), mkf64x1!(FLT_B), mku64x1!(N_ONE), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_f64() {
        verify_fn!(vcltq_f64, mkf64x2!(FLT_A, FLT_A), mkf64x2!(FLT_A, FLT_A), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vcltq_f64, mkf64x2!(FLT_A, FLT_B), mkf64x2!(FLT_A, FLT_B), mku64x2!(ZERO, ZERO), i128);
        verify_fn!(vcltq_f64, mkf64x2!(FLT_A, FLT_B), mkf64x2!(FLT_B, FLT_A), mku64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcltq_f64, mkf64x2!(FLT_B, FLT_A), mkf64x2!(FLT_A, FLT_B), mku64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcltq_f64, mkf64x2!(FLT_A, FLT_A), mkf64x2!(FLT_B, FLT_B), mku64x2!(N_ONE, N_ONE), i128);
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
        verify_fn!(vcle_f64, mkf64x1!(FLT_A), mkf64x1!(FLT_A), mku64x1!(N_ONE), u64);
        verify_fn!(vcle_f64, mkf64x1!(FLT_B), mkf64x1!(FLT_B), mku64x1!(N_ONE), u64);
        verify_fn!(vcle_f64, mkf64x1!(FLT_A), mkf64x1!(FLT_B), mku64x1!(N_ONE), u64);
        verify_fn!(vcle_f64, mkf64x1!(FLT_B), mkf64x1!(FLT_A), mku64x1!(ZERO), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_f64() {
        verify_fn!(vcleq_f64, mkf64x2!(FLT_A, FLT_A), mkf64x2!(FLT_A, FLT_A), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vcleq_f64, mkf64x2!(FLT_B, FLT_A), mkf64x2!(FLT_B, FLT_A), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vcleq_f64, mkf64x2!(FLT_A, FLT_B), mkf64x2!(FLT_B, FLT_A), mku64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcleq_f64, mkf64x2!(FLT_B, FLT_A), mkf64x2!(FLT_A, FLT_B), mku64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcleq_f64, mkf64x2!(FLT_B, FLT_B), mkf64x2!(FLT_A, FLT_A), mku64x2!(ZERO, ZERO), i128);
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
        verify_fn!(vcge_f64, mkf64x1!(FLT_A), mkf64x1!(FLT_A), mku64x1!(N_ONE), u64);
        verify_fn!(vcge_f64, mkf64x1!(FLT_B), mkf64x1!(FLT_B), mku64x1!(N_ONE), u64);
        verify_fn!(vcge_f64, mkf64x1!(FLT_B), mkf64x1!(FLT_A), mku64x1!(N_ONE), u64);
        verify_fn!(vcge_f64, mkf64x1!(FLT_A), mkf64x1!(FLT_B), mku64x1!(ZERO), u64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_f64() {
        verify_fn!(vcgeq_f64, mkf64x2!(FLT_A, FLT_A), mkf64x2!(FLT_A, FLT_A), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vcgeq_f64, mkf64x2!(FLT_A, FLT_B), mkf64x2!(FLT_A, FLT_B), mku64x2!(N_ONE, N_ONE), i128);
        verify_fn!(vcgeq_f64, mkf64x2!(FLT_A, FLT_B), mkf64x2!(FLT_B, FLT_A), mku64x2!(ZERO, N_ONE), i128);
        verify_fn!(vcgeq_f64, mkf64x2!(FLT_B, FLT_A), mkf64x2!(FLT_A, FLT_B), mku64x2!(N_ONE, ZERO), i128);
        verify_fn!(vcgeq_f64, mkf64x2!(FLT_A, FLT_A), mkf64x2!(FLT_B, FLT_B), mku64x2!(ZERO, ZERO), i128);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_f64() {
        let a: f64 = 1.0;
        let b: f64 = 2.0;
        let e: f64 = 2.0;
        let r: f64 = transmute(vmul_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_f64() {
        let a: f64x2 = f64x2::new(1.0, 2.0);
        let b: f64x2 = f64x2::new(2.0, 3.0);
        let e: f64x2 = f64x2::new(2.0, 6.0);
        let r: f64x2 = transmute(vmulq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_f64() {
        let a: f64 = 1.0;
        let b: f64 = 1.0;
        let e: f64 = 0.0;
        let r: f64 = transmute(vsub_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_f64() {
        let a: f64x2 = f64x2::new(1.0, 4.0);
        let b: f64x2 = f64x2::new(1.0, 2.0);
        let e: f64x2 = f64x2::new(0.0, 2.0);
        let r: f64x2 = transmute(vsubq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
}
