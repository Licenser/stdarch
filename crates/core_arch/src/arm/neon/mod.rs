//! ARMv7 NEON intrinsics

#[rustfmt::skip]
mod generated;
#[rustfmt::skip]
pub use self::generated::*;

use crate::{core_arch::simd_llvm::*, hint::unreachable_unchecked, mem::transmute, ptr};
#[cfg(test)]
use stdarch_test::assert_instr;

types! {
    /// ARM-specific 64-bit wide vector of eight packed `i8`.
    pub struct int8x8_t(i8, i8, i8, i8, i8, i8, i8, i8);
    /// ARM-specific 64-bit wide vector of eight packed `u8`.
    pub struct uint8x8_t(u8, u8, u8, u8, u8, u8, u8, u8);
    /// ARM-specific 64-bit wide polynomial vector of eight packed `u8`.
    pub struct poly8x8_t(u8, u8, u8, u8, u8, u8, u8, u8);
    /// ARM-specific 64-bit wide vector of four packed `i16`.
    pub struct int16x4_t(i16, i16, i16, i16);
    /// ARM-specific 64-bit wide vector of four packed `u16`.
    pub struct uint16x4_t(u16, u16, u16, u16);
    // FIXME: ARM-specific 64-bit wide vector of four packed `f16`.
    // pub struct float16x4_t(f16, f16, f16, f16);
    /// ARM-specific 64-bit wide vector of four packed `u16`.
    pub struct poly16x4_t(u16, u16, u16, u16);
    /// ARM-specific 64-bit wide vector of two packed `i32`.
    pub struct int32x2_t(i32, i32);
    /// ARM-specific 64-bit wide vector of two packed `u32`.
    pub struct uint32x2_t(u32, u32);
    /// ARM-specific 64-bit wide vector of two packed `f32`.
    pub struct float32x2_t(f32, f32);
    /// ARM-specific 64-bit wide vector of one packed `i64`.
    pub struct int64x1_t(i64);
    /// ARM-specific 64-bit wide vector of one packed `u64`.
    pub struct uint64x1_t(u64);

    /// ARM-specific 128-bit wide vector of sixteen packed `i8`.
    pub struct int8x16_t(
        i8, i8 ,i8, i8, i8, i8 ,i8, i8,
        i8, i8 ,i8, i8, i8, i8 ,i8, i8,
    );
    /// ARM-specific 128-bit wide vector of sixteen packed `u8`.
    pub struct uint8x16_t(
        u8, u8 ,u8, u8, u8, u8 ,u8, u8,
        u8, u8 ,u8, u8, u8, u8 ,u8, u8,
    );
    /// ARM-specific 128-bit wide vector of sixteen packed `u8`.
    pub struct poly8x16_t(
        u8, u8, u8, u8, u8, u8, u8, u8,
        u8, u8, u8, u8, u8, u8, u8, u8
    );
    /// ARM-specific 128-bit wide vector of eight packed `i16`.
    pub struct int16x8_t(i16, i16, i16, i16, i16, i16, i16, i16);
    /// ARM-specific 128-bit wide vector of eight packed `u16`.
    pub struct uint16x8_t(u16, u16, u16, u16, u16, u16, u16, u16);
    // FIXME: ARM-specific 128-bit wide vector of eight packed `f16`.
    // pub struct float16x8_t(f16, f16, f16, f16, f16, f16, f16);
    /// ARM-specific 128-bit wide vector of eight packed `u16`.
    pub struct poly16x8_t(u16, u16, u16, u16, u16, u16, u16, u16);
    /// ARM-specific 128-bit wide vector of four packed `i32`.
    pub struct int32x4_t(i32, i32, i32, i32);
    /// ARM-specific 128-bit wide vector of four packed `u32`.
    pub struct uint32x4_t(u32, u32, u32, u32);
    /// ARM-specific 128-bit wide vector of four packed `f32`.
    pub struct float32x4_t(f32, f32, f32, f32);
    /// ARM-specific 128-bit wide vector of two packed `i64`.
    pub struct int64x2_t(i64, i64);
    /// ARM-specific 128-bit wide vector of two packed `u64`.
    pub struct uint64x2_t(u64, u64);
}

/// ARM-specific type containing two `int8x8_t` vectors.
#[derive(Copy, Clone)]
pub struct int8x8x2_t(pub int8x8_t, pub int8x8_t);
/// ARM-specific type containing three `int8x8_t` vectors.
#[derive(Copy, Clone)]
pub struct int8x8x3_t(pub int8x8_t, pub int8x8_t, pub int8x8_t);
/// ARM-specific type containing four `int8x8_t` vectors.
#[derive(Copy, Clone)]
pub struct int8x8x4_t(pub int8x8_t, pub int8x8_t, pub int8x8_t, pub int8x8_t);

/// ARM-specific type containing two `uint8x8_t` vectors.
#[derive(Copy, Clone)]
pub struct uint8x8x2_t(pub uint8x8_t, pub uint8x8_t);
/// ARM-specific type containing three `uint8x8_t` vectors.
#[derive(Copy, Clone)]
pub struct uint8x8x3_t(pub uint8x8_t, pub uint8x8_t, pub uint8x8_t);
/// ARM-specific type containing four `uint8x8_t` vectors.
#[derive(Copy, Clone)]
pub struct uint8x8x4_t(pub uint8x8_t, pub uint8x8_t, pub uint8x8_t, pub uint8x8_t);

/// ARM-specific type containing two `poly8x8_t` vectors.
#[derive(Copy, Clone)]
pub struct poly8x8x2_t(pub poly8x8_t, pub poly8x8_t);
/// ARM-specific type containing three `poly8x8_t` vectors.
#[derive(Copy, Clone)]
pub struct poly8x8x3_t(pub poly8x8_t, pub poly8x8_t, pub poly8x8_t);
/// ARM-specific type containing four `poly8x8_t` vectors.
#[derive(Copy, Clone)]
pub struct poly8x8x4_t(pub poly8x8_t, pub poly8x8_t, pub poly8x8_t, pub poly8x8_t);

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrsqrte.v2f32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frsqrte.v2f32")]
    fn frsqrte_v2f32(a: float32x2_t) -> float32x2_t;

    //uint32x2_t vqmovn_u64 (uint64x2_t a)
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqmovnu.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqxtn.v2i32")]
    fn vqmovn_u64_(a: uint64x2_t) -> uint32x2_t;

    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmins.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sminp.v8i8")]
    fn vpmins_v8i8(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmins.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sminp.v4i16")]
    fn vpmins_v4i16(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmins.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sminp.v2i32")]
    fn vpmins_v2i32(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpminu.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uminp.v8i8")]
    fn vpminu_v8i8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpminu.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uminp.v4i16")]
    fn vpminu_v4i16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpminu.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uminp.v2i32")]
    fn vpminu_v2i32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmins.v2f32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fminp.v2f32")]
    fn vpminf_v2f32(a: float32x2_t, b: float32x2_t) -> float32x2_t;

    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxs.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smaxp.v8i8")]
    fn vpmaxs_v8i8(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxs.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smaxp.v4i16")]
    fn vpmaxs_v4i16(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxs.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smaxp.v2i32")]
    fn vpmaxs_v2i32(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxu.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umaxp.v8i8")]
    fn vpmaxu_v8i8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxu.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umaxp.v4i16")]
    fn vpmaxu_v4i16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxu.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umaxp.v2i32")]
    fn vpmaxu_v2i32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxs.v2f32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmaxp.v2f32")]
    fn vpmaxf_v2f32(a: float32x2_t, b: float32x2_t) -> float32x2_t;
}

#[cfg(target_arch = "arm")]
#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.arm.neon.vtbl1"]
    fn vtbl1(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vtbl2"]
    fn vtbl2(a: int8x8_t, b: int8x8_t, b: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vtbl3"]
    fn vtbl3(a: int8x8_t, b: int8x8_t, b: int8x8_t, c: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vtbl4"]
    fn vtbl4(a: int8x8_t, b: int8x8_t, b: int8x8_t, c: int8x8_t, d: int8x8_t) -> int8x8_t;

    #[link_name = "llvm.arm.neon.vtbx1"]
    fn vtbx1(a: int8x8_t, b: int8x8_t, b: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vtbx2"]
    fn vtbx2(a: int8x8_t, b: int8x8_t, b: int8x8_t, c: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vtbx3"]
    fn vtbx3(a: int8x8_t, b: int8x8_t, b: int8x8_t, c: int8x8_t, d: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vtbx4"]
    fn vtbx4(
        a: int8x8_t,
        b: int8x8_t,
        b: int8x8_t,
        c: int8x8_t,
        d: int8x8_t,
        e: int8x8_t,
    ) -> int8x8_t;
}

/// Unsigned saturating extract narrow.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqmovnu))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqxtn))]
pub unsafe fn vqmovn_u64(a: uint64x2_t) -> uint32x2_t {
    vqmovn_u64_(a)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vadd_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vadd_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vadd_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vadd_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vadd_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vadd_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fadd))]
pub unsafe fn vadd_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fadd))]
pub unsafe fn vaddq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_add(a, b)
}

/// Vector long add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vaddl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(saddl))]
pub unsafe fn vaddl_s8(a: int8x8_t, b: int8x8_t) -> int16x8_t {
    let a: int16x8_t = simd_cast(a);
    let b: int16x8_t = simd_cast(b);
    simd_add(a, b)
}

/// Vector long add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vaddl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(saddl))]
pub unsafe fn vaddl_s16(a: int16x4_t, b: int16x4_t) -> int32x4_t {
    let a: int32x4_t = simd_cast(a);
    let b: int32x4_t = simd_cast(b);
    simd_add(a, b)
}

/// Vector long add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vaddl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(saddl))]
pub unsafe fn vaddl_s32(a: int32x2_t, b: int32x2_t) -> int64x2_t {
    let a: int64x2_t = simd_cast(a);
    let b: int64x2_t = simd_cast(b);
    simd_add(a, b)
}

/// Vector long add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vaddl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uaddl))]
pub unsafe fn vaddl_u8(a: uint8x8_t, b: uint8x8_t) -> uint16x8_t {
    let a: uint16x8_t = simd_cast(a);
    let b: uint16x8_t = simd_cast(b);
    simd_add(a, b)
}

/// Vector long add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vaddl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uaddl))]
pub unsafe fn vaddl_u16(a: uint16x4_t, b: uint16x4_t) -> uint32x4_t {
    let a: uint32x4_t = simd_cast(a);
    let b: uint32x4_t = simd_cast(b);
    simd_add(a, b)
}

/// Vector long add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vaddl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uaddl))]
pub unsafe fn vaddl_u32(a: uint32x2_t, b: uint32x2_t) -> uint64x2_t {
    let a: uint64x2_t = simd_cast(a);
    let b: uint64x2_t = simd_cast(b);
    simd_add(a, b)
}

/// Vector narrow integer.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(xtn))]
pub unsafe fn vmovn_s16(a: int16x8_t) -> int8x8_t {
    simd_cast(a)
}

/// Vector narrow integer.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(xtn))]
pub unsafe fn vmovn_s32(a: int32x4_t) -> int16x4_t {
    simd_cast(a)
}

/// Vector narrow integer.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(xtn))]
pub unsafe fn vmovn_s64(a: int64x2_t) -> int32x2_t {
    simd_cast(a)
}

/// Vector narrow integer.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(xtn))]
pub unsafe fn vmovn_u16(a: uint16x8_t) -> uint8x8_t {
    simd_cast(a)
}

/// Vector narrow integer.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(xtn))]
pub unsafe fn vmovn_u32(a: uint32x4_t) -> uint16x4_t {
    simd_cast(a)
}

/// Vector narrow integer.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(xtn))]
pub unsafe fn vmovn_u64(a: uint64x2_t) -> uint32x2_t {
    simd_cast(a)
}

/// Vector long move.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sxtl))]
pub unsafe fn vmovl_s8(a: int8x8_t) -> int16x8_t {
    simd_cast(a)
}

/// Vector long move.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sxtl))]
pub unsafe fn vmovl_s16(a: int16x4_t) -> int32x4_t {
    simd_cast(a)
}

/// Vector long move.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sxtl))]
pub unsafe fn vmovl_s32(a: int32x2_t) -> int64x2_t {
    simd_cast(a)
}

/// Vector long move.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uxtl))]
pub unsafe fn vmovl_u8(a: uint8x8_t) -> uint16x8_t {
    simd_cast(a)
}

/// Vector long move.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uxtl))]
pub unsafe fn vmovl_u16(a: uint16x4_t) -> uint32x4_t {
    simd_cast(a)
}

/// Vector long move.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uxtl))]
pub unsafe fn vmovl_u32(a: uint32x2_t) -> uint64x2_t {
    simd_cast(a)
}

/// Reciprocal square-root estimate.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(frsqrte))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsqrte))]
pub unsafe fn vrsqrte_f32(a: float32x2_t) -> float32x2_t {
    frsqrte_v2f32(a)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvn_s8(a: int8x8_t) -> int8x8_t {
    let b = int8x8_t(-1, -1, -1, -1, -1, -1, -1, -1);
    simd_xor(a, b)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvnq_s8(a: int8x16_t) -> int8x16_t {
    let b = int8x16_t(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    simd_xor(a, b)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvn_s16(a: int16x4_t) -> int16x4_t {
    let b = int16x4_t(-1, -1, -1, -1);
    simd_xor(a, b)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvnq_s16(a: int16x8_t) -> int16x8_t {
    let b = int16x8_t(-1, -1, -1, -1, -1, -1, -1, -1);
    simd_xor(a, b)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvn_s32(a: int32x2_t) -> int32x2_t {
    let b = int32x2_t(-1, -1);
    simd_xor(a, b)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvnq_s32(a: int32x4_t) -> int32x4_t {
    let b = int32x4_t(-1, -1, -1, -1);
    simd_xor(a, b)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvn_u8(a: uint8x8_t) -> uint8x8_t {
    let b = uint8x8_t(255, 255, 255, 255, 255, 255, 255, 255);
    simd_xor(a, b)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvnq_u8(a: uint8x16_t) -> uint8x16_t {
    let b = uint8x16_t(
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    );
    simd_xor(a, b)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvn_u16(a: uint16x4_t) -> uint16x4_t {
    let b = uint16x4_t(65_535, 65_535, 65_535, 65_535);
    simd_xor(a, b)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvnq_u16(a: uint16x8_t) -> uint16x8_t {
    let b = uint16x8_t(
        65_535, 65_535, 65_535, 65_535, 65_535, 65_535, 65_535, 65_535,
    );
    simd_xor(a, b)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvn_u32(a: uint32x2_t) -> uint32x2_t {
    let b = uint32x2_t(4_294_967_295, 4_294_967_295);
    simd_xor(a, b)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvnq_u32(a: uint32x4_t) -> uint32x4_t {
    let b = uint32x4_t(4_294_967_295, 4_294_967_295, 4_294_967_295, 4_294_967_295);
    simd_xor(a, b)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvn_p8(a: poly8x8_t) -> poly8x8_t {
    let b = poly8x8_t(255, 255, 255, 255, 255, 255, 255, 255);
    simd_xor(a, b)
}

/// Vector bitwise not.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmvn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mvn))]
pub unsafe fn vmvnq_p8(a: poly8x16_t) -> poly8x16_t {
    let b = poly8x16_t(
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    );
    simd_xor(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sminp))]
pub unsafe fn vpmin_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    vpmins_v8i8(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sminp))]
pub unsafe fn vpmin_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    vpmins_v4i16(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sminp))]
pub unsafe fn vpmin_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    vpmins_v2i32(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uminp))]
pub unsafe fn vpmin_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    vpminu_v8i8(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uminp))]
pub unsafe fn vpmin_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    vpminu_v4i16(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uminp))]
pub unsafe fn vpmin_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    vpminu_v2i32(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fminp))]
pub unsafe fn vpmin_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    vpminf_v2f32(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smaxp))]
pub unsafe fn vpmax_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    vpmaxs_v8i8(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smaxp))]
pub unsafe fn vpmax_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    vpmaxs_v4i16(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smaxp))]
pub unsafe fn vpmax_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    vpmaxs_v2i32(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umaxp))]
pub unsafe fn vpmax_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    vpmaxu_v8i8(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umaxp))]
pub unsafe fn vpmax_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    vpmaxu_v4i16(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umaxp))]
pub unsafe fn vpmax_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    vpmaxu_v2i32(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmaxp))]
pub unsafe fn vpmax_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    vpmaxf_v2f32(a, b)
}

/// Table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl1_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    vtbl1(a, b)
}

/// Table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl1_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    transmute(vtbl1(transmute(a), transmute(b)))
}

/// Table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl1_p8(a: poly8x8_t, b: uint8x8_t) -> poly8x8_t {
    transmute(vtbl1(transmute(a), transmute(b)))
}

/// Table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl2_s8(a: int8x8x2_t, b: int8x8_t) -> int8x8_t {
    vtbl2(a.0, a.1, b)
}

/// Table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl2_u8(a: uint8x8x2_t, b: uint8x8_t) -> uint8x8_t {
    transmute(vtbl2(transmute(a.0), transmute(a.1), transmute(b)))
}

/// Table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl2_p8(a: poly8x8x2_t, b: uint8x8_t) -> poly8x8_t {
    transmute(vtbl2(transmute(a.0), transmute(a.1), transmute(b)))
}

/// Table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl3_s8(a: int8x8x3_t, b: int8x8_t) -> int8x8_t {
    vtbl3(a.0, a.1, a.2, b)
}

/// Table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl3_u8(a: uint8x8x3_t, b: uint8x8_t) -> uint8x8_t {
    transmute(vtbl3(
        transmute(a.0),
        transmute(a.1),
        transmute(a.2),
        transmute(b),
    ))
}

/// Table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl3_p8(a: poly8x8x3_t, b: uint8x8_t) -> poly8x8_t {
    transmute(vtbl3(
        transmute(a.0),
        transmute(a.1),
        transmute(a.2),
        transmute(b),
    ))
}

/// Table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl4_s8(a: int8x8x4_t, b: int8x8_t) -> int8x8_t {
    vtbl4(a.0, a.1, a.2, a.3, b)
}

/// Table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl4_u8(a: uint8x8x4_t, b: uint8x8_t) -> uint8x8_t {
    transmute(vtbl4(
        transmute(a.0),
        transmute(a.1),
        transmute(a.2),
        transmute(a.3),
        transmute(b),
    ))
}

/// Table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl4_p8(a: poly8x8x4_t, b: uint8x8_t) -> poly8x8_t {
    transmute(vtbl4(
        transmute(a.0),
        transmute(a.1),
        transmute(a.2),
        transmute(a.3),
        transmute(b),
    ))
}

/// Extended table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx1_s8(a: int8x8_t, b: int8x8_t, c: int8x8_t) -> int8x8_t {
    vtbx1(a, b, c)
}

/// Extended table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx1_u8(a: uint8x8_t, b: uint8x8_t, c: uint8x8_t) -> uint8x8_t {
    transmute(vtbx1(transmute(a), transmute(b), transmute(c)))
}

/// Extended table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx1_p8(a: poly8x8_t, b: poly8x8_t, c: uint8x8_t) -> poly8x8_t {
    transmute(vtbx1(transmute(a), transmute(b), transmute(c)))
}

/// Extended table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx2_s8(a: int8x8_t, b: int8x8x2_t, c: int8x8_t) -> int8x8_t {
    vtbx2(a, b.0, b.1, c)
}

/// Extended table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx2_u8(a: uint8x8_t, b: uint8x8x2_t, c: uint8x8_t) -> uint8x8_t {
    transmute(vtbx2(
        transmute(a),
        transmute(b.0),
        transmute(b.1),
        transmute(c),
    ))
}

/// Extended table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx2_p8(a: poly8x8_t, b: poly8x8x2_t, c: uint8x8_t) -> poly8x8_t {
    transmute(vtbx2(
        transmute(a),
        transmute(b.0),
        transmute(b.1),
        transmute(c),
    ))
}

/// Extended table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx3_s8(a: int8x8_t, b: int8x8x3_t, c: int8x8_t) -> int8x8_t {
    vtbx3(a, b.0, b.1, b.2, c)
}

/// Extended table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx3_u8(a: uint8x8_t, b: uint8x8x3_t, c: uint8x8_t) -> uint8x8_t {
    transmute(vtbx3(
        transmute(a),
        transmute(b.0),
        transmute(b.1),
        transmute(b.2),
        transmute(c),
    ))
}

/// Extended table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx3_p8(a: poly8x8_t, b: poly8x8x3_t, c: uint8x8_t) -> poly8x8_t {
    transmute(vtbx3(
        transmute(a),
        transmute(b.0),
        transmute(b.1),
        transmute(b.2),
        transmute(c),
    ))
}

/// Extended table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx4_s8(a: int8x8_t, b: int8x8x4_t, c: int8x8_t) -> int8x8_t {
    vtbx4(a, b.0, b.1, b.2, b.3, c)
}

/// Extended table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx4_u8(a: uint8x8_t, b: uint8x8x4_t, c: uint8x8_t) -> uint8x8_t {
    transmute(vtbx4(
        transmute(a),
        transmute(b.0),
        transmute(b.1),
        transmute(b.2),
        transmute(b.3),
        transmute(c),
    ))
}

/// Extended table look-up
#[inline]
#[cfg(target_arch = "arm")]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx4_p8(a: poly8x8_t, b: poly8x8x4_t, c: uint8x8_t) -> poly8x8_t {
    transmute(vtbx4(
        transmute(a),
        transmute(b.0),
        transmute(b.1),
        transmute(b.2),
        transmute(b.3),
        transmute(c),
    ))
}

/// Move vector element to general-purpose register
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[rustc_args_required_const(1)]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(mov, imm5 = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mov, imm5 = 1))]
// Based on the discussioj in https://github.com/rust-lang/stdarch/pull/792
// `mov` seems to be an acceptable intrinsic to compile to
// #[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(vmov, imm5 = 1))]
pub unsafe fn vgetq_lane_u64(v: uint64x2_t, imm5: i32) -> u64 {
    if (imm5) < 0 || (imm5) > 1 {
        unreachable_unchecked()
    }
    let imm5 = (imm5 & 0b1) as u32;
    simd_extract(v, imm5)
}

/// Move vector element to general-purpose register
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[rustc_args_required_const(1)]
#[cfg_attr(test, assert_instr(fmov, imm5 = 0))]
// gcc also turns this into a fmov instead of a umove
// https://clang.godbolt.org/z/J5xS2T
// #[cfg_attr(test, assert_instr(umov, imm5 = 0))]
pub unsafe fn vget_lane_u64(v: uint64x1_t, imm5: i32) -> u64 {
    if imm5 != 0 {
        unreachable_unchecked()
    }
    simd_extract(v, 0)
}

/// Move vector element to general-purpose register
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[rustc_args_required_const(1)]
#[cfg_attr(test, assert_instr(umov, imm5 = 0))]
pub unsafe fn vgetq_lane_u16(v: uint16x8_t, imm5: i32) -> u16 {
    if (imm5) < 0 || (imm5) > 7 {
        unreachable_unchecked()
    }
    let imm5 = (imm5 & 0b111) as u32;
    simd_extract(v, imm5)
}

/// Move vector element to general-purpose register
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[rustc_args_required_const(1)]
// see: https://clang.godbolt.org/z/J5xS2T
//#[cfg_attr(test, assert_instr(umov, imm5 = 0))]
#[cfg_attr(test, assert_instr(fmov, imm5 = 0))]
pub unsafe fn vgetq_lane_u32(v: uint32x4_t, imm5: i32) -> u32 {
    if (imm5) < 0 || (imm5) > 3 {
        unreachable_unchecked()
    }
    let imm5 = (imm5 & 0b11) as u32;
    simd_extract(v, imm5)
}

/// Move vector element to general-purpose register
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[rustc_args_required_const(1)]
#[cfg_attr(test, assert_instr(umov, imm5 = 0))]
pub unsafe fn vget_lane_u8(v: uint8x8_t, imm5: i32) -> u8 {
    if (imm5) < 0 || (imm5) > 7 {
        unreachable_unchecked()
    }
    let imm5 = (imm5 & 7) as u32;
    simd_extract(v, imm5)
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(dup))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup))]
pub unsafe fn vdupq_n_s8(value: i8) -> int8x16_t {
    int8x16_t(
        value, value, value, value, value, value, value, value, value, value, value, value, value,
        value, value, value,
    )
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(dup))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup))]
pub unsafe fn vdupq_n_u8(value: u8) -> uint8x16_t {
    uint8x16_t(
        value, value, value, value, value, value, value, value, value, value, value, value, value,
        value, value, value,
    )
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(dup))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup))]
pub unsafe fn vmovq_n_u8(value: u8) -> uint8x16_t {
    vdupq_n_u8(value)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_u64_u32(a: uint32x2_t) -> uint64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_s8_u8(a: uint8x16_t) -> int8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_u16_u8(a: uint8x16_t) -> uint16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_u32_u8(a: uint8x16_t) -> uint32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_u64_u8(a: uint8x16_t) -> uint64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_u8_s8(a: int8x16_t) -> uint8x16_t {
    transmute(a)
}

/// Unsigned shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(ushr, imm3 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn vshrq_n_u8(a: uint8x16_t, imm3: i32) -> uint8x16_t {
    if imm3 < 0 || imm3 > 7 {
        unreachable_unchecked();
    } else {
        uint8x16_t(
            a.0 >> imm3,
            a.1 >> imm3,
            a.2 >> imm3,
            a.3 >> imm3,
            a.4 >> imm3,
            a.5 >> imm3,
            a.6 >> imm3,
            a.7 >> imm3,
            a.8 >> imm3,
            a.9 >> imm3,
            a.10 >> imm3,
            a.11 >> imm3,
            a.12 >> imm3,
            a.13 >> imm3,
            a.14 >> imm3,
            a.15 >> imm3,
        )
    }
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(shl, imm3 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn vshlq_n_u8(a: uint8x16_t, imm3: i32) -> uint8x16_t {
    if imm3 < 0 || imm3 > 7 {
        unreachable_unchecked();
    } else {
        uint8x16_t(
            a.0 << imm3,
            a.1 << imm3,
            a.2 << imm3,
            a.3 << imm3,
            a.4 << imm3,
            a.5 << imm3,
            a.6 << imm3,
            a.7 << imm3,
            a.8 << imm3,
            a.9 << imm3,
            a.10 << imm3,
            a.11 << imm3,
            a.12 << imm3,
            a.13 << imm3,
            a.14 << imm3,
            a.15 << imm3,
        )
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(ext, n = 3))]
#[rustc_args_required_const(2)]
pub unsafe fn vextq_s8(a: int8x16_t, b: int8x16_t, n: i32) -> int8x16_t {
    if n < 0 || n > 15 {
        unreachable_unchecked();
    };
    match n & 0b1111 {
        0 => simd_shuffle16(a, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        1 => simd_shuffle16(
            a,
            b,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ),
        2 => simd_shuffle16(
            a,
            b,
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        ),
        3 => simd_shuffle16(
            a,
            b,
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        ),
        4 => simd_shuffle16(
            a,
            b,
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ),
        5 => simd_shuffle16(
            a,
            b,
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        ),
        6 => simd_shuffle16(
            a,
            b,
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        ),
        7 => simd_shuffle16(
            a,
            b,
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        ),
        8 => simd_shuffle16(
            a,
            b,
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        ),
        9 => simd_shuffle16(
            a,
            b,
            [
                9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            ],
        ),
        10 => simd_shuffle16(
            a,
            b,
            [
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            ],
        ),
        11 => simd_shuffle16(
            a,
            b,
            [
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            ],
        ),
        12 => simd_shuffle16(
            a,
            b,
            [
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            ],
        ),
        13 => simd_shuffle16(
            a,
            b,
            [
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            ],
        ),
        14 => simd_shuffle16(
            a,
            b,
            [
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            ],
        ),
        15 => simd_shuffle16(
            a,
            b,
            [
                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            ],
        ),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(ext, n = 3))]
#[rustc_args_required_const(2)]
pub unsafe fn vextq_u8(a: uint8x16_t, b: uint8x16_t, n: i32) -> uint8x16_t {
    if n < 0 || n > 15 {
        unreachable_unchecked();
    };
    match n & 0b1111 {
        0 => simd_shuffle16(a, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        1 => simd_shuffle16(
            a,
            b,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ),
        2 => simd_shuffle16(
            a,
            b,
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        ),
        3 => simd_shuffle16(
            a,
            b,
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        ),
        4 => simd_shuffle16(
            a,
            b,
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ),
        5 => simd_shuffle16(
            a,
            b,
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        ),
        6 => simd_shuffle16(
            a,
            b,
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        ),
        7 => simd_shuffle16(
            a,
            b,
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        ),
        8 => simd_shuffle16(
            a,
            b,
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        ),
        9 => simd_shuffle16(
            a,
            b,
            [
                9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            ],
        ),
        10 => simd_shuffle16(
            a,
            b,
            [
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            ],
        ),
        11 => simd_shuffle16(
            a,
            b,
            [
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            ],
        ),
        12 => simd_shuffle16(
            a,
            b,
            [
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            ],
        ),
        13 => simd_shuffle16(
            a,
            b,
            [
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            ],
        ),
        14 => simd_shuffle16(
            a,
            b,
            [
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            ],
        ),
        15 => simd_shuffle16(
            a,
            b,
            [
                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            ],
        ),
        _ => unreachable_unchecked(),
    }
}

/// Load multiple single-element structures to one, two, three, or four registers
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(ldr))]
// even gcc compiles this to ldr: https://clang.godbolt.org/z/1bvH2x
// #[cfg_attr(test, assert_instr(ld1))]
pub unsafe fn vld1q_s8(addr: *const i8) -> int8x16_t {
    ptr::read(addr as *const int8x16_t)
}

/// Load multiple single-element structures to one, two, three, or four registers
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(ldr))]
// even gcc compiles this to ldr: https://clang.godbolt.org/z/1bvH2x
// #[cfg_attr(test, assert_instr(ld1))]
pub unsafe fn vld1q_u8(addr: *const u8) -> uint8x16_t {
    ptr::read(addr as *const uint8x16_t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_arch::{arm::*, simd::*};
    use std::{i16, i32, i8, mem::transmute, u16, u32, u8, vec::Vec};
    use stdarch_test::simd_test;

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_s8() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e = a;
        let r: i8x16 = transmute(vld1q_s8(transmute(&a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_u8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e = a;
        let r: u8x16 = transmute(vld1q_u8(transmute(&a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_u8() {
        let v = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = vget_lane_u8(transmute(v), 1);
        assert_eq!(r, 2);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_u32() {
        let v = i32x4::new(1, 2, 3, 4);
        let r = vgetq_lane_u32(transmute(v), 1);
        assert_eq!(r, 2);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_u64() {
        let v: u64 = 1;
        let r = vget_lane_u64(transmute(v), 0);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_u16() {
        let v = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = vgetq_lane_u16(transmute(v), 1);
        assert_eq!(r, 2);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_s8() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = i8x16::new(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 31, 32,
        );
        let e = i8x16::new(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19);
        let r: i8x16 = transmute(vextq_s8(transmute(a), transmute(b), 3));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_u8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = u8x16::new(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 31, 32,
        );
        let e = u8x16::new(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19);
        let r: u8x16 = transmute(vextq_u8(transmute(a), transmute(b), 3));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrq_n_u8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e = u8x16::new(0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4);
        let r: u8x16 = transmute(vshrq_n_u8(transmute(a), 2));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_n_u8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e = u8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let r: u8x16 = transmute(vshlq_n_u8(transmute(a), 2));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovn_u64() {
        let a = u64x2::new(1, 2);
        let e = u32x2::new(1, 2);
        let r: u32x2 = transmute(vqmovn_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u64_u32() {
        let v: i8 = 42;
        let e = i8x16::new(
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        );
        let r: i8x16 = transmute(vdupq_n_s8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_s8() {
        let v: i8 = 42;
        let e = i8x16::new(
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        );
        let r: i8x16 = transmute(vdupq_n_s8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_u8() {
        let v: u8 = 42;
        let e = u8x16::new(
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        );
        let r: u8x16 = transmute(vdupq_n_u8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_u8() {
        let v: u8 = 42;
        let e = u8x16::new(
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        );
        let r: u8x16 = transmute(vmovq_n_u8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_u64() {
        let v = i64x2::new(1, 2);
        let r = vgetq_lane_u64(transmute(v), 1);
        assert_eq!(r, 2);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_s8() {
        test_ari_s8(
            |i, j| vadd_s8(i, j),
            |a: i8, b: i8| -> i8 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_s8() {
        testq_ari_s8(
            |i, j| vaddq_s8(i, j),
            |a: i8, b: i8| -> i8 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_s16() {
        test_ari_s16(
            |i, j| vadd_s16(i, j),
            |a: i16, b: i16| -> i16 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_s16() {
        testq_ari_s16(
            |i, j| vaddq_s16(i, j),
            |a: i16, b: i16| -> i16 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_s32() {
        test_ari_s32(
            |i, j| vadd_s32(i, j),
            |a: i32, b: i32| -> i32 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_s32() {
        testq_ari_s32(
            |i, j| vaddq_s32(i, j),
            |a: i32, b: i32| -> i32 { a.overflowing_add(b).0 },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_u8() {
        test_ari_u8(
            |i, j| vadd_u8(i, j),
            |a: u8, b: u8| -> u8 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_u8() {
        testq_ari_u8(
            |i, j| vaddq_u8(i, j),
            |a: u8, b: u8| -> u8 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_u16() {
        test_ari_u16(
            |i, j| vadd_u16(i, j),
            |a: u16, b: u16| -> u16 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_u16() {
        testq_ari_u16(
            |i, j| vaddq_u16(i, j),
            |a: u16, b: u16| -> u16 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_u32() {
        test_ari_u32(
            |i, j| vadd_u32(i, j),
            |a: u32, b: u32| -> u32 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_u32() {
        testq_ari_u32(
            |i, j| vaddq_u32(i, j),
            |a: u32, b: u32| -> u32 { a.overflowing_add(b).0 },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_f32() {
        test_ari_f32(|i, j| vadd_f32(i, j), |a: f32, b: f32| -> f32 { a + b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_f32() {
        testq_ari_f32(|i, j| vaddq_f32(i, j), |a: f32, b: f32| -> f32 { a + b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_s8() {
        let v = i8::MAX;
        let a = i8x8::new(v, v, v, v, v, v, v, v);
        let v = 2 * (v as i16);
        let e = i16x8::new(v, v, v, v, v, v, v, v);
        let r: i16x8 = transmute(vaddl_s8(transmute(a), transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_s16() {
        let v = i16::MAX;
        let a = i16x4::new(v, v, v, v);
        let v = 2 * (v as i32);
        let e = i32x4::new(v, v, v, v);
        let r: i32x4 = transmute(vaddl_s16(transmute(a), transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_s32() {
        let v = i32::MAX;
        let a = i32x2::new(v, v);
        let v = 2 * (v as i64);
        let e = i64x2::new(v, v);
        let r: i64x2 = transmute(vaddl_s32(transmute(a), transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_u8() {
        let v = u8::MAX;
        let a = u8x8::new(v, v, v, v, v, v, v, v);
        let v = 2 * (v as u16);
        let e = u16x8::new(v, v, v, v, v, v, v, v);
        let r: u16x8 = transmute(vaddl_u8(transmute(a), transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_u16() {
        let v = u16::MAX;
        let a = u16x4::new(v, v, v, v);
        let v = 2 * (v as u32);
        let e = u32x4::new(v, v, v, v);
        let r: u32x4 = transmute(vaddl_u16(transmute(a), transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_u32() {
        let v = u32::MAX;
        let a = u32x2::new(v, v);
        let v = 2 * (v as u64);
        let e = u64x2::new(v, v);
        let r: u64x2 = transmute(vaddl_u32(transmute(a), transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_s8() {
        let a = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e = i8x8::new(-1, -2, -3, -4, -5, -6, -7, -8);
        let r: i8x8 = transmute(vmvn_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_s8() {
        let a = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e = i8x16::new(
            -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
        );
        let r: i8x16 = transmute(vmvnq_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_s16() {
        let a = i16x4::new(0, 1, 2, 3);
        let e = i16x4::new(-1, -2, -3, -4);
        let r: i16x4 = transmute(vmvn_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_s16() {
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e = i16x8::new(-1, -2, -3, -4, -5, -6, -7, -8);
        let r: i16x8 = transmute(vmvnq_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_s32() {
        let a = i32x2::new(0, 1);
        let e = i32x2::new(-1, -2);
        let r: i32x2 = transmute(vmvn_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_s32() {
        let a = i32x4::new(0, 1, 2, 3);
        let e = i32x4::new(-1, -2, -3, -4);
        let r: i32x4 = transmute(vmvnq_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_u8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e = u8x8::new(255, 254, 253, 252, 251, 250, 249, 248);
        let r: u8x8 = transmute(vmvn_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_u8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e = u8x16::new(
            255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240,
        );
        let r: u8x16 = transmute(vmvnq_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_u16() {
        let a = u16x4::new(0, 1, 2, 3);
        let e = u16x4::new(65_535, 65_534, 65_533, 65_532);
        let r: u16x4 = transmute(vmvn_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_u16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e = u16x8::new(
            65_535, 65_534, 65_533, 65_532, 65_531, 65_530, 65_529, 65_528,
        );
        let r: u16x8 = transmute(vmvnq_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_u32() {
        let a = u32x2::new(0, 1);
        let e = u32x2::new(4_294_967_295, 4_294_967_294);
        let r: u32x2 = transmute(vmvn_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_u32() {
        let a = u32x4::new(0, 1, 2, 3);
        let e = u32x4::new(4_294_967_295, 4_294_967_294, 4_294_967_293, 4_294_967_292);
        let r: u32x4 = transmute(vmvnq_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_p8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e = u8x8::new(255, 254, 253, 252, 251, 250, 249, 248);
        let r: u8x8 = transmute(vmvn_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_p8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e = u8x16::new(
            255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240,
        );
        let r: u8x16 = transmute(vmvnq_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_s16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vmovn_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_s32() {
        let a = i32x4::new(1, 2, 3, 4);
        let e = i16x4::new(1, 2, 3, 4);
        let r: i16x4 = transmute(vmovn_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_s64() {
        let a = i64x2::new(1, 2);
        let e = i32x2::new(1, 2);
        let r: i32x2 = transmute(vmovn_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = transmute(vmovn_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let e = u16x4::new(1, 2, 3, 4);
        let r: u16x4 = transmute(vmovn_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_u64() {
        let a = u64x2::new(1, 2);
        let e = u32x2::new(1, 2);
        let r: u32x2 = transmute(vmovn_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovl_s8() {
        let e = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let a = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i16x8 = transmute(vmovl_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovl_s16() {
        let e = i32x4::new(1, 2, 3, 4);
        let a = i16x4::new(1, 2, 3, 4);
        let r: i32x4 = transmute(vmovl_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovl_s32() {
        let e = i64x2::new(1, 2);
        let a = i32x2::new(1, 2);
        let r: i64x2 = transmute(vmovl_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovl_u8() {
        let e = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u16x8 = transmute(vmovl_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovl_u16() {
        let e = u32x4::new(1, 2, 3, 4);
        let a = u16x4::new(1, 2, 3, 4);
        let r: u32x4 = transmute(vmovl_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovl_u32() {
        let e = u64x2::new(1, 2);
        let a = u32x2::new(1, 2);
        let r: u64x2 = transmute(vmovl_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsqrt_f32() {
        let a = f32x2::new(1.0, 2.0);
        let e = f32x2::new(0.9980469, 0.7050781);
        let r: f32x2 = transmute(vrsqrte_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmin_s8() {
        let a = i8x8::new(1, -2, 3, -4, 5, 6, 7, 8);
        let b = i8x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = i8x8::new(-2, -4, 5, 7, 0, 2, 4, 6);
        let r: i8x8 = transmute(vpmin_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmin_s16() {
        let a = i16x4::new(1, 2, 3, -4);
        let b = i16x4::new(0, 3, 2, 5);
        let e = i16x4::new(1, -4, 0, 2);
        let r: i16x4 = transmute(vpmin_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmin_s32() {
        let a = i32x2::new(1, -2);
        let b = i32x2::new(0, 3);
        let e = i32x2::new(-2, 0);
        let r: i32x2 = transmute(vpmin_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmin_u8() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = u8x8::new(1, 3, 5, 7, 0, 2, 4, 6);
        let r: u8x8 = transmute(vpmin_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmin_u16() {
        let a = u16x4::new(1, 2, 3, 4);
        let b = u16x4::new(0, 3, 2, 5);
        let e = u16x4::new(1, 3, 0, 2);
        let r: u16x4 = transmute(vpmin_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmin_u32() {
        let a = u32x2::new(1, 2);
        let b = u32x2::new(0, 3);
        let e = u32x2::new(1, 0);
        let r: u32x2 = transmute(vpmin_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmin_f32() {
        let a = f32x2::new(1., -2.);
        let b = f32x2::new(0., 3.);
        let e = f32x2::new(-2., 0.);
        let r: f32x2 = transmute(vpmin_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmax_s8() {
        let a = i8x8::new(1, -2, 3, -4, 5, 6, 7, 8);
        let b = i8x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = i8x8::new(1, 3, 6, 8, 3, 5, 7, 9);
        let r: i8x8 = transmute(vpmax_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmax_s16() {
        let a = i16x4::new(1, 2, 3, -4);
        let b = i16x4::new(0, 3, 2, 5);
        let e = i16x4::new(2, 3, 3, 5);
        let r: i16x4 = transmute(vpmax_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmax_s32() {
        let a = i32x2::new(1, -2);
        let b = i32x2::new(0, 3);
        let e = i32x2::new(1, 3);
        let r: i32x2 = transmute(vpmax_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmax_u8() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = u8x8::new(2, 4, 6, 8, 3, 5, 7, 9);
        let r: u8x8 = transmute(vpmax_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmax_u16() {
        let a = u16x4::new(1, 2, 3, 4);
        let b = u16x4::new(0, 3, 2, 5);
        let e = u16x4::new(2, 4, 3, 5);
        let r: u16x4 = transmute(vpmax_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmax_u32() {
        let a = u32x2::new(1, 2);
        let b = u32x2::new(0, 3);
        let e = u32x2::new(2, 3);
        let r: u32x2 = transmute(vpmax_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmax_f32() {
        let a = f32x2::new(1., -2.);
        let b = f32x2::new(0., 3.);
        let e = f32x2::new(1., 3.);
        let r: f32x2 = transmute(vpmax_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    macro_rules! V_u8 {
        () => {
            vec![0x00u8, 0x01u8, 0x02u8, 0x0Fu8, 0x80u8, 0xF0u8, 0xFFu8]
        };
    }
    macro_rules! V_u16 {
        () => {
            vec![
                0x0000u16, 0x0101u16, 0x0202u16, 0x0F0Fu16, 0x8000u16, 0xF0F0u16, 0xFFFFu16,
            ]
        };
    }
    macro_rules! V_u32 {
        () => {
            vec![
                0x00000000u32,
                0x01010101u32,
                0x02020202u32,
                0x0F0F0F0Fu32,
                0x80000000u32,
                0xF0F0F0F0u32,
                0xFFFFFFFFu32,
            ]
        };
    }
    macro_rules! V_u64 {
        () => {
            vec![
                0x0000000000000000u64,
                0x0101010101010101u64,
                0x0202020202020202u64,
                0x0F0F0F0F0F0F0F0Fu64,
                0x8080808080808080u64,
                0xF0F0F0F0F0F0F0F0u64,
                0xFFFFFFFFFFFFFFFFu64,
            ]
        };
    }

    macro_rules! V_i8 {
        () => {
            vec![
                0x00i8, 0x01i8, 0x02i8, 0x0Fi8, -128i8, /* 0x80 */
                -16i8,  /* 0xF0 */
                -1i8,   /* 0xFF */
            ]
        };
    }
    macro_rules! V_i16 {
        () => {
            vec![
                0x0000i16, 0x0101i16, 0x0202i16, 0x0F0Fi16, -32768i16, /* 0x8000 */
                -3856i16,  /* 0xF0F0 */
                -1i16,     /* 0xFFF */
            ]
        };
    }
    macro_rules! V_i32 {
        () => {
            vec![
                0x00000000i32,
                0x01010101i32,
                0x02020202i32,
                0x0F0F0F0Fi32,
                -2139062144i32, /* 0x80000000 */
                -252645136i32,  /* 0xF0F0F0F0 */
                -1i32,          /* 0xFFFFFFFF */
            ]
        };
    }

    macro_rules! V_i64 {
        () => {
            vec![
                0x0000000000000000i64,
                0x0101010101010101i64,
                0x0202020202020202i64,
                0x0F0F0F0F0F0F0F0Fi64,
                -9223372036854775808i64, /* 0x8000000000000000 */
                -1152921504606846976i64, /* 0xF000000000000000 */
                -1i64,                   /* 0xFFFFFFFFFFFFFFFF */
            ]
        };
    }

    macro_rules! V_f32 {
        () => {
            vec![
                0.0f32,
                1.0f32,
                -1.0f32,
                1.2f32,
                2.4f32,
                std::f32::MAX,
                std::f32::MIN,
                std::f32::INFINITY,
                std::f32::NEG_INFINITY,
                std::f32::NAN,
            ]
        };
    }

    macro_rules! to64 {
        ($t : ident) => {
            |v: $t| -> u64 { transmute(v) }
        };
    }

    macro_rules! to128 {
        ($t : ident) => {
            |v: $t| -> u128 { transmute(v) }
        };
    }

    unsafe fn test<T, U, V, W, X>(
        vals: Vec<T>,
        fill1: fn(T) -> V,
        fill2: fn(U) -> W,
        cast: fn(W) -> X,
        test_fun: fn(V, V) -> W,
        verify_fun: fn(T, T) -> U,
    ) where
        T: Copy + core::fmt::Debug + std::cmp::PartialEq,
        U: Copy + core::fmt::Debug + std::cmp::PartialEq,
        V: Copy + core::fmt::Debug,
        W: Copy + core::fmt::Debug,
        X: Copy + core::fmt::Debug + std::cmp::PartialEq,
    {
        let pairs = vals.iter().zip(vals.iter());

        for (i, j) in pairs {
            let a: V = fill1(*i);
            let b: V = fill1(*j);

            let actual_pre: W = test_fun(a, b);
            let expected_pre: W = fill2(verify_fun(*i, *j));

            let actual: X = cast(actual_pre);
            let expected: X = cast(expected_pre);

            assert_eq!(
                actual, expected,
                "[{:?}:{:?}] :\nf({:?}, {:?}) = {:?}\ng({:?}, {:?}) = {:?}\n",
                *i, *j, &a, &b, actual_pre, &a, &b, expected_pre
            );
        }
    }

    macro_rules! gen_test_fn {
        ($n: ident, $t: ident, $u: ident, $v: ident, $w: ident, $x: ident, $vals: expr, $fill1: expr, $fill2: expr, $cast: expr) => {
            unsafe fn $n(test_fun: fn($v, $v) -> $w, verify_fun: fn($t, $t) -> $u) {
                test::<$t, $u, $v, $w, $x>($vals, $fill1, $fill2, $cast, test_fun, verify_fun);
            }
        };
    }

    macro_rules! gen_fill_fn {
        ($id: ident, $el_width: expr, $num_els: expr, $in_t : ident, $out_t: ident, $cmp_t: ident) => {
            fn $id(val: $in_t) -> $out_t {
                let initial: [$in_t; $num_els] = [val; $num_els];
                let result: $cmp_t = unsafe { transmute(initial) };
                let result_out: $out_t = unsafe { transmute(result) };

                // println!("FILL: {:016x} as {} x {}: {:016x}", val.reverse_bits(), $el_width, $num_els, (result as u64).reverse_bits());

                result_out
            }
        };
    }

    gen_fill_fn!(fill_u8, 8, 8, u8, uint8x8_t, u64);
    gen_fill_fn!(fill_s8, 8, 8, i8, int8x8_t, u64);
    gen_fill_fn!(fillq_u8, 8, 16, u8, uint8x16_t, u128);
    gen_fill_fn!(fillq_s8, 8, 16, i8, int8x16_t, u128);

    gen_fill_fn!(fill_u16, 16, 4, u16, uint16x4_t, u64);
    gen_fill_fn!(fill_s16, 16, 4, i16, int16x4_t, u64);
    gen_fill_fn!(fillq_u16, 16, 8, u16, uint16x8_t, u128);
    gen_fill_fn!(fillq_s16, 16, 8, i16, int16x8_t, u128);

    gen_fill_fn!(fill_u32, 32, 2, u32, uint32x2_t, u64);
    gen_fill_fn!(fill_s32, 32, 2, i32, int32x2_t, u64);
    gen_fill_fn!(fillq_u32, 32, 4, u32, uint32x4_t, u128);
    gen_fill_fn!(fillq_s32, 32, 4, i32, int32x4_t, u128);

    gen_fill_fn!(fill_u64, 64, 1, u64, uint64x1_t, u64);
    gen_fill_fn!(fill_s64, 64, 1, i64, int64x1_t, u64);
    gen_fill_fn!(fillq_u64, 64, 2, u64, uint64x2_t, u128);
    gen_fill_fn!(fillq_s64, 64, 2, i64, int64x2_t, u128);

    gen_fill_fn!(fill_f32, 32, 2, f32, float32x2_t, u64);
    gen_fill_fn!(fillq_f32, 32, 4, f32, float32x4_t, u128);

    gen_test_fn!(
        test_ari_u8,
        u8,
        u8,
        uint8x8_t,
        uint8x8_t,
        u64,
        V_u8!(),
        fill_u8,
        fill_u8,
        to64!(uint8x8_t)
    );
    gen_test_fn!(
        test_bit_u8,
        u8,
        u8,
        uint8x8_t,
        uint8x8_t,
        u64,
        V_u8!(),
        fill_u8,
        fill_u8,
        to64!(uint8x8_t)
    );
    gen_test_fn!(
        test_cmp_u8,
        u8,
        u8,
        uint8x8_t,
        uint8x8_t,
        u64,
        V_u8!(),
        fill_u8,
        fill_u8,
        to64!(uint8x8_t)
    );
    gen_test_fn!(
        testq_ari_u8,
        u8,
        u8,
        uint8x16_t,
        uint8x16_t,
        u128,
        V_u8!(),
        fillq_u8,
        fillq_u8,
        to128!(uint8x16_t)
    );
    gen_test_fn!(
        testq_bit_u8,
        u8,
        u8,
        uint8x16_t,
        uint8x16_t,
        u128,
        V_u8!(),
        fillq_u8,
        fillq_u8,
        to128!(uint8x16_t)
    );
    gen_test_fn!(
        testq_cmp_u8,
        u8,
        u8,
        uint8x16_t,
        uint8x16_t,
        u128,
        V_u8!(),
        fillq_u8,
        fillq_u8,
        to128!(uint8x16_t)
    );

    gen_test_fn!(
        test_ari_s8,
        i8,
        i8,
        int8x8_t,
        int8x8_t,
        u64,
        V_i8!(),
        fill_s8,
        fill_s8,
        to64!(int8x8_t)
    );
    gen_test_fn!(
        test_bit_s8,
        i8,
        i8,
        int8x8_t,
        int8x8_t,
        u64,
        V_i8!(),
        fill_s8,
        fill_s8,
        to64!(int8x8_t)
    );
    gen_test_fn!(
        test_cmp_s8,
        i8,
        u8,
        int8x8_t,
        uint8x8_t,
        u64,
        V_i8!(),
        fill_s8,
        fill_u8,
        to64!(uint8x8_t)
    );
    gen_test_fn!(
        testq_ari_s8,
        i8,
        i8,
        int8x16_t,
        int8x16_t,
        u128,
        V_i8!(),
        fillq_s8,
        fillq_s8,
        to128!(int8x16_t)
    );
    gen_test_fn!(
        testq_bit_s8,
        i8,
        i8,
        int8x16_t,
        int8x16_t,
        u128,
        V_i8!(),
        fillq_s8,
        fillq_s8,
        to128!(int8x16_t)
    );
    gen_test_fn!(
        testq_cmp_s8,
        i8,
        u8,
        int8x16_t,
        uint8x16_t,
        u128,
        V_i8!(),
        fillq_s8,
        fillq_u8,
        to128!(uint8x16_t)
    );

    gen_test_fn!(
        test_ari_u16,
        u16,
        u16,
        uint16x4_t,
        uint16x4_t,
        u64,
        V_u16!(),
        fill_u16,
        fill_u16,
        to64!(uint16x4_t)
    );
    gen_test_fn!(
        test_bit_u16,
        u16,
        u16,
        uint16x4_t,
        uint16x4_t,
        u64,
        V_u16!(),
        fill_u16,
        fill_u16,
        to64!(uint16x4_t)
    );
    gen_test_fn!(
        test_cmp_u16,
        u16,
        u16,
        uint16x4_t,
        uint16x4_t,
        u64,
        V_u16!(),
        fill_u16,
        fill_u16,
        to64!(uint16x4_t)
    );
    gen_test_fn!(
        testq_ari_u16,
        u16,
        u16,
        uint16x8_t,
        uint16x8_t,
        u128,
        V_u16!(),
        fillq_u16,
        fillq_u16,
        to128!(uint16x8_t)
    );
    gen_test_fn!(
        testq_bit_u16,
        u16,
        u16,
        uint16x8_t,
        uint16x8_t,
        u128,
        V_u16!(),
        fillq_u16,
        fillq_u16,
        to128!(uint16x8_t)
    );
    gen_test_fn!(
        testq_cmp_u16,
        u16,
        u16,
        uint16x8_t,
        uint16x8_t,
        u128,
        V_u16!(),
        fillq_u16,
        fillq_u16,
        to128!(uint16x8_t)
    );

    gen_test_fn!(
        test_ari_s16,
        i16,
        i16,
        int16x4_t,
        int16x4_t,
        u64,
        V_i16!(),
        fill_s16,
        fill_s16,
        to64!(int16x4_t)
    );
    gen_test_fn!(
        test_bit_s16,
        i16,
        i16,
        int16x4_t,
        int16x4_t,
        u64,
        V_i16!(),
        fill_s16,
        fill_s16,
        to64!(int16x4_t)
    );
    gen_test_fn!(
        test_cmp_s16,
        i16,
        u16,
        int16x4_t,
        uint16x4_t,
        u64,
        V_i16!(),
        fill_s16,
        fill_u16,
        to64!(uint16x4_t)
    );
    gen_test_fn!(
        testq_ari_s16,
        i16,
        i16,
        int16x8_t,
        int16x8_t,
        u128,
        V_i16!(),
        fillq_s16,
        fillq_s16,
        to128!(int16x8_t)
    );
    gen_test_fn!(
        testq_bit_s16,
        i16,
        i16,
        int16x8_t,
        int16x8_t,
        u128,
        V_i16!(),
        fillq_s16,
        fillq_s16,
        to128!(int16x8_t)
    );
    gen_test_fn!(
        testq_cmp_s16,
        i16,
        u16,
        int16x8_t,
        uint16x8_t,
        u128,
        V_i16!(),
        fillq_s16,
        fillq_u16,
        to128!(uint16x8_t)
    );

    gen_test_fn!(
        test_ari_u32,
        u32,
        u32,
        uint32x2_t,
        uint32x2_t,
        u64,
        V_u32!(),
        fill_u32,
        fill_u32,
        to64!(uint32x2_t)
    );
    gen_test_fn!(
        test_bit_u32,
        u32,
        u32,
        uint32x2_t,
        uint32x2_t,
        u64,
        V_u32!(),
        fill_u32,
        fill_u32,
        to64!(uint32x2_t)
    );
    gen_test_fn!(
        test_cmp_u32,
        u32,
        u32,
        uint32x2_t,
        uint32x2_t,
        u64,
        V_u32!(),
        fill_u32,
        fill_u32,
        to64!(uint32x2_t)
    );
    gen_test_fn!(
        testq_ari_u32,
        u32,
        u32,
        uint32x4_t,
        uint32x4_t,
        u128,
        V_u32!(),
        fillq_u32,
        fillq_u32,
        to128!(uint32x4_t)
    );
    gen_test_fn!(
        testq_bit_u32,
        u32,
        u32,
        uint32x4_t,
        uint32x4_t,
        u128,
        V_u32!(),
        fillq_u32,
        fillq_u32,
        to128!(uint32x4_t)
    );
    gen_test_fn!(
        testq_cmp_u32,
        u32,
        u32,
        uint32x4_t,
        uint32x4_t,
        u128,
        V_u32!(),
        fillq_u32,
        fillq_u32,
        to128!(uint32x4_t)
    );

    gen_test_fn!(
        test_ari_s32,
        i32,
        i32,
        int32x2_t,
        int32x2_t,
        u64,
        V_i32!(),
        fill_s32,
        fill_s32,
        to64!(int32x2_t)
    );
    gen_test_fn!(
        test_bit_s32,
        i32,
        i32,
        int32x2_t,
        int32x2_t,
        u64,
        V_i32!(),
        fill_s32,
        fill_s32,
        to64!(int32x2_t)
    );
    gen_test_fn!(
        test_cmp_s32,
        i32,
        u32,
        int32x2_t,
        uint32x2_t,
        u64,
        V_i32!(),
        fill_s32,
        fill_u32,
        to64!(uint32x2_t)
    );
    gen_test_fn!(
        testq_ari_s32,
        i32,
        i32,
        int32x4_t,
        int32x4_t,
        u128,
        V_i32!(),
        fillq_s32,
        fillq_s32,
        to128!(int32x4_t)
    );
    gen_test_fn!(
        testq_bit_s32,
        i32,
        i32,
        int32x4_t,
        int32x4_t,
        u128,
        V_i32!(),
        fillq_s32,
        fillq_s32,
        to128!(int32x4_t)
    );
    gen_test_fn!(
        testq_cmp_s32,
        i32,
        u32,
        int32x4_t,
        uint32x4_t,
        u128,
        V_i32!(),
        fillq_s32,
        fillq_u32,
        to128!(uint32x4_t)
    );

    gen_test_fn!(
        test_ari_u64,
        u64,
        u64,
        uint64x1_t,
        uint64x1_t,
        u64,
        V_u64!(),
        fill_u64,
        fill_u64,
        to64!(uint64x1_t)
    );
    gen_test_fn!(
        test_bit_u64,
        u64,
        u64,
        uint64x1_t,
        uint64x1_t,
        u64,
        V_u64!(),
        fill_u64,
        fill_u64,
        to64!(uint64x1_t)
    );
    gen_test_fn!(
        test_cmp_u64,
        u64,
        u64,
        uint64x1_t,
        uint64x1_t,
        u64,
        V_u64!(),
        fill_u64,
        fill_u64,
        to64!(uint64x1_t)
    );
    gen_test_fn!(
        testq_ari_u64,
        u64,
        u64,
        uint64x2_t,
        uint64x2_t,
        u128,
        V_u64!(),
        fillq_u64,
        fillq_u64,
        to128!(uint64x2_t)
    );
    gen_test_fn!(
        testq_bit_u64,
        u64,
        u64,
        uint64x2_t,
        uint64x2_t,
        u128,
        V_u64!(),
        fillq_u64,
        fillq_u64,
        to128!(uint64x2_t)
    );
    gen_test_fn!(
        testq_cmp_u64,
        u64,
        u64,
        uint64x2_t,
        uint64x2_t,
        u128,
        V_u64!(),
        fillq_u64,
        fillq_u64,
        to128!(uint64x2_t)
    );

    gen_test_fn!(
        test_ari_s64,
        i64,
        i64,
        int64x1_t,
        int64x1_t,
        u64,
        V_i64!(),
        fill_s64,
        fill_s64,
        to64!(int64x1_t)
    );
    gen_test_fn!(
        test_bit_s64,
        i64,
        i64,
        int64x1_t,
        int64x1_t,
        u64,
        V_i64!(),
        fill_s64,
        fill_s64,
        to64!(int64x1_t)
    );
    gen_test_fn!(
        test_cmp_s64,
        i64,
        u64,
        int64x1_t,
        uint64x1_t,
        u64,
        V_i64!(),
        fill_s64,
        fill_u64,
        to64!(uint64x1_t)
    );
    gen_test_fn!(
        testq_ari_s64,
        i64,
        i64,
        int64x2_t,
        int64x2_t,
        u128,
        V_i64!(),
        fillq_s64,
        fillq_s64,
        to128!(int64x2_t)
    );
    gen_test_fn!(
        testq_bit_s64,
        i64,
        i64,
        int64x2_t,
        int64x2_t,
        u128,
        V_i64!(),
        fillq_s64,
        fillq_s64,
        to128!(int64x2_t)
    );
    gen_test_fn!(
        testq_cmp_s64,
        i64,
        u64,
        int64x2_t,
        uint64x2_t,
        u128,
        V_i64!(),
        fillq_s64,
        fillq_u64,
        to128!(uint64x2_t)
    );

    gen_test_fn!(
        test_ari_f32,
        f32,
        f32,
        float32x2_t,
        float32x2_t,
        u64,
        V_f32!(),
        fill_f32,
        fill_f32,
        to64!(float32x2_t)
    );
    gen_test_fn!(
        test_cmp_f32,
        f32,
        u32,
        float32x2_t,
        uint32x2_t,
        u64,
        V_f32!(),
        fill_f32,
        fill_u32,
        to64!(uint32x2_t)
    );
    gen_test_fn!(
        testq_ari_f32,
        f32,
        f32,
        float32x4_t,
        float32x4_t,
        u128,
        V_f32!(),
        fillq_f32,
        fillq_f32,
        to128!(float32x4_t)
    );
    gen_test_fn!(
        testq_cmp_f32,
        f32,
        u32,
        float32x4_t,
        uint32x4_t,
        u128,
        V_f32!(),
        fillq_f32,
        fillq_u32,
        to128!(uint32x4_t)
    );

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s8() {
        test_bit_s8(|i, j| vand_s8(i, j), |a: i8, b: i8| -> i8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s8() {
        testq_bit_s8(|i, j| vandq_s8(i, j), |a: i8, b: i8| -> i8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s16() {
        test_bit_s16(|i, j| vand_s16(i, j), |a: i16, b: i16| -> i16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s16() {
        testq_bit_s16(|i, j| vandq_s16(i, j), |a: i16, b: i16| -> i16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s32() {
        test_bit_s32(|i, j| vand_s32(i, j), |a: i32, b: i32| -> i32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s32() {
        testq_bit_s32(|i, j| vandq_s32(i, j), |a: i32, b: i32| -> i32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s64() {
        test_bit_s64(|i, j| vand_s64(i, j), |a: i64, b: i64| -> i64 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s64() {
        testq_bit_s64(|i, j| vandq_s64(i, j), |a: i64, b: i64| -> i64 { a & b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u8() {
        test_bit_u8(|i, j| vand_u8(i, j), |a: u8, b: u8| -> u8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u8() {
        testq_bit_u8(|i, j| vandq_u8(i, j), |a: u8, b: u8| -> u8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u16() {
        test_bit_u16(|i, j| vand_u16(i, j), |a: u16, b: u16| -> u16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u16() {
        testq_bit_u16(|i, j| vandq_u16(i, j), |a: u16, b: u16| -> u16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u32() {
        test_bit_u32(|i, j| vand_u32(i, j), |a: u32, b: u32| -> u32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u32() {
        testq_bit_u32(|i, j| vandq_u32(i, j), |a: u32, b: u32| -> u32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u64() {
        test_bit_u64(|i, j| vand_u64(i, j), |a: u64, b: u64| -> u64 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u64() {
        testq_bit_u64(|i, j| vandq_u64(i, j), |a: u64, b: u64| -> u64 { a & b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s8() {
        test_bit_s8(|i, j| vorr_s8(i, j), |a: i8, b: i8| -> i8 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s8() {
        testq_bit_s8(|i, j| vorrq_s8(i, j), |a: i8, b: i8| -> i8 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s16() {
        test_bit_s16(|i, j| vorr_s16(i, j), |a: i16, b: i16| -> i16 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s16() {
        testq_bit_s16(|i, j| vorrq_s16(i, j), |a: i16, b: i16| -> i16 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s32() {
        test_bit_s32(|i, j| vorr_s32(i, j), |a: i32, b: i32| -> i32 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s32() {
        testq_bit_s32(|i, j| vorrq_s32(i, j), |a: i32, b: i32| -> i32 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s64() {
        test_bit_s64(|i, j| vorr_s64(i, j), |a: i64, b: i64| -> i64 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s64() {
        testq_bit_s64(|i, j| vorrq_s64(i, j), |a: i64, b: i64| -> i64 { a | b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u8() {
        test_bit_u8(|i, j| vorr_u8(i, j), |a: u8, b: u8| -> u8 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u8() {
        testq_bit_u8(|i, j| vorrq_u8(i, j), |a: u8, b: u8| -> u8 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u16() {
        test_bit_u16(|i, j| vorr_u16(i, j), |a: u16, b: u16| -> u16 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u16() {
        testq_bit_u16(|i, j| vorrq_u16(i, j), |a: u16, b: u16| -> u16 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u32() {
        test_bit_u32(|i, j| vorr_u32(i, j), |a: u32, b: u32| -> u32 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u32() {
        testq_bit_u32(|i, j| vorrq_u32(i, j), |a: u32, b: u32| -> u32 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u64() {
        test_bit_u64(|i, j| vorr_u64(i, j), |a: u64, b: u64| -> u64 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u64() {
        testq_bit_u64(|i, j| vorrq_u64(i, j), |a: u64, b: u64| -> u64 { a | b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s8() {
        test_bit_s8(|i, j| veor_s8(i, j), |a: i8, b: i8| -> i8 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s8() {
        testq_bit_s8(|i, j| veorq_s8(i, j), |a: i8, b: i8| -> i8 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s16() {
        test_bit_s16(|i, j| veor_s16(i, j), |a: i16, b: i16| -> i16 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s16() {
        testq_bit_s16(|i, j| veorq_s16(i, j), |a: i16, b: i16| -> i16 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s32() {
        test_bit_s32(|i, j| veor_s32(i, j), |a: i32, b: i32| -> i32 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s32() {
        testq_bit_s32(|i, j| veorq_s32(i, j), |a: i32, b: i32| -> i32 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s64() {
        test_bit_s64(|i, j| veor_s64(i, j), |a: i64, b: i64| -> i64 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s64() {
        testq_bit_s64(|i, j| veorq_s64(i, j), |a: i64, b: i64| -> i64 { a ^ b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u8() {
        test_bit_u8(|i, j| veor_u8(i, j), |a: u8, b: u8| -> u8 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u8() {
        testq_bit_u8(|i, j| veorq_u8(i, j), |a: u8, b: u8| -> u8 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u16() {
        test_bit_u16(|i, j| veor_u16(i, j), |a: u16, b: u16| -> u16 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u16() {
        testq_bit_u16(|i, j| veorq_u16(i, j), |a: u16, b: u16| -> u16 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u32() {
        test_bit_u32(|i, j| veor_u32(i, j), |a: u32, b: u32| -> u32 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u32() {
        testq_bit_u32(|i, j| veorq_u32(i, j), |a: u32, b: u32| -> u32 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u64() {
        test_bit_u64(|i, j| veor_u64(i, j), |a: u64, b: u64| -> u64 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u64() {
        testq_bit_u64(|i, j| veorq_u64(i, j), |a: u64, b: u64| -> u64 { a ^ b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s8() {
        test_cmp_s8(
            |i, j| vceq_s8(i, j),
            |a: i8, b: i8| -> u8 {
                if a == b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s8() {
        testq_cmp_s8(
            |i, j| vceqq_s8(i, j),
            |a: i8, b: i8| -> u8 {
                if a == b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s16() {
        test_cmp_s16(
            |i, j| vceq_s16(i, j),
            |a: i16, b: i16| -> u16 {
                if a == b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s16() {
        testq_cmp_s16(
            |i, j| vceqq_s16(i, j),
            |a: i16, b: i16| -> u16 {
                if a == b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s32() {
        test_cmp_s32(
            |i, j| vceq_s32(i, j),
            |a: i32, b: i32| -> u32 {
                if a == b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s32() {
        testq_cmp_s32(
            |i, j| vceqq_s32(i, j),
            |a: i32, b: i32| -> u32 {
                if a == b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u8() {
        test_cmp_u8(
            |i, j| vceq_u8(i, j),
            |a: u8, b: u8| -> u8 {
                if a == b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u8() {
        testq_cmp_u8(
            |i, j| vceqq_u8(i, j),
            |a: u8, b: u8| -> u8 {
                if a == b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u16() {
        test_cmp_u16(
            |i, j| vceq_u16(i, j),
            |a: u16, b: u16| -> u16 {
                if a == b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u16() {
        testq_cmp_u16(
            |i, j| vceqq_u16(i, j),
            |a: u16, b: u16| -> u16 {
                if a == b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u32() {
        test_cmp_u32(
            |i, j| vceq_u32(i, j),
            |a: u32, b: u32| -> u32 {
                if a == b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u32() {
        testq_cmp_u32(
            |i, j| vceqq_u32(i, j),
            |a: u32, b: u32| -> u32 {
                if a == b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_f32() {
        test_cmp_f32(
            |i, j| vcge_f32(i, j),
            |a: f32, b: f32| -> u32 {
                if a == b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_f32() {
        testq_cmp_f32(
            |i, j| vcgeq_f32(i, j),
            |a: f32, b: f32| -> u32 {
                if a == b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s8() {
        test_cmp_s8(
            |i, j| vcgt_s8(i, j),
            |a: i8, b: i8| -> u8 {
                if a > b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s8() {
        testq_cmp_s8(
            |i, j| vcgtq_s8(i, j),
            |a: i8, b: i8| -> u8 {
                if a > b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s16() {
        test_cmp_s16(
            |i, j| vcgt_s16(i, j),
            |a: i16, b: i16| -> u16 {
                if a > b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s16() {
        testq_cmp_s16(
            |i, j| vcgtq_s16(i, j),
            |a: i16, b: i16| -> u16 {
                if a > b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s32() {
        test_cmp_s32(
            |i, j| vcgt_s32(i, j),
            |a: i32, b: i32| -> u32 {
                if a > b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s32() {
        testq_cmp_s32(
            |i, j| vcgtq_s32(i, j),
            |a: i32, b: i32| -> u32 {
                if a > b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u8() {
        test_cmp_u8(
            |i, j| vcgt_u8(i, j),
            |a: u8, b: u8| -> u8 {
                if a > b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u8() {
        testq_cmp_u8(
            |i, j| vcgtq_u8(i, j),
            |a: u8, b: u8| -> u8 {
                if a > b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u16() {
        test_cmp_u16(
            |i, j| vcgt_u16(i, j),
            |a: u16, b: u16| -> u16 {
                if a > b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u16() {
        testq_cmp_u16(
            |i, j| vcgtq_u16(i, j),
            |a: u16, b: u16| -> u16 {
                if a > b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u32() {
        test_cmp_u32(
            |i, j| vcgt_u32(i, j),
            |a: u32, b: u32| -> u32 {
                if a > b {
                    0xFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u32() {
        testq_cmp_u32(
            |i, j| vcgtq_u32(i, j),
            |a: u32, b: u32| -> u32 {
                if a > b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_f32() {
        test_cmp_f32(
            |i, j| vcgt_f32(i, j),
            |a: f32, b: f32| -> u32 {
                if a > b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_f32() {
        testq_cmp_f32(
            |i, j| vcgtq_f32(i, j),
            |a: f32, b: f32| -> u32 {
                if a > b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s8() {
        test_cmp_s8(
            |i, j| vclt_s8(i, j),
            |a: i8, b: i8| -> u8 {
                if a < b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s8() {
        testq_cmp_s8(
            |i, j| vcltq_s8(i, j),
            |a: i8, b: i8| -> u8 {
                if a < b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s16() {
        test_cmp_s16(
            |i, j| vclt_s16(i, j),
            |a: i16, b: i16| -> u16 {
                if a < b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s16() {
        testq_cmp_s16(
            |i, j| vcltq_s16(i, j),
            |a: i16, b: i16| -> u16 {
                if a < b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s32() {
        test_cmp_s32(
            |i, j| vclt_s32(i, j),
            |a: i32, b: i32| -> u32 {
                if a < b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s32() {
        testq_cmp_s32(
            |i, j| vcltq_s32(i, j),
            |a: i32, b: i32| -> u32 {
                if a < b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u8() {
        test_cmp_u8(
            |i, j| vclt_u8(i, j),
            |a: u8, b: u8| -> u8 {
                if a < b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u8() {
        testq_cmp_u8(
            |i, j| vcltq_u8(i, j),
            |a: u8, b: u8| -> u8 {
                if a < b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u16() {
        test_cmp_u16(
            |i, j| vclt_u16(i, j),
            |a: u16, b: u16| -> u16 {
                if a < b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u16() {
        testq_cmp_u16(
            |i, j| vcltq_u16(i, j),
            |a: u16, b: u16| -> u16 {
                if a < b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u32() {
        test_cmp_u32(
            |i, j| vclt_u32(i, j),
            |a: u32, b: u32| -> u32 {
                if a < b {
                    0xFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u32() {
        testq_cmp_u32(
            |i, j| vcltq_u32(i, j),
            |a: u32, b: u32| -> u32 {
                if a < b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_f32() {
        test_cmp_f32(
            |i, j| vclt_f32(i, j),
            |a: f32, b: f32| -> u32 {
                if a < b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_f32() {
        testq_cmp_f32(
            |i, j| vcltq_f32(i, j),
            |a: f32, b: f32| -> u32 {
                if a < b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s8() {
        test_cmp_s8(
            |i, j| vcle_s8(i, j),
            |a: i8, b: i8| -> u8 {
                if a <= b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s8() {
        testq_cmp_s8(
            |i, j| vcleq_s8(i, j),
            |a: i8, b: i8| -> u8 {
                if a <= b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s16() {
        test_cmp_s16(
            |i, j| vcle_s16(i, j),
            |a: i16, b: i16| -> u16 {
                if a <= b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s16() {
        testq_cmp_s16(
            |i, j| vcleq_s16(i, j),
            |a: i16, b: i16| -> u16 {
                if a <= b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s32() {
        test_cmp_s32(
            |i, j| vcle_s32(i, j),
            |a: i32, b: i32| -> u32 {
                if a <= b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s32() {
        testq_cmp_s32(
            |i, j| vcleq_s32(i, j),
            |a: i32, b: i32| -> u32 {
                if a <= b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u8() {
        test_cmp_u8(
            |i, j| vcle_u8(i, j),
            |a: u8, b: u8| -> u8 {
                if a <= b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u8() {
        testq_cmp_u8(
            |i, j| vcleq_u8(i, j),
            |a: u8, b: u8| -> u8 {
                if a <= b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u16() {
        test_cmp_u16(
            |i, j| vcle_u16(i, j),
            |a: u16, b: u16| -> u16 {
                if a <= b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u16() {
        testq_cmp_u16(
            |i, j| vcleq_u16(i, j),
            |a: u16, b: u16| -> u16 {
                if a <= b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u32() {
        test_cmp_u32(
            |i, j| vcle_u32(i, j),
            |a: u32, b: u32| -> u32 {
                if a <= b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u32() {
        testq_cmp_u32(
            |i, j| vcleq_u32(i, j),
            |a: u32, b: u32| -> u32 {
                if a <= b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_f32() {
        test_cmp_f32(
            |i, j| vcle_f32(i, j),
            |a: f32, b: f32| -> u32 {
                if a <= b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_f32() {
        testq_cmp_f32(
            |i, j| vcleq_f32(i, j),
            |a: f32, b: f32| -> u32 {
                if a <= b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s8() {
        test_cmp_s8(
            |i, j| vcge_s8(i, j),
            |a: i8, b: i8| -> u8 {
                if a >= b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s8() {
        testq_cmp_s8(
            |i, j| vcgeq_s8(i, j),
            |a: i8, b: i8| -> u8 {
                if a >= b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s16() {
        test_cmp_s16(
            |i, j| vcge_s16(i, j),
            |a: i16, b: i16| -> u16 {
                if a >= b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s16() {
        testq_cmp_s16(
            |i, j| vcgeq_s16(i, j),
            |a: i16, b: i16| -> u16 {
                if a >= b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s32() {
        test_cmp_s32(
            |i, j| vcge_s32(i, j),
            |a: i32, b: i32| -> u32 {
                if a >= b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s32() {
        testq_cmp_s32(
            |i, j| vcgeq_s32(i, j),
            |a: i32, b: i32| -> u32 {
                if a >= b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u8() {
        test_cmp_u8(
            |i, j| vcge_u8(i, j),
            |a: u8, b: u8| -> u8 {
                if a >= b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u8() {
        testq_cmp_u8(
            |i, j| vcgeq_u8(i, j),
            |a: u8, b: u8| -> u8 {
                if a >= b {
                    0xFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u16() {
        test_cmp_u16(
            |i, j| vcge_u16(i, j),
            |a: u16, b: u16| -> u16 {
                if a >= b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u16() {
        testq_cmp_u16(
            |i, j| vcgeq_u16(i, j),
            |a: u16, b: u16| -> u16 {
                if a >= b {
                    0xFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u32() {
        test_cmp_u32(
            |i, j| vcge_u32(i, j),
            |a: u32, b: u32| -> u32 {
                if a >= b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u32() {
        testq_cmp_u32(
            |i, j| vcgeq_u32(i, j),
            |a: u32, b: u32| -> u32 {
                if a >= b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_f32() {
        test_cmp_f32(
            |i, j| vcge_f32(i, j),
            |a: f32, b: f32| -> u32 {
                if a >= b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_f32() {
        testq_cmp_f32(
            |i, j| vcgeq_f32(i, j),
            |a: f32, b: f32| -> u32 {
                if a >= b {
                    0xFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s8() {
        test_ari_s8(
            |i, j| vqsub_s8(i, j),
            |a: i8, b: i8| -> i8 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s8() {
        testq_ari_s8(
            |i, j| vqsubq_s8(i, j),
            |a: i8, b: i8| -> i8 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s16() {
        test_ari_s16(
            |i, j| vqsub_s16(i, j),
            |a: i16, b: i16| -> i16 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s16() {
        testq_ari_s16(
            |i, j| vqsubq_s16(i, j),
            |a: i16, b: i16| -> i16 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s32() {
        test_ari_s32(
            |i, j| vqsub_s32(i, j),
            |a: i32, b: i32| -> i32 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s32() {
        testq_ari_s32(
            |i, j| vqsubq_s32(i, j),
            |a: i32, b: i32| -> i32 { a.saturating_sub(b) },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u8() {
        test_ari_u8(
            |i, j| vqsub_u8(i, j),
            |a: u8, b: u8| -> u8 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u8() {
        testq_ari_u8(
            |i, j| vqsubq_u8(i, j),
            |a: u8, b: u8| -> u8 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u16() {
        test_ari_u16(
            |i, j| vqsub_u16(i, j),
            |a: u16, b: u16| -> u16 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u16() {
        testq_ari_u16(
            |i, j| vqsubq_u16(i, j),
            |a: u16, b: u16| -> u16 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u32() {
        test_ari_u32(
            |i, j| vqsub_u32(i, j),
            |a: u32, b: u32| -> u32 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u32() {
        testq_ari_u32(
            |i, j| vqsubq_u32(i, j),
            |a: u32, b: u32| -> u32 { a.saturating_sub(b) },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s8() {
        test_ari_s8(|i, j| vhadd_s8(i, j), |a: i8, b: i8| -> i8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s8() {
        testq_ari_s8(|i, j| vhaddq_s8(i, j), |a: i8, b: i8| -> i8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s16() {
        test_ari_s16(|i, j| vhadd_s16(i, j), |a: i16, b: i16| -> i16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s16() {
        testq_ari_s16(|i, j| vhaddq_s16(i, j), |a: i16, b: i16| -> i16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s32() {
        test_ari_s32(|i, j| vhadd_s32(i, j), |a: i32, b: i32| -> i32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s32() {
        testq_ari_s32(|i, j| vhaddq_s32(i, j), |a: i32, b: i32| -> i32 { a & b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u8() {
        test_ari_u8(|i, j| vhadd_u8(i, j), |a: u8, b: u8| -> u8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u8() {
        testq_ari_u8(|i, j| vhaddq_u8(i, j), |a: u8, b: u8| -> u8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u16() {
        test_ari_u16(|i, j| vhadd_u16(i, j), |a: u16, b: u16| -> u16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u16() {
        testq_ari_u16(|i, j| vhaddq_u16(i, j), |a: u16, b: u16| -> u16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u32() {
        test_ari_u32(|i, j| vhadd_u32(i, j), |a: u32, b: u32| -> u32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u32() {
        testq_ari_u32(|i, j| vhaddq_u32(i, j), |a: u32, b: u32| -> u32 { a & b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_s8() {
        test_ari_s8(|i, j| vrhadd_s8(i, j), |a: i8, b: i8| -> i8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_s8() {
        testq_ari_s8(|i, j| vrhaddq_s8(i, j), |a: i8, b: i8| -> i8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_s16() {
        test_ari_s16(|i, j| vrhadd_s16(i, j), |a: i16, b: i16| -> i16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_s16() {
        testq_ari_s16(|i, j| vrhaddq_s16(i, j), |a: i16, b: i16| -> i16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_s32() {
        test_ari_s32(|i, j| vrhadd_s32(i, j), |a: i32, b: i32| -> i32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_s32() {
        testq_ari_s32(|i, j| vrhaddq_s32(i, j), |a: i32, b: i32| -> i32 { a & b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_u8() {
        test_ari_u8(|i, j| vrhadd_u8(i, j), |a: u8, b: u8| -> u8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_u8() {
        testq_ari_u8(|i, j| vrhaddq_u8(i, j), |a: u8, b: u8| -> u8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_u16() {
        test_ari_u16(|i, j| vrhadd_u16(i, j), |a: u16, b: u16| -> u16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_u16() {
        testq_ari_u16(|i, j| vrhaddq_u16(i, j), |a: u16, b: u16| -> u16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_u32() {
        test_ari_u32(|i, j| vrhadd_u32(i, j), |a: u32, b: u32| -> u32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_u32() {
        testq_ari_u32(|i, j| vrhaddq_u32(i, j), |a: u32, b: u32| -> u32 { a & b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s8() {
        test_ari_s8(
            |i, j| vqadd_s8(i, j),
            |a: i8, b: i8| -> i8 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s8() {
        testq_ari_s8(
            |i, j| vqaddq_s8(i, j),
            |a: i8, b: i8| -> i8 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s16() {
        test_ari_s16(
            |i, j| vqadd_s16(i, j),
            |a: i16, b: i16| -> i16 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s16() {
        testq_ari_s16(
            |i, j| vqaddq_s16(i, j),
            |a: i16, b: i16| -> i16 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s32() {
        test_ari_s32(
            |i, j| vqadd_s32(i, j),
            |a: i32, b: i32| -> i32 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s32() {
        testq_ari_s32(
            |i, j| vqaddq_s32(i, j),
            |a: i32, b: i32| -> i32 { a.saturating_add(b) },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u8() {
        test_ari_u8(
            |i, j| vqadd_u8(i, j),
            |a: u8, b: u8| -> u8 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u8() {
        testq_ari_u8(
            |i, j| vqaddq_u8(i, j),
            |a: u8, b: u8| -> u8 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u16() {
        test_ari_u16(
            |i, j| vqadd_u16(i, j),
            |a: u16, b: u16| -> u16 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u16() {
        testq_ari_u16(
            |i, j| vqaddq_u16(i, j),
            |a: u16, b: u16| -> u16 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u32() {
        test_ari_u32(
            |i, j| vqadd_u32(i, j),
            |a: u32, b: u32| -> u32 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u32() {
        testq_ari_u32(
            |i, j| vqaddq_u32(i, j),
            |a: u32, b: u32| -> u32 { a.saturating_add(b) },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_s8() {
        test_ari_s8(
            |i, j| vmul_s8(i, j),
            |a: i8, b: i8| -> i8 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_s8() {
        testq_ari_s8(
            |i, j| vmulq_s8(i, j),
            |a: i8, b: i8| -> i8 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_s16() {
        test_ari_s16(
            |i, j| vmul_s16(i, j),
            |a: i16, b: i16| -> i16 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_s16() {
        testq_ari_s16(
            |i, j| vmulq_s16(i, j),
            |a: i16, b: i16| -> i16 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_s32() {
        test_ari_s32(
            |i, j| vmul_s32(i, j),
            |a: i32, b: i32| -> i32 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_s32() {
        testq_ari_s32(
            |i, j| vmulq_s32(i, j),
            |a: i32, b: i32| -> i32 { a.overflowing_mul(b).0 },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_u8() {
        test_ari_u8(
            |i, j| vmul_u8(i, j),
            |a: u8, b: u8| -> u8 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_u8() {
        testq_ari_u8(
            |i, j| vmulq_u8(i, j),
            |a: u8, b: u8| -> u8 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_u16() {
        test_ari_u16(
            |i, j| vmul_u16(i, j),
            |a: u16, b: u16| -> u16 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_u16() {
        testq_ari_u16(
            |i, j| vmulq_u16(i, j),
            |a: u16, b: u16| -> u16 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_u32() {
        test_ari_u32(
            |i, j| vmul_u32(i, j),
            |a: u32, b: u32| -> u32 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_u32() {
        testq_ari_u32(
            |i, j| vmulq_u32(i, j),
            |a: u32, b: u32| -> u32 { a.overflowing_mul(b).0 },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_f32() {
        test_ari_f32(|i, j| vmul_f32(i, j), |a: f32, b: f32| -> f32 { a * b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_f32() {
        testq_ari_f32(|i, j| vmulq_f32(i, j), |a: f32, b: f32| -> f32 { a * b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s8() {
        test_ari_s8(|i, j| vsub_s8(i, j), |a: i8, b: i8| -> i8 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s8() {
        testq_ari_s8(|i, j| vsubq_s8(i, j), |a: i8, b: i8| -> i8 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s16() {
        test_ari_s16(|i, j| vsub_s16(i, j), |a: i16, b: i16| -> i16 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s16() {
        testq_ari_s16(|i, j| vsubq_s16(i, j), |a: i16, b: i16| -> i16 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s32() {
        test_ari_s32(|i, j| vsub_s32(i, j), |a: i32, b: i32| -> i32 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s32() {
        testq_ari_s32(|i, j| vsubq_s32(i, j), |a: i32, b: i32| -> i32 { a - b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u8() {
        test_ari_u8(|i, j| vsub_u8(i, j), |a: u8, b: u8| -> u8 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u8() {
        testq_ari_u8(|i, j| vsubq_u8(i, j), |a: u8, b: u8| -> u8 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u16() {
        test_ari_u16(|i, j| vsub_u16(i, j), |a: u16, b: u16| -> u16 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u16() {
        testq_ari_u16(|i, j| vsubq_u16(i, j), |a: u16, b: u16| -> u16 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u32() {
        test_ari_u32(|i, j| vsub_u32(i, j), |a: u32, b: u32| -> u32 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u32() {
        testq_ari_u32(|i, j| vsubq_u32(i, j), |a: u32, b: u32| -> u32 { a - b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_f32() {
        test_ari_f32(|i, j| vsub_f32(i, j), |a: f32, b: f32| -> f32 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_f32() {
        testq_ari_f32(|i, j| vsubq_f32(i, j), |a: f32, b: f32| -> f32 { a - b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_s8() {
        test_ari_s8(
            |i, j| vhsub_s8(i, j),
            |a: i8, b: i8| -> i8 { (((a as i16) - (b as i16)) / 2) as i8 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_s8() {
        testq_ari_s8(
            |i, j| vhsubq_s8(i, j),
            |a: i8, b: i8| -> i8 { (((a as i16) - (b as i16)) / 2) as i8 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_s16() {
        test_ari_s16(
            |i, j| vhsub_s16(i, j),
            |a: i16, b: i16| -> i16 { (((a as i32) - (b as i32)) / 2) as i16 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_s16() {
        testq_ari_s16(
            |i, j| vhsubq_s16(i, j),
            |a: i16, b: i16| -> i16 { (((a as i32) - (b as i32)) / 2) as i16 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_s32() {
        test_ari_s32(
            |i, j| vhsub_s32(i, j),
            |a: i32, b: i32| -> i32 { (((a as i64) - (b as i64)) / 2) as i32 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_s32() {
        testq_ari_s32(
            |i, j| vhsubq_s32(i, j),
            |a: i32, b: i32| -> i32 { (((a as i64) - (b as i64)) / 2) as i32 },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_u8() {
        test_ari_u8(
            |i, j| vhsub_u8(i, j),
            |a: u8, b: u8| -> u8 { (((a as u16) - (b as u16)) / 2) as u8 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_u8() {
        testq_ari_u8(
            |i, j| vhsubq_u8(i, j),
            |a: u8, b: u8| -> u8 { (((a as u16) - (b as u16)) / 2) as u8 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_u16() {
        test_ari_u16(
            |i, j| vhsub_u16(i, j),
            |a: u16, b: u16| -> u16 { (((a as u16) - (b as u16)) / 2) as u16 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_u16() {
        testq_ari_u16(
            |i, j| vhsubq_u16(i, j),
            |a: u16, b: u16| -> u16 { (((a as u16) - (b as u16)) / 2) as u16 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_u32() {
        test_ari_u32(
            |i, j| vhsub_u32(i, j),
            |a: u32, b: u32| -> u32 { (((a as u64) - (b as u64)) / 2) as u32 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_u32() {
        testq_ari_u32(
            |i, j| vhsubq_u32(i, j),
            |a: u32, b: u32| -> u32 { (((a as u64) - (b as u64)) / 2) as u32 },
        );
    }
}

#[cfg(test)]
#[cfg(target_endian = "little")]
mod table_lookup_tests;
