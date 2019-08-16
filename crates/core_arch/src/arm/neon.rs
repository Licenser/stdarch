//! ARMv7 NEON intrinsics

use crate::{core_arch::simd_llvm::*, hint::unreachable_unchecked, mem::transmute};
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

    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v16i8")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubu.v16i8")]
    fn vqsubq_u8_(a: uint8x16_t, a: uint8x16_t) -> uint8x16_t;

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

//uint8x16_t vqsubq_u8 (uint8x16_t a, uint8x16_t b)
///Unsigned saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(uqsub))]
pub unsafe fn vqsubq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vqsubq_u8_(a, b)
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

macro_rules! arm_simd_2 {
    ($name:ident, $type:ty, $simd_fn:ident, $intrarm:ident, $intraarch:ident) => {
        arm_simd_2!($name, $type, $type, $simd_fn, $intrarm, $intraarch);
    };
    ($name:ident, $type:ty, $res:ty, $simd_fn:ident, $intrarm:ident, $intraarch:ident) => {
        #[inline]
        #[target_feature(enable = "neon")]
        #[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
        #[cfg_attr(all(test, target_arch = "arm"), assert_instr($intrarm))]
        #[cfg_attr(all(test, target_arch = "aarch64"), assert_instr($intraarch))]
        pub unsafe fn $name(a: $type, b: $type) -> $res {
            $simd_fn(a, b)
        }
    };
}

macro_rules! arm_simd_and {
    ($name:ident, $type:ty) => {
        /// Vector bitwise and.
        arm_simd_2!($name, $type, simd_and, vand, and);
    };
}
arm_simd_and!(vand_s8, int8x8_t);
arm_simd_and!(vandq_s8, int8x16_t);
arm_simd_and!(vand_s16, int16x4_t);
arm_simd_and!(vandq_s16, int16x8_t);
arm_simd_and!(vand_s32, int32x2_t);
arm_simd_and!(vandq_s32, int32x4_t);
arm_simd_and!(vand_s64, int64x1_t);
arm_simd_and!(vandq_s64, int64x2_t);
arm_simd_and!(vand_u8, uint8x8_t);
arm_simd_and!(vandq_u8, uint8x16_t);
arm_simd_and!(vand_u16, uint16x4_t);
arm_simd_and!(vandq_u16, uint16x8_t);
arm_simd_and!(vand_u32, uint32x2_t);
arm_simd_and!(vandq_u32, uint32x4_t);
arm_simd_and!(vand_u64, uint64x1_t);
arm_simd_and!(vandq_u64, uint64x2_t);

macro_rules! arm_simd_orr {
    ($name:ident, $type:ty) => {
        /// Vector bitwise or (immediate, inclusive).
        arm_simd_2!($name, $type, simd_or, vorr, orr);
    };
}

arm_simd_orr!(vorr_s8, int8x8_t);
arm_simd_orr!(vorrq_s8, int8x16_t);
arm_simd_orr!(vorr_s16, int16x4_t);
arm_simd_orr!(vorrq_s16, int16x8_t);
arm_simd_orr!(vorr_s32, int32x2_t);
arm_simd_orr!(vorrq_s32, int32x4_t);
arm_simd_orr!(vorr_s64, int64x1_t);
arm_simd_orr!(vorrq_s64, int64x2_t);
arm_simd_orr!(vorr_u8, uint8x8_t);
arm_simd_orr!(vorrq_u8, uint8x16_t);
arm_simd_orr!(vorr_u16, uint16x4_t);
arm_simd_orr!(vorrq_u16, uint16x8_t);
arm_simd_orr!(vorr_u32, uint32x2_t);
arm_simd_orr!(vorrq_u32, uint32x4_t);
arm_simd_orr!(vorr_u64, uint64x1_t);
arm_simd_orr!(vorrq_u64, uint64x2_t);

macro_rules! arm_simd_eor {
    ($name:ident, $type:ty) => {
        /// Vector bitwise exclusive or (vector).
        arm_simd_2!($name, $type, simd_xor, veor, eor);
    };
}

arm_simd_eor!(veor_s8, int8x8_t);
arm_simd_eor!(veorq_s8, int8x16_t);
arm_simd_eor!(veor_s16, int16x4_t);
arm_simd_eor!(veorq_s16, int16x8_t);
arm_simd_eor!(veor_s32, int32x2_t);
arm_simd_eor!(veorq_s32, int32x4_t);
arm_simd_eor!(veor_s64, int64x1_t);
arm_simd_eor!(veorq_s64, int64x2_t);
arm_simd_eor!(veor_u8, uint8x8_t);
arm_simd_eor!(veorq_u8, uint8x16_t);
arm_simd_eor!(veor_u16, uint16x4_t);
arm_simd_eor!(veorq_u16, uint16x8_t);
arm_simd_eor!(veor_u32, uint32x2_t);
arm_simd_eor!(veorq_u32, uint32x4_t);
arm_simd_eor!(veor_u64, uint64x1_t);
arm_simd_eor!(veorq_u64, uint64x2_t);

macro_rules! arm_simd_ceq {
    ($name:ident, $type:ty, $res:ty) => {
        /// Compare bitwise Equal (vector)
        arm_simd_2!($name, $type, $res, simd_eq, cmeq, cmeq);
    };
}

arm_simd_ceq!(vceq_s8, int8x8_t, uint8x8_t);
arm_simd_ceq!(vceqq_s8, int8x16_t, uint8x16_t);
arm_simd_ceq!(vceq_s16, int16x4_t, uint16x4_t);
arm_simd_ceq!(vceqq_s16, int16x8_t, uint16x8_t);
arm_simd_ceq!(vceq_s32, int32x2_t, uint32x2_t);
arm_simd_ceq!(vceqq_s32, int32x4_t, uint32x4_t);
arm_simd_ceq!(vceq_u8, uint8x8_t, uint8x8_t);
arm_simd_ceq!(vceqq_u8, uint8x16_t, uint8x16_t);
arm_simd_ceq!(vceq_u16, uint16x4_t, uint16x4_t);
arm_simd_ceq!(vceqq_u16, uint16x8_t, uint16x8_t);
arm_simd_ceq!(vceq_u32, uint32x2_t, uint32x2_t);
arm_simd_ceq!(vceqq_u32, uint32x4_t, uint32x4_t);
arm_simd_2!(vceq_f32, float32x2_t, uint32x2_t, simd_eq, fcmeq, fcmeq);
arm_simd_2!(vceqq_f32, float32x4_t, uint32x4_t, simd_eq, fcmeq, fcmeq);
arm_simd_ceq!(vceq_p8, poly8x8_t, poly8x8_t);
arm_simd_ceq!(vceqq_p8, poly8x16_t, poly8x16_t);

macro_rules! arm_simd_cgt {
    ($name:ident, $type:ty, $res:ty) => {
        /// Compare signed Greater than (vector)
        arm_simd_2!($name, $type, $res, simd_gt, cmgt, cmgt);
    };
}

macro_rules! arm_simd_cgtu {
    ($name:ident, $type:ty) => {
        /// Compare Greater than (vector)
        arm_simd_2!($name, $type, simd_gt, cmhi, cmhi);
    };
}
arm_simd_cgt!(vcgt_s8, int8x8_t, uint8x8_t);
arm_simd_cgt!(vcgtq_s8, int8x16_t, uint8x16_t);
arm_simd_cgt!(vcgt_s16, int16x4_t, uint16x4_t);
arm_simd_cgt!(vcgtq_s16, int16x8_t, uint16x8_t);
arm_simd_cgt!(vcgt_s32, int32x2_t, uint32x2_t);
arm_simd_cgt!(vcgtq_s32, int32x4_t, uint32x4_t);
arm_simd_cgtu!(vcgt_u8, uint8x8_t);
arm_simd_cgtu!(vcgtq_u8, uint8x16_t);
arm_simd_cgtu!(vcgt_u16, uint16x4_t);
arm_simd_cgtu!(vcgtq_u16, uint16x8_t);
arm_simd_cgtu!(vcgt_u32, uint32x2_t);
arm_simd_cgtu!(vcgtq_u32, uint32x4_t);
arm_simd_2!(vcgt_f32, float32x2_t, uint32x2_t, simd_gt, fcmgt, fcmgt);
arm_simd_2!(vcgtq_f32, float32x4_t, uint32x4_t, simd_gt, fcmgt, fcmgt);

macro_rules! arm_simd_clt {
    ($name:ident, $type:ty, $res:ty) => {
        /// Compare signed Lesser than (vector)
        arm_simd_2!($name, $type, $res, simd_lt, cmgt, cmgt);
    };
}

macro_rules! arm_simd_cltu {
    ($name:ident, $type:ty) => {
        /// Compare Lesser than (vector)
        arm_simd_2!($name, $type, simd_lt, cmhi, cmhi);
    };
}
arm_simd_clt!(vclt_s8, int8x8_t, uint8x8_t);
arm_simd_clt!(vcltq_s8, int8x16_t, uint8x16_t);
arm_simd_clt!(vclt_s16, int16x4_t, uint16x4_t);
arm_simd_clt!(vcltq_s16, int16x8_t, uint16x8_t);
arm_simd_clt!(vclt_s32, int32x2_t, uint32x2_t);
arm_simd_clt!(vcltq_s32, int32x4_t, uint32x4_t);
arm_simd_cltu!(vclt_u8, uint8x8_t);
arm_simd_cltu!(vcltq_u8, uint8x16_t);
arm_simd_cltu!(vclt_u16, uint16x4_t);
arm_simd_cltu!(vcltq_u16, uint16x8_t);
arm_simd_cltu!(vclt_u32, uint32x2_t);
arm_simd_cltu!(vcltq_u32, uint32x4_t);
arm_simd_2!(vclt_f32, float32x2_t, uint32x2_t, simd_lt, fcmgt, fcmgt);
arm_simd_2!(vcltq_f32, float32x4_t, uint32x4_t, simd_lt, fcmgt, fcmgt);

macro_rules! arm_simd_cge {
    ($name:ident, $type:ty, $res:ty) => {
        /// Compare signed Greater than equals (vector)
        arm_simd_2!($name, $type, $res, simd_ge, cmge, cmge);
    };
}

macro_rules! arm_simd_cgeu {
    ($name:ident, $type:ty) => {
        /// Compare Greater than equals (vector)
        arm_simd_2!($name, $type, simd_ge, cmhs, cmhs);
    };
}
arm_simd_cge!(vcge_s8, int8x8_t, uint8x8_t);
arm_simd_cge!(vcgeq_s8, int8x16_t, uint8x16_t);
arm_simd_cge!(vcge_s16, int16x4_t, uint16x4_t);
arm_simd_cge!(vcgeq_s16, int16x8_t, uint16x8_t);
arm_simd_cge!(vcge_s32, int32x2_t, uint32x2_t);
arm_simd_cge!(vcgeq_s32, int32x4_t, uint32x4_t);
arm_simd_cgeu!(vcge_u8, uint8x8_t);
arm_simd_cgeu!(vcgeq_u8, uint8x16_t);
arm_simd_cgeu!(vcge_u16, uint16x4_t);
arm_simd_cgeu!(vcgeq_u16, uint16x8_t);
arm_simd_cgeu!(vcge_u32, uint32x2_t);
arm_simd_cgeu!(vcgeq_u32, uint32x4_t);
arm_simd_2!(vcge_f32, float32x2_t, uint32x2_t, simd_ge, fcmge, fcmge);
arm_simd_2!(vcgeq_f32, float32x4_t, uint32x4_t, simd_ge, fcmge, fcmge);

macro_rules! arm_simd_cle {
    ($name:ident, $type:ty, $res:ty) => {
        /// Compare signed Lesser than equals (vector)
        arm_simd_2!($name, $type, $res, simd_le, cmge, cmge);
    };
}

macro_rules! arm_simd_cleu {
    ($name:ident, $type:ty) => {
        /// Compare Lesser than equals (vector)
        arm_simd_2!($name, $type, simd_le, cmhs, cmhs);
    };
}
arm_simd_cle!(vcle_s8, int8x8_t, uint8x8_t);
arm_simd_cle!(vcleq_s8, int8x16_t, uint8x16_t);
arm_simd_cle!(vcle_s16, int16x4_t, uint16x4_t);
arm_simd_cle!(vcleq_s16, int16x8_t, uint16x8_t);
arm_simd_cle!(vcle_s32, int32x2_t, uint32x2_t);
arm_simd_cle!(vcleq_s32, int32x4_t, uint32x4_t);
arm_simd_cleu!(vcle_u8, uint8x8_t);
arm_simd_cleu!(vcleq_u8, uint8x16_t);
arm_simd_cleu!(vcle_u16, uint16x4_t);
arm_simd_cleu!(vcleq_u16, uint16x8_t);
arm_simd_cleu!(vcle_u32, uint32x2_t);
arm_simd_cleu!(vcleq_u32, uint32x4_t);
arm_simd_2!(vcle_f32, float32x2_t, uint32x2_t, simd_le, fcmge, fcmge);
arm_simd_2!(vcleq_f32, float32x4_t, uint32x4_t, simd_le, fcmge, fcmge);

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

macro_rules! arm_vget_lane {
    ($name:ident, $from:ty, $to:ty, $lanes:literal) => {
        #[inline]
        #[target_feature(enable = "neon")]
        #[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
        #[cfg_attr(test, assert_instr(umov))]
        pub unsafe fn $name(v: $from, lane: i32) -> $to {
            macro_rules! call {
                ($imm5:expr) => {
                    if ($imm5) < 0 || ($imm5) > $lanes {
                        unreachable_unchecked()
                    } else {
                        simd_extract(v, $imm5)
                    }
                };
            }
            constify_imm5!(lane, call)
        }
    };
}

//  uint64_t vgetq_lane_u64 (uint64x2_t v, const int lane)
arm_vget_lane!(vgetq_lane_u64, uint64x2_t, u64, 2);

#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[rustc_args_required_const(1)]
#[cfg_attr(test, assert_instr(umov, imm5 = 0))]
pub unsafe fn vgetq_lane_u16(v: uint16x8_t, imm5: i32) -> u16 {
    if (imm5) < 0 || (imm5) > 7 {
        unreachable_unchecked()
    }
    let imm5 = (imm5 & 7) as u32;
    simd_extract(v, imm5)
}

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

// int8x16_t vdupq_n_s8 (int8_t value)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
// Those should devinetly fail!
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(dup))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup))]
pub unsafe fn vdupq_n_s8(value: i8) -> int8x16_t {
    int8x16_t(
        value, value, value, value, value, value, value, value, value, value, value, value, value,
        value, value, value,
    )
}

// uint8x16_t vdupq_n_u8 (uint8_t value)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
// Those should devinetly fail!
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(dup))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup))]
pub unsafe fn vdupq_n_u8(value: u8) -> uint8x16_t {
    uint8x16_t(
        value, value, value, value, value, value, value, value, value, value, value, value, value,
        value, value, value,
    )
}

// uint8x16_t vmovq_n_u8 (uint8_t value)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
// Those should devinetly fail!
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(dup))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup))]
pub unsafe fn vmovq_n_u8(value: u8) -> uint8x16_t {
    vdupq_n_u8(value)
}

macro_rules! arm_reinterpret {
    ($name:ident, $from:ty, $to:ty) => {
        // Vector reinterpret cast operation
        #[inline]
        #[target_feature(enable = "neon")]
        #[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
        pub unsafe fn $name(a: $from) -> $to {
            transmute(a)
        }
    };
}

// uint64x1_t vreinterpret_u64_u32 (uint32x2_t a)
arm_reinterpret!(vreinterpret_u64_u32, uint32x2_t, uint64x1_t);

// int8x16_t vreinterpretq_s8_u8 (uint8x16_t a)
arm_reinterpret!(vreinterpretq_s8_u8, uint8x16_t, int8x16_t);

// uint16x8_t vreinterpretq_u16_u8 (uint8x16_t a)
arm_reinterpret!(vreinterpretq_u16_u8, uint8x16_t, uint16x8_t);

// uint32x4_t vreinterpretq_u32_u8 (uint8x16_t a)
arm_reinterpret!(vreinterpretq_u32_u8, uint8x16_t, uint32x4_t);

// uint64x2_t vreinterpretq_u64_u8 (uint8x16_t a)
arm_reinterpret!(vreinterpretq_u64_u8, uint8x16_t, uint64x2_t);

// uint8x16_t vreinterpretq_u8_s8 (int8x16_t a)
arm_reinterpret!(vreinterpretq_u8_s8, int8x16_t, uint8x16_t);

//uint8x16_t vshrq_n_u8 (uint8x16_t a, const int n)
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

//uint8x16_t vshlq_n_u8 (uint8x16_t a, const int n)
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
//int8x16_t vextq_s8 (int8x16_t a, int8x16_t b, const int n)
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
//uint8x16_t vextq_s8 (uint8x16_t a, uint8x16_t b, const int n)
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

#[cfg(test)]
mod tests {
    use crate::core_arch::{arm::*, simd::*};
    use std::{i16, i32, i8, mem::transmute, u16, u32, u8};
    use stdarch_test::simd_test;

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_u8() {
        let v = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = vget_lane_u8(transmute(v), 1);
        assert_eq!(r, 2);
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
    unsafe fn test_vqsubq_u8() {
        let a = u8x16::new(
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        );
        let b = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e = u8x16::new(
            41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26,
        );
        let r: u8x16 = transmute(vqsubq_u8(transmute(a), transmute(b)));
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
        let lane = 1;
        let r = vgetq_lane_u64(transmute(v), lane);
        assert_eq!(r, 2);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_s8() {
        let a = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i8x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let e = i8x8::new(9, 9, 9, 9, 9, 9, 9, 9);
        let r: i8x8 = transmute(vadd_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_s8() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        let b = i8x16::new(8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1);
        let e = i8x16::new(9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9);
        let r: i8x16 = transmute(vaddq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_s16() {
        let a = i16x4::new(1, 2, 3, 4);
        let b = i16x4::new(8, 7, 6, 5);
        let e = i16x4::new(9, 9, 9, 9);
        let r: i16x4 = transmute(vadd_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_s16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let e = i16x8::new(9, 9, 9, 9, 9, 9, 9, 9);
        let r: i16x8 = transmute(vaddq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_s32() {
        let a = i32x2::new(1, 2);
        let b = i32x2::new(8, 7);
        let e = i32x2::new(9, 9);
        let r: i32x2 = transmute(vadd_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_s32() {
        let a = i32x4::new(1, 2, 3, 4);
        let b = i32x4::new(8, 7, 6, 5);
        let e = i32x4::new(9, 9, 9, 9);
        let r: i32x4 = transmute(vaddq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_u8() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let e = u8x8::new(9, 9, 9, 9, 9, 9, 9, 9);
        let r: u8x8 = transmute(vadd_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_u8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x16::new(8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1);
        let e = u8x16::new(9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9);
        let r: u8x16 = transmute(vaddq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_u16() {
        let a = u16x4::new(1, 2, 3, 4);
        let b = u16x4::new(8, 7, 6, 5);
        let e = u16x4::new(9, 9, 9, 9);
        let r: u16x4 = transmute(vadd_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u16x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let e = u16x8::new(9, 9, 9, 9, 9, 9, 9, 9);
        let r: u16x8 = transmute(vaddq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_u32() {
        let a = u32x2::new(1, 2);
        let b = u32x2::new(8, 7);
        let e = u32x2::new(9, 9);
        let r: u32x2 = transmute(vadd_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let b = u32x4::new(8, 7, 6, 5);
        let e = u32x4::new(9, 9, 9, 9);
        let r: u32x4 = transmute(vaddq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_f32() {
        let a = f32x2::new(1., 2.);
        let b = f32x2::new(8., 7.);
        let e = f32x2::new(9., 9.);
        let r: f32x2 = transmute(vadd_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_f32() {
        let a = f32x4::new(1., 2., 3., 4.);
        let b = f32x4::new(8., 7., 6., 5.);
        let e = f32x4::new(9., 9., 9., 9.);
        let r: f32x4 = transmute(vaddq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
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
    unsafe fn test_vand_s8() {
        let a = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b = i8x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let r: i8x8 = transmute(vand_s8(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s8() {
        let a = i8x16::new(
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
            0x0E, 0x00,
        );
        let b = i8x16::new(
            0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
            0x0F, 0x0F,
        );
        let r: i8x16 = transmute(vandq_s8(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s16() {
        let a = i16x4::new(0x0001, 0x0203, 0x0405, 0x0607);
        let b = i16x4::new(0x0F0F, 0x0F0F, 0x0F0F, 0x0F0F);
        let r: i16x4 = transmute(vand_s16(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s16() {
        let a = i16x8::new(
            0x0001, 0x0203, 0x0405, 0x0607, 0x0809, 0x0A0B, 0x0C0D, 0x0E0F,
        );
        let b = i16x8::new(
            0x0F0F, 0x0F0F, 0x0F0F, 0x0F0F, 0x0F0F, 0x0F0F, 0x0F0F, 0x0F0F,
        );
        let r: i16x8 = transmute(vandq_s16(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s32() {
        let a = i32x2::new(0x00010203, 0x04050607);
        let b = i32x2::new(0x0F0F0F0F, 0x0F0F0F0F);
        let r: i32x2 = transmute(vand_s32(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s32() {
        let a = i32x4::new(0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F);
        let b = i32x4::new(0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F);
        let r: i32x4 = transmute(vandq_s32(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s64() {
        let a = i64x1::new(0x0001020304050607);
        let b = i64x1::new(0x000F0F0F0F0F0F0F);
        let r: i64x1 = transmute(vand_s64(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s64() {
        let a = i64x2::new(0x0001020304050607, 0x08090A0B0C0D0E0F);
        let b = i64x2::new(0x000F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F);
        let r: i64x2 = transmute(vandq_s64(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u8() {
        let a = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b = u8x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let r: u8x8 = transmute(vand_u8(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u8() {
        let a = u8x16::new(
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
            0x0E, 0x0F,
        );
        let b = u8x16::new(
            0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
            0x0F, 0x0F,
        );
        let r: u8x16 = transmute(vandq_u8(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u16() {
        let a = u16x4::new(0x0001, 0x0203, 0x0405, 0x0607);
        let b = u16x4::new(0x0F0F, 0x0F0F, 0x0F0F, 0x0F0F);
        let r: u16x4 = transmute(vand_u16(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u16() {
        let a = u16x8::new(
            0x0001, 0x0203, 0x0405, 0x0607, 0x0809, 0x0A0B, 0x0C0D, 0x0E0F,
        );
        let b = u16x8::new(
            0x0F0F, 0x0F0F, 0x0F0F, 0x0F0F, 0x0F0F, 0x0F0F, 0x0F0F, 0x0F0F,
        );
        let r: u16x8 = transmute(vandq_u16(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u32() {
        let a = u32x2::new(0x00010203, 0x04050607);
        let b = u32x2::new(0x0F0F0F0F, 0x0F0F0F0F);
        let r: u32x2 = transmute(vand_u32(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u32() {
        let a = u32x4::new(0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F);
        let b = u32x4::new(0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F);
        let r: u32x4 = transmute(vandq_u32(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u64() {
        let a = u64x1::new(0x0001020304050607);
        let b = u64x1::new(0x0F0F0F0F0F0F0F0F);
        let r: u64x1 = transmute(vand_u64(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u64() {
        let a = u64x2::new(0x0001020304050607, 0x08090A0B0C0D0E0F);
        let b = u64x2::new(0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F);
        let r: u64x2 = transmute(vandq_u64(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s8() {
        let a = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b = i8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: i8x8 = transmute(vorr_s8(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s8() {
        let a = i8x16::new(
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
            0x0E, 0x0F,
        );
        let b = i8x16::new(
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        );
        let r: i8x16 = transmute(vorrq_s8(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s16() {
        let a = i16x4::new(0x0001, 0x0203, 0x0405, 0x0607);
        let b = i16x4::new(0x0000, 0x0000, 0x0000, 0x0000);
        let r: i16x4 = transmute(vorr_s16(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s16() {
        let a = i16x8::new(
            0x0001, 0x0203, 0x0405, 0x0607, 0x0809, 0x0A0B, 0x0C0D, 0x0E0F,
        );
        let b = i16x8::new(
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        );
        let r: i16x8 = transmute(vorrq_s16(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s32() {
        let a = i32x2::new(0x00010203, 0x04050607);
        let b = i32x2::new(0x00000000, 0x00000000);
        let r: i32x2 = transmute(vorr_s32(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s32() {
        let a = i32x4::new(0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F);
        let b = i32x4::new(0x00000000, 0x00000000, 0x00000000, 0x00000000);
        let r: i32x4 = transmute(vorrq_s32(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s64() {
        let a = i64x1::new(0x0001020304050607);
        let b = i64x1::new(0x0000000000000000);
        let r: i64x1 = transmute(vorr_s64(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s64() {
        let a = i64x2::new(0x0001020304050607, 0x08090A0B0C0D0E0F);
        let b = i64x2::new(0x0000000000000000, 0x0000000000000000);
        let r: i64x2 = transmute(vorrq_s64(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u8() {
        let a = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b = u8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: u8x8 = transmute(vorr_u8(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u8() {
        let a = u8x16::new(
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
            0x0E, 0x0F,
        );
        let b = u8x16::new(
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        );
        let r: u8x16 = transmute(vorrq_u8(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u16() {
        let a = u16x4::new(0x0001, 0x0203, 0x0405, 0x0607);
        let b = u16x4::new(0x0000, 0x0000, 0x0000, 0x0000);
        let r: u16x4 = transmute(vorr_u16(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u16() {
        let a = u16x8::new(
            0x0001, 0x0203, 0x0405, 0x0607, 0x0809, 0x0A0B, 0x0C0D, 0x0E0F,
        );
        let b = u16x8::new(
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        );
        let r: u16x8 = transmute(vorrq_u16(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u32() {
        let a = u32x2::new(0x00010203, 0x04050607);
        let b = u32x2::new(0x00000000, 0x00000000);
        let r: u32x2 = transmute(vorr_u32(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u32() {
        let a = u32x4::new(0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F);
        let b = u32x4::new(0x00000000, 0x00000000, 0x00000000, 0x00000000);
        let r: u32x4 = transmute(vorrq_u32(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u64() {
        let a = u64x1::new(0x0001020304050607);
        let b = u64x1::new(0x0000000000000000);
        let r: u64x1 = transmute(vorr_u64(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u64() {
        let a = u64x2::new(0x0001020304050607, 0x08090A0B0C0D0E0F);
        let b = u64x2::new(0x0000000000000000, 0x0000000000000000);
        let r: u64x2 = transmute(vorrq_u64(transmute(a), transmute(b)));
        assert_eq!(r, a);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s8() {
        let a = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b = i8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: i8x8 = transmute(veor_s8(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s8() {
        let a = i8x16::new(
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
            0x0E, 0x0F,
        );
        let b = i8x16::new(
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        );
        let r: i8x16 = transmute(veorq_s8(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s16() {
        let a = i16x4::new(0x0001, 0x0203, 0x0405, 0x0607);
        let b = i16x4::new(0x0000, 0x0000, 0x0000, 0x0000);
        let r: i16x4 = transmute(veor_s16(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s16() {
        let a = i16x8::new(
            0x0001, 0x0203, 0x0405, 0x0607, 0x0809, 0x0A0B, 0x0C0D, 0x0E0F,
        );
        let b = i16x8::new(
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        );
        let r: i16x8 = transmute(veorq_s16(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s32() {
        let a = i32x2::new(0x00010203, 0x04050607);
        let b = i32x2::new(0x00000000, 0x00000000);
        let r: i32x2 = transmute(veor_s32(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s32() {
        let a = i32x4::new(0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F);
        let b = i32x4::new(0x00000000, 0x00000000, 0x00000000, 0x00000000);
        let r: i32x4 = transmute(veorq_s32(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s64() {
        let a = i64x1::new(0x0001020304050607);
        let b = i64x1::new(0x0000000000000000);
        let r: i64x1 = transmute(veor_s64(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s64() {
        let a = i64x2::new(0x0001020304050607, 0x08090A0B0C0D0E0F);
        let b = i64x2::new(0x0000000000000000, 0x0000000000000000);
        let r: i64x2 = transmute(veorq_s64(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u8() {
        let a = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b = u8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: u8x8 = transmute(veor_u8(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u8() {
        let a = u8x16::new(
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
            0x0E, 0x0F,
        );
        let b = u8x16::new(
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        );
        let r: u8x16 = transmute(veorq_u8(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u16() {
        let a = u16x4::new(0x0001, 0x0203, 0x0405, 0x0607);
        let b = u16x4::new(0x0000, 0x0000, 0x0000, 0x0000);
        let r: u16x4 = transmute(veor_u16(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u16() {
        let a = u16x8::new(
            0x0001, 0x0203, 0x0405, 0x0607, 0x0809, 0x0A0B, 0x0C0D, 0x0E0F,
        );
        let b = u16x8::new(
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        );
        let r: u16x8 = transmute(veorq_u16(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u32() {
        let a = u32x2::new(0x00010203, 0x04050607);
        let b = u32x2::new(0x00000000, 0x00000000);
        let r: u32x2 = transmute(veor_u32(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u32() {
        let a = u32x4::new(0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F);
        let b = u32x4::new(0x00000000, 0x00000000, 0x00000000, 0x00000000);
        let r: u32x4 = transmute(veorq_u32(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u64() {
        let a = u64x1::new(0x0001020304050607);
        let b = u64x1::new(0x0000000000000000);
        let r: u64x1 = transmute(veor_u64(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u64() {
        let a = u64x2::new(0x0001020304050607, 0x08090A0B0C0D0E0F);
        let b = u64x2::new(0x0000000000000000, 0x0000000000000000);
        let r: u64x2 = transmute(veorq_u64(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s8() {
        let a = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b = i8x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let r: i8x8 = transmute(vceq_s8(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s8() {
        let a = i8x16::new(
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
            0x0E, 0x0F,
        );
        let b = i8x16::new(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        let r: i8x16 = transmute(vceqq_s8(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s16() {
        let a = i16x4::new(0x0001, 0x0203, 0x0405, 0x0607);
        let b = i16x4::new(-1, -1, -1, -1);
        let r: i16x4 = transmute(vceq_s16(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s16() {
        let a = i16x8::new(
            0x0001, 0x0203, 0x0405, 0x0607, 0x0809, 0x0A0B, 0x0C0D, 0x0E0F,
        );
        let b = i16x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let r: i16x8 = transmute(vceqq_s16(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s32() {
        let a = i32x2::new(0x00010203, 0x04050607);
        let b = i32x2::new(-1, -1);
        let r: i32x2 = transmute(vceq_s32(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s32() {
        let a = i32x4::new(0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F);
        let b = i32x4::new(-1, -1, -1, -1);
        let r: i32x4 = transmute(vceqq_s32(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u8() {
        let a = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vceq_u8(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u8() {
        let a = u8x16::new(
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
            0x0E, 0x0F,
        );
        let b = u8x16::new(
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF,
        );
        let r: u8x16 = transmute(vceqq_u8(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u16() {
        let a = u16x4::new(0x0001, 0x0203, 0x0405, 0x0607);
        let b = u16x4::new(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF);
        let r: u16x4 = transmute(vceq_u16(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u16() {
        let a = u16x8::new(
            0x0001, 0x0203, 0x0405, 0x0607, 0x0809, 0x0A0B, 0x0C0D, 0x0E0F,
        );
        let b = u16x8::new(
            0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        );
        let r: u16x8 = transmute(vceqq_u16(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u32() {
        let a = u32x2::new(0x00010203, 0x04050607);
        let b = u32x2::new(0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x2 = transmute(vceq_u32(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u32() {
        let a = u32x4::new(0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F);
        let b = u32x4::new(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x4 = transmute(vceqq_u32(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_f32() {
        let a = f32x2::new(1.2, 2.3);
        let b = u32x2::new(0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x2 = transmute(vceq_f32(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_f32() {
        let a = f32x4::new(1.2, 3.4, 5.6, 7.8);
        let b = u32x4::new(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x4 = transmute(vceqq_f32(transmute(a), transmute(a)));
        assert_eq!(r, b);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s8() {
        let a = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let c = i8x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let r: i8x8 = transmute(vcgt_s8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s8() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let c = i8x16::new(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        let r: i8x16 = transmute(vcgtq_s8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s16() {
        let a = i16x4::new(1, 2, 3, 4);
        let b = i16x4::new(0, 1, 2, 3);
        let c = i16x4::new(-1, -1, -1, -1);
        let r: i16x4 = transmute(vcgt_s16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let c = i16x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let r: i16x8 = transmute(vcgtq_s16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s32() {
        let a = i32x2::new(1, 2);
        let b = i32x2::new(0, 1);
        let c = i32x2::new(-1, -1);
        let r: i32x2 = transmute(vcgt_s32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s32() {
        let a = i32x4::new(1, 2, 3, 4);
        let b = i32x4::new(0, 1, 2, 3);
        let c = i32x4::new(-1, -1, -1, -1);
        let r: i32x4 = transmute(vcgtq_s32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u8() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let c = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcgt_u8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let c = u8x16::new(
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF,
        );
        let r: u8x16 = transmute(vcgtq_u8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u16() {
        let a = u16x4::new(1, 2, 3, 4);
        let b = u16x4::new(0, 1, 2, 3);
        let c = u16x4::new(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF);
        let r: u16x4 = transmute(vcgt_u16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let c = u16x8::new(
            0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        );
        let r: u16x8 = transmute(vcgtq_u16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u32() {
        let a = u32x2::new(1, 2);
        let b = u32x2::new(0, 1);
        let c = u32x2::new(0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x2 = transmute(vcgt_u32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let b = u32x4::new(0, 1, 2, 3);
        let c = u32x4::new(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x4 = transmute(vcgtq_u32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_f32() {
        let a = f32x2::new(1.2, 3.4);
        let b = f32x2::new(0.1, 2.3);
        let c = u32x2::new(0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x2 = transmute(vcgt_f32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_f32() {
        let a = f32x4::new(1.2, 3.4, 5.6, 7.8);
        let b = f32x4::new(0.1, 2.3, 4.5, 6.7);
        let c = u32x4::new(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x4 = transmute(vcgtq_f32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s8() {
        let a = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let c = i8x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let r: i8x8 = transmute(vclt_s8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s8() {
        let a = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let c = i8x16::new(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        let r: i8x16 = transmute(vcltq_s8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s16() {
        let a = i16x4::new(0, 1, 2, 3);
        let b = i16x4::new(1, 2, 3, 4);
        let c = i16x4::new(-1, -1, -1, -1);
        let r: i16x4 = transmute(vclt_s16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s16() {
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let c = i16x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let r: i16x8 = transmute(vcltq_s16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s32() {
        let a = i32x2::new(0, 1);
        let b = i32x2::new(1, 2);
        let c = i32x2::new(-1, -1);
        let r: i32x2 = transmute(vclt_s32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s32() {
        let a = i32x4::new(0, 1, 2, 3);
        let b = i32x4::new(1, 2, 3, 4);
        let c = i32x4::new(-1, -1, -1, -1);
        let r: i32x4 = transmute(vcltq_s32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let c = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vclt_u8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let c = u8x16::new(
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF,
        );
        let r: u8x16 = transmute(vcltq_u8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u16() {
        let a = u16x4::new(0, 1, 2, 3);
        let b = u16x4::new(1, 2, 3, 4);
        let c = u16x4::new(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF);
        let r: u16x4 = transmute(vclt_u16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let c = u16x8::new(
            0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        );
        let r: u16x8 = transmute(vcltq_u16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u32() {
        let a = u32x2::new(0, 1);
        let b = u32x2::new(1, 2);
        let c = u32x2::new(0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x2 = transmute(vclt_u32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u32() {
        let a = u32x4::new(0, 1, 2, 3);
        let b = u32x4::new(1, 2, 3, 4);
        let c = u32x4::new(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x4 = transmute(vcltq_u32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_f32() {
        let a = f32x2::new(0.1, 2.3);
        let b = f32x2::new(1.2, 3.4);
        let c = u32x2::new(0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x2 = transmute(vclt_f32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_f32() {
        let a = f32x4::new(0.1, 2.3, 4.5, 6.7);
        let b = f32x4::new(1.2, 3.4, 5.6, 7.8);
        let c = u32x4::new(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x4 = transmute(vcltq_f32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s8() {
        let a = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let c = i8x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let r: i8x8 = transmute(vcge_s8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s8() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let c = i8x16::new(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        let r: i8x16 = transmute(vcgeq_s8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s16() {
        let a = i16x4::new(1, 2, 3, 4);
        let b = i16x4::new(0, 1, 2, 3);
        let c = i16x4::new(-1, -1, -1, -1);
        let r: i16x4 = transmute(vcge_s16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let c = i16x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let r: i16x8 = transmute(vcgeq_s16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s32() {
        let a = i32x2::new(1, 2);
        let b = i32x2::new(0, 1);
        let c = i32x2::new(-1, -1);
        let r: i32x2 = transmute(vcge_s32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s32() {
        let a = i32x4::new(1, 2, 3, 4);
        let b = i32x4::new(0, 1, 2, 3);
        let c = i32x4::new(-1, -1, -1, -1);
        let r: i32x4 = transmute(vcgeq_s32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u8() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let c = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcge_u8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let c = u8x16::new(
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF,
        );
        let r: u8x16 = transmute(vcgeq_u8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u16() {
        let a = u16x4::new(1, 2, 3, 4);
        let b = u16x4::new(0, 1, 2, 3);
        let c = u16x4::new(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF);
        let r: u16x4 = transmute(vcge_u16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let c = u16x8::new(
            0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        );
        let r: u16x8 = transmute(vcgeq_u16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u32() {
        let a = u32x2::new(1, 2);
        let b = u32x2::new(0, 1);
        let c = u32x2::new(0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x2 = transmute(vcge_u32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let b = u32x4::new(0, 1, 2, 3);
        let c = u32x4::new(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x4 = transmute(vcgeq_u32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_f32() {
        let a = f32x2::new(1.2, 3.4);
        let b = f32x2::new(0.1, 2.3);
        let c = u32x2::new(0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x2 = transmute(vcge_f32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_f32() {
        let a = f32x4::new(1.2, 3.4, 5.6, 7.8);
        let b = f32x4::new(0.1, 2.3, 4.5, 6.7);
        let c = u32x4::new(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x4 = transmute(vcgeq_f32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s8() {
        let a = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let c = i8x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let r: i8x8 = transmute(vcle_s8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s8() {
        let a = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let c = i8x16::new(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        let r: i8x16 = transmute(vcleq_s8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s16() {
        let a = i16x4::new(0, 1, 2, 3);
        let b = i16x4::new(1, 2, 3, 4);
        let c = i16x4::new(-1, -1, -1, -1);
        let r: i16x4 = transmute(vcle_s16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s16() {
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let c = i16x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let r: i16x8 = transmute(vcleq_s16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s32() {
        let a = i32x2::new(0, 1);
        let b = i32x2::new(1, 2);
        let c = i32x2::new(-1, -1);
        let r: i32x2 = transmute(vcle_s32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s32() {
        let a = i32x4::new(0, 1, 2, 3);
        let b = i32x4::new(1, 2, 3, 4);
        let c = i32x4::new(-1, -1, -1, -1);
        let r: i32x4 = transmute(vcleq_s32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let c = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcle_u8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let c = u8x16::new(
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF,
        );
        let r: u8x16 = transmute(vcleq_u8(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u16() {
        let a = u16x4::new(0, 1, 2, 3);
        let b = u16x4::new(1, 2, 3, 4);
        let c = u16x4::new(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF);
        let r: u16x4 = transmute(vcle_u16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let c = u16x8::new(
            0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        );
        let r: u16x8 = transmute(vcleq_u16(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u32() {
        let a = u32x2::new(0, 1);
        let b = u32x2::new(1, 2);
        let c = u32x2::new(0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x2 = transmute(vcle_u32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u32() {
        let a = u32x4::new(0, 1, 2, 3);
        let b = u32x4::new(1, 2, 3, 4);
        let c = u32x4::new(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x4 = transmute(vcleq_u32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_f32() {
        let a = f32x2::new(0.1, 2.3);
        let b = f32x2::new(1.2, 3.4);
        let c = u32x2::new(0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x2 = transmute(vcle_f32(transmute(a), transmute(b)));
        assert_eq!(r, c);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_f32() {
        let a = f32x4::new(0.1, 2.3, 4.5, 6.7);
        let b = f32x4::new(1.2, 3.4, 5.6, 7.8);
        let c = u32x4::new(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        let r: u32x4 = transmute(vcleq_f32(transmute(a), transmute(b)));
        assert_eq!(r, c);
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
}

#[cfg(test)]
#[cfg(target_endian = "little")]
#[path = "table_lookup_tests.rs"]
mod table_lookup_tests;
