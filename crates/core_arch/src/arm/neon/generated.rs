
#![rustfmt::skip]
use super::*;
use crate::core_arch::simd_llvm::*;
#[cfg(test)]
use stdarch_test::assert_instr;

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_and(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_or(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_xor(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_eq(a, b)
}

/// Floating-point compare equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(fcmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmeq))]
pub unsafe fn vceq_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_eq(a, b)
}

/// Floating-point compare equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(fcmeq))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmeq))]
pub unsafe fn vceqq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_eq(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgt_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgtq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgt_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgtq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgt_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgtq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhi))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgt_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhi))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgtq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhi))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgt_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhi))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgtq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhi))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgt_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhi))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgtq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_gt(a, b)
}

/// Floating-point compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(fcmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmgt))]
pub unsafe fn vcgt_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_gt(a, b)
}

/// Floating-point compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(fcmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmgt))]
pub unsafe fn vcgtq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_gt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vclt_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcltq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vclt_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcltq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vclt_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcltq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhi))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vclt_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhi))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcltq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhi))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vclt_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhi))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcltq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhi))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vclt_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhi))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcltq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_lt(a, b)
}

/// Floating-point compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(fcmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmgt))]
pub unsafe fn vclt_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_lt(a, b)
}

/// Floating-point compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(fcmgt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmgt))]
pub unsafe fn vcltq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_lt(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcle_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcleq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcle_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcleq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcle_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcleq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcle_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcleq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcle_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcleq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcle_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcleq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_le(a, b)
}

/// Floating-point compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(fcmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmge))]
pub unsafe fn vcle_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_le(a, b)
}

/// Floating-point compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(fcmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmge))]
pub unsafe fn vcleq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_le(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcge_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcgeq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcge_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcgeq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcge_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcgeq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcge_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcgeq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcge_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcgeq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcge_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(cmhs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcgeq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_ge(a, b)
}

/// Floating-point compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(fcmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmge))]
pub unsafe fn vcge_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_ge(a, b)
}

/// Floating-point compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(fcmge))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmge))]
pub unsafe fn vcgeq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_ge(a, b)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core_arch::simd::*;
    use std::mem::transmute;
    use stdarch_test::simd_test;

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s8() {
        let a:i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:i8x8 = i8x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e:i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i8x8 = transmute(vand_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s8() {
        let a:i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let b:i8x16 = i8x16::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e:i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let r: i8x16 = transmute(vandq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s16() {
        let a:i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let b:i16x4 = i16x4::new(0x0F, 0x0F, 0x0F, 0x0F);
        let e:i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i16x4 = transmute(vand_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s16() {
        let a:i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:i16x8 = i16x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e:i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i16x8 = transmute(vandq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s32() {
        let a:i32x2 = i32x2::new(0x00, 0x01);
        let b:i32x2 = i32x2::new(0x0F, 0x0F);
        let e:i32x2 = i32x2::new(0x00, 0x01);
        let r: i32x2 = transmute(vand_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s32() {
        let a:i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let b:i32x4 = i32x4::new(0x0F, 0x0F, 0x0F, 0x0F);
        let e:i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i32x4 = transmute(vandq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s64() {
        let a:i64x1 = i64x1::new(0x00);
        let b:i64x1 = i64x1::new(0x0F);
        let e:i64x1 = i64x1::new(0x00);
        let r: i64x1 = transmute(vand_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s64() {
        let a:i64x2 = i64x2::new(0x00, 0x01);
        let b:i64x2 = i64x2::new(0x0F, 0x0F);
        let e:i64x2 = i64x2::new(0x00, 0x01);
        let r: i64x2 = transmute(vandq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u8() {
        let a:u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:u8x8 = u8x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e:u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u8x8 = transmute(vand_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u8() {
        let a:u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let b:u8x16 = u8x16::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e:u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let r: u8x16 = transmute(vandq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u16() {
        let a:u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let b:u16x4 = u16x4::new(0x0F, 0x0F, 0x0F, 0x0F);
        let e:u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u16x4 = transmute(vand_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u16() {
        let a:u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:u16x8 = u16x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e:u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u16x8 = transmute(vandq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u32() {
        let a:u32x2 = u32x2::new(0x00, 0x01);
        let b:u32x2 = u32x2::new(0x0F, 0x0F);
        let e:u32x2 = u32x2::new(0x00, 0x01);
        let r: u32x2 = transmute(vand_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u32() {
        let a:u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let b:u32x4 = u32x4::new(0x0F, 0x0F, 0x0F, 0x0F);
        let e:u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u32x4 = transmute(vandq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u64() {
        let a:u64x1 = u64x1::new(0x00);
        let b:u64x1 = u64x1::new(0x0F);
        let e:u64x1 = u64x1::new(0x00);
        let r: u64x1 = transmute(vand_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u64() {
        let a:u64x2 = u64x2::new(0x00, 0x01);
        let b:u64x2 = u64x2::new(0x0F, 0x0F);
        let e:u64x2 = u64x2::new(0x00, 0x01);
        let r: u64x2 = transmute(vandq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s8() {
        let a:i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:i8x8 = i8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e:i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i8x8 = transmute(vorr_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s8() {
        let a:i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b:i8x16 = i8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e:i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let r: i8x16 = transmute(vorrq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s16() {
        let a:i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let b:i16x4 = i16x4::new(0x00, 0x00, 0x00, 0x00);
        let e:i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i16x4 = transmute(vorr_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s16() {
        let a:i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:i16x8 = i16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e:i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i16x8 = transmute(vorrq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s32() {
        let a:i32x2 = i32x2::new(0x00, 0x01);
        let b:i32x2 = i32x2::new(0x00, 0x00);
        let e:i32x2 = i32x2::new(0x00, 0x01);
        let r: i32x2 = transmute(vorr_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s32() {
        let a:i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let b:i32x4 = i32x4::new(0x00, 0x00, 0x00, 0x00);
        let e:i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i32x4 = transmute(vorrq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s64() {
        let a:i64x1 = i64x1::new(0x00);
        let b:i64x1 = i64x1::new(0x00);
        let e:i64x1 = i64x1::new(0x00);
        let r: i64x1 = transmute(vorr_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s64() {
        let a:i64x2 = i64x2::new(0x00, 0x01);
        let b:i64x2 = i64x2::new(0x00, 0x00);
        let e:i64x2 = i64x2::new(0x00, 0x01);
        let r: i64x2 = transmute(vorrq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u8() {
        let a:u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:u8x8 = u8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e:u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u8x8 = transmute(vorr_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u8() {
        let a:u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b:u8x16 = u8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e:u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let r: u8x16 = transmute(vorrq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u16() {
        let a:u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let b:u16x4 = u16x4::new(0x00, 0x00, 0x00, 0x00);
        let e:u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u16x4 = transmute(vorr_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u16() {
        let a:u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:u16x8 = u16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e:u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u16x8 = transmute(vorrq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u32() {
        let a:u32x2 = u32x2::new(0x00, 0x01);
        let b:u32x2 = u32x2::new(0x00, 0x00);
        let e:u32x2 = u32x2::new(0x00, 0x01);
        let r: u32x2 = transmute(vorr_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u32() {
        let a:u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let b:u32x4 = u32x4::new(0x00, 0x00, 0x00, 0x00);
        let e:u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u32x4 = transmute(vorrq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u64() {
        let a:u64x1 = u64x1::new(0x00);
        let b:u64x1 = u64x1::new(0x00);
        let e:u64x1 = u64x1::new(0x00);
        let r: u64x1 = transmute(vorr_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u64() {
        let a:u64x2 = u64x2::new(0x00, 0x01);
        let b:u64x2 = u64x2::new(0x00, 0x00);
        let e:u64x2 = u64x2::new(0x00, 0x01);
        let r: u64x2 = transmute(vorrq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s8() {
        let a:i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:i8x8 = i8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e:i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i8x8 = transmute(veor_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s8() {
        let a:i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b:i8x16 = i8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e:i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let r: i8x16 = transmute(veorq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s16() {
        let a:i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let b:i16x4 = i16x4::new(0x00, 0x00, 0x00, 0x00);
        let e:i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i16x4 = transmute(veor_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s16() {
        let a:i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:i16x8 = i16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e:i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i16x8 = transmute(veorq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s32() {
        let a:i32x2 = i32x2::new(0x00, 0x01);
        let b:i32x2 = i32x2::new(0x00, 0x00);
        let e:i32x2 = i32x2::new(0x00, 0x01);
        let r: i32x2 = transmute(veor_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s32() {
        let a:i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let b:i32x4 = i32x4::new(0x00, 0x00, 0x00, 0x00);
        let e:i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i32x4 = transmute(veorq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s64() {
        let a:i64x1 = i64x1::new(0x00);
        let b:i64x1 = i64x1::new(0x00);
        let e:i64x1 = i64x1::new(0x00);
        let r: i64x1 = transmute(veor_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s64() {
        let a:i64x2 = i64x2::new(0x00, 0x01);
        let b:i64x2 = i64x2::new(0x00, 0x00);
        let e:i64x2 = i64x2::new(0x00, 0x01);
        let r: i64x2 = transmute(veorq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u8() {
        let a:u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:u8x8 = u8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e:u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u8x8 = transmute(veor_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u8() {
        let a:u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b:u8x16 = u8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e:u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let r: u8x16 = transmute(veorq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u16() {
        let a:u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let b:u16x4 = u16x4::new(0x00, 0x00, 0x00, 0x00);
        let e:u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u16x4 = transmute(veor_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u16() {
        let a:u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:u16x8 = u16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e:u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u16x8 = transmute(veorq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u32() {
        let a:u32x2 = u32x2::new(0x00, 0x01);
        let b:u32x2 = u32x2::new(0x00, 0x00);
        let e:u32x2 = u32x2::new(0x00, 0x01);
        let r: u32x2 = transmute(veor_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u32() {
        let a:u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let b:u32x4 = u32x4::new(0x00, 0x00, 0x00, 0x00);
        let e:u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u32x4 = transmute(veorq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u64() {
        let a:u64x1 = u64x1::new(0x00);
        let b:u64x1 = u64x1::new(0x00);
        let e:u64x1 = u64x1::new(0x00);
        let r: u64x1 = transmute(veor_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u64() {
        let a:u64x2 = u64x2::new(0x00, 0x01);
        let b:u64x2 = u64x2::new(0x00, 0x00);
        let e:u64x2 = u64x2::new(0x00, 0x01);
        let r: u64x2 = transmute(veorq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u8() {
        let a:u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e:u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vceq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u8() {
        let a:u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b:u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let e:u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vceqq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u16() {
        let a:u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let b:u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let e:u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vceq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u16() {
        let a:u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e:u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vceqq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u32() {
        let a:u32x2 = u32x2::new(0x00, 0x01);
        let b:u32x2 = u32x2::new(0x00, 0x01);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vceq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u32() {
        let a:u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let b:u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vceqq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s8() {
        let a:i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e:u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vceq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s8() {
        let a:i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b:i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let e:u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vceqq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s16() {
        let a:i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let b:i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let e:u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vceq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s16() {
        let a:i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b:i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e:u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vceqq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s32() {
        let a:i32x2 = i32x2::new(0x00, 0x01);
        let b:i32x2 = i32x2::new(0x00, 0x01);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vceq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s32() {
        let a:i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let b:i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vceqq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_f32() {
        let a:f32x2 = f32x2::new(1.2, 3.4);
        let b:f32x2 = f32x2::new(1.2, 3.4);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vceq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_f32() {
        let a:f32x4 = f32x4::new(1.2, 3.4, 5.6, 7.8);
        let b:f32x4 = f32x4::new(1.2, 3.4, 5.6, 7.8);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vceqq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s8() {
        let a:i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b:i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e:u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcgt_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s8() {
        let a:i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b:i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e:u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgtq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s16() {
        let a:i16x4 = i16x4::new(1, 2, 3, 4);
        let b:i16x4 = i16x4::new(0, 1, 2, 3);
        let e:u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcgt_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s16() {
        let a:i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b:i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e:u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgtq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s32() {
        let a:i32x2 = i32x2::new(1, 2);
        let b:i32x2 = i32x2::new(0, 1);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcgt_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s32() {
        let a:i32x4 = i32x4::new(1, 2, 3, 4);
        let b:i32x4 = i32x4::new(0, 1, 2, 3);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgtq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u8() {
        let a:u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b:u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e:u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcgt_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u8() {
        let a:u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b:u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e:u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgtq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u16() {
        let a:u16x4 = u16x4::new(1, 2, 3, 4);
        let b:u16x4 = u16x4::new(0, 1, 2, 3);
        let e:u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcgt_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u16() {
        let a:u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b:u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e:u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgtq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u32() {
        let a:u32x2 = u32x2::new(1, 2);
        let b:u32x2 = u32x2::new(0, 1);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcgt_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u32() {
        let a:u32x4 = u32x4::new(1, 2, 3, 4);
        let b:u32x4 = u32x4::new(0, 1, 2, 3);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgtq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_f32() {
        let a:f32x2 = f32x2::new(1.2, 2.3);
        let b:f32x2 = f32x2::new(0.1, 1.2);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcgt_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_f32() {
        let a:f32x4 = f32x4::new(1.2, 2.3, 3.4, 4.5);
        let b:f32x4 = f32x4::new(0.1, 1.2, 2.3, 3.4);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgtq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s8() {
        let a:i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b:i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vclt_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s8() {
        let a:i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b:i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e:u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcltq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s16() {
        let a:i16x4 = i16x4::new(0, 1, 2, 3);
        let b:i16x4 = i16x4::new(1, 2, 3, 4);
        let e:u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vclt_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s16() {
        let a:i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b:i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcltq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s32() {
        let a:i32x2 = i32x2::new(0, 1);
        let b:i32x2 = i32x2::new(1, 2);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vclt_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s32() {
        let a:i32x4 = i32x4::new(0, 1, 2, 3);
        let b:i32x4 = i32x4::new(1, 2, 3, 4);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcltq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u8() {
        let a:u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b:u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vclt_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u8() {
        let a:u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b:u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e:u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcltq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u16() {
        let a:u16x4 = u16x4::new(0, 1, 2, 3);
        let b:u16x4 = u16x4::new(1, 2, 3, 4);
        let e:u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vclt_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u16() {
        let a:u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b:u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcltq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u32() {
        let a:u32x2 = u32x2::new(0, 1);
        let b:u32x2 = u32x2::new(1, 2);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vclt_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u32() {
        let a:u32x4 = u32x4::new(0, 1, 2, 3);
        let b:u32x4 = u32x4::new(1, 2, 3, 4);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcltq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_f32() {
        let a:f32x2 = f32x2::new(0.1, 1.2);
        let b:f32x2 = f32x2::new(1.2, 2.3);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vclt_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_f32() {
        let a:f32x4 = f32x4::new(0.1, 1.2, 2.3, 3.4);
        let b:f32x4 = f32x4::new(1.2, 2.3, 3.4, 4.5);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcltq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s8() {
        let a:i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b:i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcle_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s8() {
        let a:i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b:i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e:u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcleq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s16() {
        let a:i16x4 = i16x4::new(0, 1, 2, 3);
        let b:i16x4 = i16x4::new(1, 2, 3, 4);
        let e:u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcle_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s16() {
        let a:i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b:i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcleq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s32() {
        let a:i32x2 = i32x2::new(0, 1);
        let b:i32x2 = i32x2::new(1, 2);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcle_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s32() {
        let a:i32x4 = i32x4::new(0, 1, 2, 3);
        let b:i32x4 = i32x4::new(1, 2, 3, 4);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcleq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u8() {
        let a:u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b:u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcle_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u8() {
        let a:u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b:u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e:u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcleq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u16() {
        let a:u16x4 = u16x4::new(0, 1, 2, 3);
        let b:u16x4 = u16x4::new(1, 2, 3, 4);
        let e:u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcle_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u16() {
        let a:u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b:u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcleq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u32() {
        let a:u32x2 = u32x2::new(0, 1);
        let b:u32x2 = u32x2::new(1, 2);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcle_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u32() {
        let a:u32x4 = u32x4::new(0, 1, 2, 3);
        let b:u32x4 = u32x4::new(1, 2, 3, 4);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcleq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_f32() {
        let a:f32x2 = f32x2::new(0.1, 1.2);
        let b:f32x2 = f32x2::new(1.2, 2.3);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcle_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_f32() {
        let a:f32x4 = f32x4::new(0.1, 1.2, 2.3, 3.4);
        let b:f32x4 = f32x4::new(1.2, 2.3, 3.4, 4.5);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcleq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s8() {
        let a:i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b:i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e:u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcge_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s8() {
        let a:i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b:i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e:u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgeq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s16() {
        let a:i16x4 = i16x4::new(1, 2, 3, 4);
        let b:i16x4 = i16x4::new(0, 1, 2, 3);
        let e:u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcge_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s16() {
        let a:i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b:i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e:u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgeq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s32() {
        let a:i32x2 = i32x2::new(1, 2);
        let b:i32x2 = i32x2::new(0, 1);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcge_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s32() {
        let a:i32x4 = i32x4::new(1, 2, 3, 4);
        let b:i32x4 = i32x4::new(0, 1, 2, 3);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgeq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u8() {
        let a:u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b:u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e:u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcge_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u8() {
        let a:u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b:u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e:u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgeq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u16() {
        let a:u16x4 = u16x4::new(1, 2, 3, 4);
        let b:u16x4 = u16x4::new(0, 1, 2, 3);
        let e:u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcge_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u16() {
        let a:u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b:u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e:u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgeq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u32() {
        let a:u32x2 = u32x2::new(1, 2);
        let b:u32x2 = u32x2::new(0, 1);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcge_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u32() {
        let a:u32x4 = u32x4::new(1, 2, 3, 4);
        let b:u32x4 = u32x4::new(0, 1, 2, 3);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgeq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_f32() {
        let a:f32x2 = f32x2::new(1.2, 2.3);
        let b:f32x2 = f32x2::new(0.1, 1.2);
        let e:u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcge_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_f32() {
        let a:f32x4 = f32x4::new(1.2, 2.3, 3.4, 4.5);
        let b:f32x4 = f32x4::new(0.1, 1.2, 2.3, 3.4);
        let e:u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgeq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
            }