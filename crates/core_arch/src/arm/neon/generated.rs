
//
// THIS FILE IS GENERATED FORM neon.spec DO NOT CHANGE IT MANUALLY
//
#![rustfmt::skip]
use super::*;
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

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubu.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v8i8")]
    fn vqsub_u8_(a: uint8x8_t, a: uint8x8_t) -> uint8x8_t;
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uqsub))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsub_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    vqsub_u8_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubu.v16i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v16i8")]
    fn vqsubq_u8_(a: uint8x16_t, a: uint8x16_t) -> uint8x16_t;
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uqsub))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsubq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vqsubq_u8_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubu.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v4i16")]
    fn vqsub_u16_(a: uint16x4_t, a: uint16x4_t) -> uint16x4_t;
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uqsub))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsub_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    vqsub_u16_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubu.v8i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v8i16")]
    fn vqsubq_u16_(a: uint16x8_t, a: uint16x8_t) -> uint16x8_t;
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uqsub))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsubq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vqsubq_u16_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubu.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v2i32")]
    fn vqsub_u32_(a: uint32x2_t, a: uint32x2_t) -> uint32x2_t;
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uqsub))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsub_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    vqsub_u32_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubu.v4i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v4i32")]
    fn vqsubq_u32_(a: uint32x4_t, a: uint32x4_t) -> uint32x4_t;
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uqsub))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsubq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vqsubq_u32_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubs.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v8i8")]
    fn vqsub_s8_(a: int8x8_t, a: int8x8_t) -> int8x8_t;
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(sqsub))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsub_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    vqsub_s8_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubs.v16i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v16i8")]
    fn vqsubq_s8_(a: int8x16_t, a: int8x16_t) -> int8x16_t;
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(sqsub))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsubq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vqsubq_s8_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubs.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v4i16")]
    fn vqsub_s16_(a: int16x4_t, a: int16x4_t) -> int16x4_t;
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(sqsub))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsub_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    vqsub_s16_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubs.v8i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v8i16")]
    fn vqsubq_s16_(a: int16x8_t, a: int16x8_t) -> int16x8_t;
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(sqsub))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsubq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vqsubq_s16_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubs.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v2i32")]
    fn vqsub_s32_(a: int32x2_t, a: int32x2_t) -> int32x2_t;
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(sqsub))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsub_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    vqsub_s32_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubs.v4i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v4i32")]
    fn vqsubq_s32_(a: int32x4_t, a: int32x4_t) -> int32x4_t;
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(sqsub))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsubq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vqsubq_s32_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqaddu.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v8i8")]
    fn vqadd_u8_(a: uint8x8_t, a: uint8x8_t) -> uint8x8_t;
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uqadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqadd_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    vqadd_u8_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqaddu.v16i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v16i8")]
    fn vqaddq_u8_(a: uint8x16_t, a: uint8x16_t) -> uint8x16_t;
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uqadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vqaddq_u8_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqaddu.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v4i16")]
    fn vqadd_u16_(a: uint16x4_t, a: uint16x4_t) -> uint16x4_t;
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uqadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqadd_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    vqadd_u16_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqaddu.v8i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v8i16")]
    fn vqaddq_u16_(a: uint16x8_t, a: uint16x8_t) -> uint16x8_t;
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uqadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vqaddq_u16_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqaddu.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v2i32")]
    fn vqadd_u32_(a: uint32x2_t, a: uint32x2_t) -> uint32x2_t;
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uqadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqadd_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    vqadd_u32_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqaddu.v4i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v4i32")]
    fn vqaddq_u32_(a: uint32x4_t, a: uint32x4_t) -> uint32x4_t;
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uqadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqaddq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vqaddq_u32_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqadds.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v8i8")]
    fn vqadd_s8_(a: int8x8_t, a: int8x8_t) -> int8x8_t;
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(sqadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqadd_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    vqadd_s8_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqadds.v16i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v16i8")]
    fn vqaddq_s8_(a: int8x16_t, a: int8x16_t) -> int8x16_t;
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(sqadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vqaddq_s8_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqadds.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v4i16")]
    fn vqadd_s16_(a: int16x4_t, a: int16x4_t) -> int16x4_t;
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(sqadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqadd_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    vqadd_s16_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqadds.v8i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v8i16")]
    fn vqaddq_s16_(a: int16x8_t, a: int16x8_t) -> int16x8_t;
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(sqadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqaddq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vqaddq_s16_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqadds.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v2i32")]
    fn vqadd_s32_(a: int32x2_t, a: int32x2_t) -> int32x2_t;
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(sqadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqadd_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    vqadd_s32_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqadds.v4i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v4i32")]
    fn vqaddq_s32_(a: int32x4_t, a: int32x4_t) -> int32x4_t;
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(sqadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqaddq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vqaddq_s32_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v8i8")]
    fn vhadd_u8_(a: uint8x8_t, a: uint8x8_t) -> uint8x8_t;
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uhadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhadd_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    vhadd_u8_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v16i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v16i8")]
    fn vhaddq_u8_(a: uint8x16_t, a: uint8x16_t) -> uint8x16_t;
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uhadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vhaddq_u8_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v4i16")]
    fn vhadd_u16_(a: uint16x4_t, a: uint16x4_t) -> uint16x4_t;
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uhadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhadd_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    vhadd_u16_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v8i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v8i16")]
    fn vhaddq_u16_(a: uint16x8_t, a: uint16x8_t) -> uint16x8_t;
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uhadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vhaddq_u16_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v2i32")]
    fn vhadd_u32_(a: uint32x2_t, a: uint32x2_t) -> uint32x2_t;
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uhadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhadd_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    vhadd_u32_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v4i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v4i32")]
    fn vhaddq_u32_(a: uint32x4_t, a: uint32x4_t) -> uint32x4_t;
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(uhadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhaddq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vhaddq_u32_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v8i8")]
    fn vhadd_s8_(a: int8x8_t, a: int8x8_t) -> int8x8_t;
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(shadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhadd_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    vhadd_s8_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v16i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v16i8")]
    fn vhaddq_s8_(a: int8x16_t, a: int8x16_t) -> int8x16_t;
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(shadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vhaddq_s8_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v4i16")]
    fn vhadd_s16_(a: int16x4_t, a: int16x4_t) -> int16x4_t;
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(shadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhadd_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    vhadd_s16_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v8i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v8i16")]
    fn vhaddq_s16_(a: int16x8_t, a: int16x8_t) -> int16x8_t;
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(shadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhaddq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vhaddq_s16_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v2i32")]
    fn vhadd_s32_(a: int32x2_t, a: int32x2_t) -> int32x2_t;
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(shadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhadd_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    vhadd_s32_(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v4i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v4i32")]
    fn vhaddq_s32_(a: int32x4_t, a: int32x4_t) -> int32x4_t;
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(shadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhaddq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vhaddq_s32_(a, b)
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

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u8() {
        let a:u8x8 = u8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b:u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u8x8 = u8x8::new(41, 40, 39, 38, 37, 36, 35, 34);
        let r: u8x8 = transmute(vqsub_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u8() {
        let a:u8x16 = u8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b:u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e:u8x16 = u8x16::new(41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26);
        let r: u8x16 = transmute(vqsubq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u16() {
        let a:u16x4 = u16x4::new(42, 42, 42, 42);
        let b:u16x4 = u16x4::new(1, 2, 3, 4);
        let e:u16x4 = u16x4::new(41, 40, 39, 38);
        let r: u16x4 = transmute(vqsub_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u16() {
        let a:u16x8 = u16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b:u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u16x8 = u16x8::new(41, 40, 39, 38, 37, 36, 35, 34);
        let r: u16x8 = transmute(vqsubq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u32() {
        let a:u32x2 = u32x2::new(42, 42);
        let b:u32x2 = u32x2::new(1, 2);
        let e:u32x2 = u32x2::new(41, 40);
        let r: u32x2 = transmute(vqsub_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u32() {
        let a:u32x4 = u32x4::new(42, 42, 42, 42);
        let b:u32x4 = u32x4::new(1, 2, 3, 4);
        let e:u32x4 = u32x4::new(41, 40, 39, 38);
        let r: u32x4 = transmute(vqsubq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s8() {
        let a:i8x8 = i8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b:i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:i8x8 = i8x8::new(41, 40, 39, 38, 37, 36, 35, 34);
        let r: i8x8 = transmute(vqsub_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s8() {
        let a:i8x16 = i8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b:i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e:i8x16 = i8x16::new(41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26);
        let r: i8x16 = transmute(vqsubq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s16() {
        let a:i16x4 = i16x4::new(42, 42, 42, 42);
        let b:i16x4 = i16x4::new(1, 2, 3, 4);
        let e:i16x4 = i16x4::new(41, 40, 39, 38);
        let r: i16x4 = transmute(vqsub_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s16() {
        let a:i16x8 = i16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b:i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:i16x8 = i16x8::new(41, 40, 39, 38, 37, 36, 35, 34);
        let r: i16x8 = transmute(vqsubq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s32() {
        let a:i32x2 = i32x2::new(42, 42);
        let b:i32x2 = i32x2::new(1, 2);
        let e:i32x2 = i32x2::new(41, 40);
        let r: i32x2 = transmute(vqsub_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s32() {
        let a:i32x4 = i32x4::new(42, 42, 42, 42);
        let b:i32x4 = i32x4::new(1, 2, 3, 4);
        let e:i32x4 = i32x4::new(41, 40, 39, 38);
        let r: i32x4 = transmute(vqsubq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u8() {
        let a:u8x8 = u8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b:u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u8x8 = u8x8::new(43, 44, 45, 46, 47, 48, 49, 50);
        let r: u8x8 = transmute(vqadd_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u8() {
        let a:u8x16 = u8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b:u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e:u8x16 = u8x16::new(43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58);
        let r: u8x16 = transmute(vqaddq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u16() {
        let a:u16x4 = u16x4::new(42, 42, 42, 42);
        let b:u16x4 = u16x4::new(1, 2, 3, 4);
        let e:u16x4 = u16x4::new(43, 44, 45, 46);
        let r: u16x4 = transmute(vqadd_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u16() {
        let a:u16x8 = u16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b:u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u16x8 = u16x8::new(43, 44, 45, 46, 47, 48, 49, 50);
        let r: u16x8 = transmute(vqaddq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u32() {
        let a:u32x2 = u32x2::new(42, 42);
        let b:u32x2 = u32x2::new(1, 2);
        let e:u32x2 = u32x2::new(43, 44);
        let r: u32x2 = transmute(vqadd_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u32() {
        let a:u32x4 = u32x4::new(42, 42, 42, 42);
        let b:u32x4 = u32x4::new(1, 2, 3, 4);
        let e:u32x4 = u32x4::new(43, 44, 45, 46);
        let r: u32x4 = transmute(vqaddq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s8() {
        let a:i8x8 = i8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b:i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:i8x8 = i8x8::new(43, 44, 45, 46, 47, 48, 49, 50);
        let r: i8x8 = transmute(vqadd_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s8() {
        let a:i8x16 = i8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b:i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e:i8x16 = i8x16::new(43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58);
        let r: i8x16 = transmute(vqaddq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s16() {
        let a:i16x4 = i16x4::new(42, 42, 42, 42);
        let b:i16x4 = i16x4::new(1, 2, 3, 4);
        let e:i16x4 = i16x4::new(43, 44, 45, 46);
        let r: i16x4 = transmute(vqadd_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s16() {
        let a:i16x8 = i16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b:i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:i16x8 = i16x8::new(43, 44, 45, 46, 47, 48, 49, 50);
        let r: i16x8 = transmute(vqaddq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s32() {
        let a:i32x2 = i32x2::new(42, 42);
        let b:i32x2 = i32x2::new(1, 2);
        let e:i32x2 = i32x2::new(43, 44);
        let r: i32x2 = transmute(vqadd_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s32() {
        let a:i32x4 = i32x4::new(42, 42, 42, 42);
        let b:i32x4 = i32x4::new(1, 2, 3, 4);
        let e:i32x4 = i32x4::new(43, 44, 45, 46);
        let r: i32x4 = transmute(vqaddq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u8() {
        let a:u8x8 = u8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b:u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u8x8 = u8x8::new(21, 22, 22, 23, 23, 24, 24, 25);
        let r: u8x8 = transmute(vhadd_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u8() {
        let a:u8x16 = u8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b:u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e:u8x16 = u8x16::new(21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29);
        let r: u8x16 = transmute(vhaddq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u16() {
        let a:u16x4 = u16x4::new(42, 42, 42, 42);
        let b:u16x4 = u16x4::new(1, 2, 3, 4);
        let e:u16x4 = u16x4::new(21, 22, 22, 23);
        let r: u16x4 = transmute(vhadd_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u16() {
        let a:u16x8 = u16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b:u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:u16x8 = u16x8::new(21, 22, 22, 23, 23, 24, 24, 25);
        let r: u16x8 = transmute(vhaddq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u32() {
        let a:u32x2 = u32x2::new(42, 42);
        let b:u32x2 = u32x2::new(1, 2);
        let e:u32x2 = u32x2::new(21, 22);
        let r: u32x2 = transmute(vhadd_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u32() {
        let a:u32x4 = u32x4::new(42, 42, 42, 42);
        let b:u32x4 = u32x4::new(1, 2, 3, 4);
        let e:u32x4 = u32x4::new(21, 22, 22, 23);
        let r: u32x4 = transmute(vhaddq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s8() {
        let a:i8x8 = i8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b:i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:i8x8 = i8x8::new(21, 22, 22, 23, 23, 24, 24, 25);
        let r: i8x8 = transmute(vhadd_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s8() {
        let a:i8x16 = i8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b:i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e:i8x16 = i8x16::new(21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29);
        let r: i8x16 = transmute(vhaddq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s16() {
        let a:i16x4 = i16x4::new(42, 42, 42, 42);
        let b:i16x4 = i16x4::new(1, 2, 3, 4);
        let e:i16x4 = i16x4::new(21, 22, 22, 23);
        let r: i16x4 = transmute(vhadd_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s16() {
        let a:i16x8 = i16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b:i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e:i16x8 = i16x8::new(21, 22, 22, 23, 23, 24, 24, 25);
        let r: i16x8 = transmute(vhaddq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s32() {
        let a:i32x2 = i32x2::new(42, 42);
        let b:i32x2 = i32x2::new(1, 2);
        let e:i32x2 = i32x2::new(21, 22);
        let r: i32x2 = transmute(vhadd_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s32() {
        let a:i32x4 = i32x4::new(42, 42, 42, 42);
        let b:i32x4 = i32x4::new(1, 2, 3, 4);
        let e:i32x4 = i32x4::new(21, 22, 22, 23);
        let r: i32x4 = transmute(vhaddq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
}