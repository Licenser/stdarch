// ARM Neon intrinsic specification.
// 
// This file contains the sepcification for a number of 
// intrinsics that allows us to generate them along with
// their test cases.
//
// To the syntax of the file - it's not very intelligently parsed!
//
// # Comments
// start with AT LEAST two, or four or more slashes  so // is a
// comment /////// is too.
//
// # Sections
// Sections start with EXACTLY three slasshes followed
// by AT LEAST one space. Sections are used for two things:
//
// 1) they serve as the doc comment for the given intrinics.
// 2) they reset all variables (name, fn, etc.)
//
// # Variables
//
// name    - The prefix of the function, suffixes are auto
//           generated by the type they get passed.
//
// fn      - The function to call in rust-land.
//
// aarch64 - The intrinisc to check on aarch64 architecture.
//           If this is given but no arm intrinsic is provided,
//           the function will exclusively be generated for
//           aarch64.
//           This is used to generate both aarch64 specific and
//           shared intrinics by first only specifying th aarch64
//           variant then the arm variant.
// 
// arm     - The arm v7 intrinics used to checked for arm code
//           generation. All neon functions available in arm are
//           also available in aarch64. If no aarch64 intrinic was
//           set they are assumed to be the same.
//
// a       - First input for tests, it gets scaled to the size of
//           the type.
//
// b       - Second input for tests, it gets scaled to the size of
//           the type.
//
// e       - Expected result of the test. The special values 'TRUE'
//           and 'FALSE' can be used to represent the corect NEON
//           representation of true or false values.
//           It too gets scaled to the type.
//
// # generate
// The generate command generates the intrinsics, it uses the
// Variables set and can be called multiple times while overwriting
// some of the variables.

/// Vector bitwise and
name = vand
fn = simd_and
arm = vand
aarch64 = and
a = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00
b = 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F
e = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00
generate int*_t, uint*_t, int64x*_t, uint64x*_t

/// Vector bitwise or (immediate, inclusive)
name = vorr
fn = simd_or
arm = vorr
aarch64 = orr
a = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
b = 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
e = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
generate int*_t, uint*_t, int64x*_t, uint64x*_t

/// Vector bitwise exclusive or (vector)
name = veor
fn = simd_xor
arm = veor
aarch64 = eor
a = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
b = 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
e = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
generate int*_t, uint*_t, int64x*_t, uint64x*_t

////////////////////
// equality
////////////////////

/// Compare bitwise Equal (vector)
name = vceq
fn = simd_eq
a = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
b = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE

aarch64 = cmeq
generate uint64x*_t, int64x1_t:uint64x1_t, int64x2_t:uint64x2_t, poly64x1_t:uint64x1_t, poly64x2_t:uint64x2_t

arm = cmeq
generate uint*_t, int8x8_t:uint8x8_t, int8x16_t:uint8x16_t, int16x4_t:uint16x4_t, int16x8_t:uint16x8_t, int32x2_t:uint32x2_t, int32x4_t:uint32x4_t

/// Floating-point compare equal
name = vceq
fn = simd_eq
a = 1.2, 3.4, 5.6, 7.8
b = 1.2, 3.4, 5.6, 7.8
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE

aarch64 = fcmeq
generate float64x1_t:uint64x1_t, float64x2_t:uint64x2_t

arm = fcmeq
// we are missing float16x4_t:uint16x4_t, float16x8_t:uint16x8_t
generate float32x2_t:uint32x2_t, float32x4_t:uint32x4_t

////////////////////
// greater then
////////////////////

/// Compare signed greater than
name = vcgt
fn = simd_gt
a = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
b = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE
aarch64 = cmgt
generate int64x1_t:uint64x1_t, int64x2_t:uint64x2_t

arm = cmgt
generate int8x8_t:uint8x8_t, int8x16_t:uint8x16_t, int16x4_t:uint16x4_t, int16x8_t:uint16x8_t, int32x2_t:uint32x2_t, int32x4_t:uint32x4_t

/// Compare unsigned highe
name = vcgt
fn = simd_gt
a = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
b = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE

aarch64 = cmhi
generate uint64x*_t

arm = cmhi
generate uint*_t

/// Floating-point compare greater than
name = vcgt
fn = simd_gt
a = 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9 
b = 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE

aarch64 = fcmgt
generate float64x1_t:uint64x1_t, float64x2_t:uint64x2_t

arm = fcmgt
// we are missing float16x4_t:uint16x4_t, float16x8_t:uint16x8_t
generate float32x2_t:uint32x2_t, float32x4_t:uint32x4_t

////////////////////
// lesser then
////////////////////

/// Compare signed less than
name = vclt
fn = simd_lt
a = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE
aarch64 = cmgt
generate int64x1_t:uint64x1_t, int64x2_t:uint64x2_t

arm = cmgt
generate int8x8_t:uint8x8_t, int8x16_t:uint8x16_t, int16x4_t:uint16x4_t, int16x8_t:uint16x8_t, int32x2_t:uint32x2_t, int32x4_t:uint32x4_t

/// Compare unsigned less than
name = vclt
fn = simd_lt
a = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE

aarch64 = cmhi
generate uint64x*_t

arm = cmhi
generate uint*_t

/// Floating-point compare less than
name = vclt
fn = simd_lt
a = 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8
b = 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9 
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE

aarch64 = fcmgt
generate float64x1_t:uint64x1_t, float64x2_t:uint64x2_t

arm = fcmgt
// we are missing float16x4_t:uint16x4_t, float16x8_t:uint16x8_t
generate float32x2_t:uint32x2_t, float32x4_t:uint32x4_t

////////////////////
// lesser then equals
////////////////////

/// Compare signed less than or equal
name = vcle
fn = simd_le
a = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE

aarch64 = cmge
generate int64x1_t:uint64x1_t, int64x2_t:uint64x2_t

arm = cmge
generate int8x8_t:uint8x8_t, int8x16_t:uint8x16_t, int16x4_t:uint16x4_t, int16x8_t:uint16x8_t, int32x2_t:uint32x2_t, int32x4_t:uint32x4_t

/// Compare unsigned less than or equal
name = vcle
fn = simd_le
a = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE

aarch64 = cmhs
generate uint64x*_t

arm = cmhs
generate uint*_t

/// Floating-point compare less than or equal
name = vcle
fn = simd_le
a = 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8
b = 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9 
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE
aarch64 = fcmge
generate float64x1_t:uint64x1_t, float64x2_t:uint64x2_t

// we are missing float16x4_t:uint16x4_t, float16x8_t:uint16x8_t
arm = fcmge
generate float32x2_t:uint32x2_t, float32x4_t:uint32x4_t

////////////////////
// greater then equals
////////////////////

/// Compare signed greater than or equal
name = vcge
fn = simd_ge
a = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
b = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE

aarch64 = cmge
generate int64x1_t:uint64x1_t, int64x2_t:uint64x2_t

arm = cmge
generate int8x8_t:uint8x8_t, int8x16_t:uint8x16_t, int16x4_t:uint16x4_t, int16x8_t:uint16x8_t, int32x2_t:uint32x2_t, int32x4_t:uint32x4_t

/// Compare unsigned greater than or equal
name = vcge
fn = simd_ge
a = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
b = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE

aarch64 = cmhs
generate uint64x*_t

arm = cmhs
generate uint*_t

/// Floating-point compare greater than or equal
name = vcge
fn = simd_ge
a = 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9 
b = 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8
e = TRUE, TRUE, TRUE, TRUE, TRUE, TRUE

aarch64 = fcmge
generate float64x1_t:uint64x1_t, float64x2_t:uint64x2_t

arm = fcmge
// we are missing float16x4_t:uint16x4_t, float16x8_t:uint16x8_t
generate float32x2_t:uint32x2_t, float32x4_t:uint32x4_t

/// Saturating subtract
name = vqsub
a = 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26

arm = uqsub
link-arm = llvm.arm.neon.vqsubu._EXT_
link-aarch64 = llvm.aarch64.neon.uqsub._EXT_
generate uint*_t

arm = sqsub
link-arm = llvm.arm.neon.vqsubs._EXT_
link-aarch64 = llvm.aarch64.neon.sqsub._EXT_
generate int*_t

/// Saturating add
name = vqadd
a = 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58

arm = uqadd
link-arm = llvm.arm.neon.vqaddu._EXT_
link-aarch64 = llvm.aarch64.neon.uqadd._EXT_
generate uint*_t

arm = sqadd
link-arm = llvm.arm.neon.vqadds._EXT_
link-aarch64 = llvm.aarch64.neon.sqadd._EXT_
generate int*_t

/// Halving add
name = vhadd
a = 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29

arm = uhadd
link-arm = llvm.arm.neon.vhaddu._EXT_
link-aarch64 = llvm.aarch64.neon.uhadd._EXT_
generate uint*_t

arm = shadd
link-arm = llvm.arm.neon.vhadds._EXT_
link-aarch64 = llvm.aarch64.neon.shadd._EXT_
generate int*_t