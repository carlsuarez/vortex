#ifndef _BLOCKS_H_
#define _BLOCKS_H_

#include <stdint.h>

struct Q8_Block {
  uint16_t d;
  int8_t qs[32];
};

struct Q4_Block {
  uint16_t d;
  uint8_t qs[16];
};

// Q4_0's layout, but each nibble indexes kvalues_iq4nl instead of having 8 subtracted
struct IQ4NL_Block {
  uint16_t d;
  uint8_t qs[16];
};

// One E8M0 scale byte, 2^(e-127), then E2M1 nibbles in Q4_0's order
struct MXFP4_Block {
  uint8_t e;
  uint8_t qs[16];
};

// Q4_K: 256 weights in 8 sub-blocks of 32. w = d*sc*q - dmin*m, with q a nibble and
// sc, m 6-bit values packed into scales[12] (get_scale_min_k4). In each 64-weight group,
// elements 0-31 are the low nibbles of qs[0..31] and elements 32-63 the high nibbles.
struct Q4K_Block {
  uint16_t d;     // fp16, scales the sub-block scales
  uint16_t dmin;  // fp16, scales the sub-block mins
  uint8_t scales[12];
  uint8_t qs[128];
};

// Q4_K's per-block header as the device reads it: everything but qs, 16 B and aligned
struct Q4K_Scales {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
};

// Q3_K: 256 weights in 16 sub-blocks of 16. w = d*(sc-32)*(q2 - (hbit ? 0 : 4)), with q2
// the low 2 bits from qs, hbit from hmask, and sc a 6-bit scale (get_scale_k3). Each 128
// weights share 32 qs bytes: shift s (0..3) takes bits 2s..2s+1 for weights 32s..32s+31,
// whose high bits are bit (4*half + s) of hmask[0..31].
struct Q3K_Block {
  uint8_t hmask[32];
  uint8_t qs[64];
  uint8_t scales[12];
  uint16_t d;  // fp16
};

// Q3_K's per-block data besides qs, as the device reads it. 46 B, so 2-byte aligned only:
// read it with byte and halfword loads.
struct Q3K_Meta {
  uint8_t hmask[32];
  uint8_t scales[12];
  uint16_t d;
};

// Q8_K activations: the K-quants' partner. fp32 scale, plus sums of each 16 quants
struct Q8K_Block {
  float d;
  int8_t qs[256];
  int16_t bsums[16];
};

static_assert(sizeof(Q8_Block) == 34, "Q8_Block not expected size");
static_assert(sizeof(Q4_Block) == 18, "Q4_Block not expected size");
static_assert(sizeof(IQ4NL_Block) == 18, "IQ4NL_Block not expected size");
static_assert(sizeof(MXFP4_Block) == 17, "MXFP4_Block not expected size");
static_assert(sizeof(Q4K_Block) == 144, "Q4K_Block not expected size");
static_assert(sizeof(Q4K_Scales) == 16, "Q4K_Scales not expected size");
static_assert(sizeof(Q3K_Block) == 110, "Q3K_Block not expected size");
static_assert(sizeof(Q3K_Meta) == 46, "Q3K_Meta not expected size");
static_assert(sizeof(Q8K_Block) == 292, "Q8K_Block not expected size");

// Sub-block j's signed scale (sc - 32) from Q3_K's 12 packed bytes. The per-byte form of
// ggml's kmask shuffle in dequantize_row_q3_K: the low nibble is the low (j < 8) or high
// (j >= 8) half of byte j % 8, and the top 2 bits are bits 2*(j/4)..+1 of byte 8 + j % 4.
static inline int get_scale_k3(int j, const uint8_t *q) {
  const int lo = j < 8 ? (q[j] & 0xF) : (q[j - 8] >> 4);
  const int hi = (q[8 + (j & 3)] >> (2 * (j >> 2))) & 3;
  return (lo | (hi << 4)) - 32;
}

// Sub-block j's 6-bit scale and min from Q4_K's 12 packed bytes. Ports ggml's
// get_scale_min_k4: j < 4 are the low 6 bits of bytes j and j+4; j >= 4 take a nibble
// from byte j+4 and their top 2 bits from the spare high bits of bytes j-4 and j.
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// ggml's IQ4_NL values, from ggml-common.h
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// ggml's E2M1 values doubled to stay integers, from ggml-common.h; e8m0_to_fp32_half undoes it
static const int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static inline float bits_to_fp32(uint32_t w) {
  float f;
  __builtin_memcpy(&f, &w, sizeof f);
  return f;
}

static inline uint32_t fp32_to_bits(float f) {
  uint32_t w;
  __builtin_memcpy(&w, &f, sizeof w);
  return w;
}

// fp16 scale bits to fp32. rv32imaf has no fp16 instructions, so this ports ggml's
// ggml_compute_fp16_to_fp32, which is exact for every input, subnormals included.
static inline float fp16_to_fp32(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & 0x80000000u;
  const uint32_t two_w = w + w;
  const float normalized = bits_to_fp32((two_w >> 4) + (0xE0u << 23)) * 0x1.0p-112f;
  const float denormalized = bits_to_fp32((two_w >> 17) | (126u << 23)) - 0.5f;
  const uint32_t result =
      sign | (two_w < (1u << 27) ? fp32_to_bits(denormalized) : fp32_to_bits(normalized));
  return bits_to_fp32(result);
}

// E8M0 scale byte to fp32, halved to match kvalues_mxfp4. Ports ggml's ggml_e8m0_to_fp32_half:
// exponents 0 and 1 give subnormals, and 255 is not treated as NaN.
static inline float e8m0_to_fp32_half(uint8_t e) {
  return bits_to_fp32(e < 2 ? 0x00200000u << e : (uint32_t)(e - 1) << 23);
}

#endif
