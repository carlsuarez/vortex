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

static_assert(sizeof(Q8_Block) == 34, "Q8_Block not expected size");
static_assert(sizeof(Q4_Block) == 18, "Q4_Block not expected size");
static_assert(sizeof(IQ4NL_Block) == 18, "IQ4NL_Block not expected size");
static_assert(sizeof(MXFP4_Block) == 17, "MXFP4_Block not expected size");

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
