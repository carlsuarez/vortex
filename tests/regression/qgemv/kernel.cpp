#include "blocks.h"
#include "common.h"
#include <vx_spawn2.h>

// Entry points are named <type>_<engine>; main.cpp resolves them from --type and --engine.
__kernel void q8_0_scalar(kernel_arg_t *__UNIFORM__ arg) {
  auto w_qs = reinterpret_cast<const int8_t *>(arg->w_qs_addr); // [M][K]
  auto w_d = reinterpret_cast<const uint16_t *>(arg->w_d_addr); // [M][K/32], fp16 bits
  auto x_qs = reinterpret_cast<const int8_t *>(arg->x_qs_addr); // [K]
  auto x_d = reinterpret_cast<const float *>(arg->x_d_addr);    // [K/32], fp32
  auto y = reinterpret_cast<float *>(arg->y_addr);
  uint32_t M = arg->M, K = arg->K, nb = K / 32;

  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M)
    return;

  const int8_t *wq = w_qs + row * K;
  const uint16_t *wd = w_d + row * nb;
  float sum = 0.0f;
  for (uint32_t b = 0; b < nb; ++b) {
    int32_t sumi = 0;
#pragma unroll 8
    for (int j = 0; j < 32; ++j)
      sumi += wq[b * 32 + j] * x_qs[b * 32 + j];
    sum += sumi * (fp16_to_fp32(wd[b]) * x_d[b]);
  }
  y[row] = sum;
}

__kernel void q4_0_scalar(kernel_arg_t *__UNIFORM__ arg) {
  auto w_qs = reinterpret_cast<const uint8_t *>(arg->w_qs_addr); // [M][K/2]
  auto w_d = reinterpret_cast<const uint16_t *>(arg->w_d_addr);  // [M][K/32], fp16 bits
  auto x_qs = reinterpret_cast<const int8_t *>(arg->x_qs_addr);  // [K]
  auto x_d = reinterpret_cast<const float *>(arg->x_d_addr);     // [K/32], fp32
  auto y = reinterpret_cast<float *>(arg->y_addr);
  uint32_t M = arg->M, K = arg->K, nb = K / 32;

  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M)
    return;

  const uint8_t *wq = w_qs + row * (K / 2);
  const uint16_t *wd = w_d + row * nb;
  float sum = 0.0f;
  for (uint32_t b = 0; b < nb; ++b) {
    int32_t sumi0 = 0;
    int32_t sumi1 = 0;
#pragma unroll 4
    for (int j = 0; j < 16; ++j) {
      const int byte = wq[b * 16 + j];
      sumi0 += ((byte & 0x0F) - 8) * x_qs[b * 32 + j];
      sumi1 += ((byte >> 4) - 8) * x_qs[b * 32 + j + 16];
    }

    int sumi = sumi0 + sumi1;
    sum += sumi * (fp16_to_fp32(wd[b]) * x_d[b]);
  }
  y[row] = sum;
}

__kernel void iq4_nl_scalar(kernel_arg_t *__UNIFORM__ arg) {
  auto w_qs = reinterpret_cast<const uint8_t *>(arg->w_qs_addr); // [M][K/2]
  auto w_d = reinterpret_cast<const uint16_t *>(arg->w_d_addr);  // [M][K/32], fp16 bits
  auto x_qs = reinterpret_cast<const int8_t *>(arg->x_qs_addr);  // [K]
  auto x_d = reinterpret_cast<const float *>(arg->x_d_addr);     // [K/32], fp32
  auto y = reinterpret_cast<float *>(arg->y_addr);
  uint32_t M = arg->M, K = arg->K, nb = K / 32;

  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M)
    return;

  const uint8_t *wq = w_qs + row * (K / 2);
  const uint16_t *wd = w_d + row * nb;
  float sum = 0.0f;
  for (uint32_t b = 0; b < nb; ++b) {
    int32_t sumi0 = 0;
    int32_t sumi1 = 0;
#pragma unroll 4
    for (int j = 0; j < 16; ++j) {
      const int byte = wq[b * 16 + j];
      sumi0 += kvalues_iq4nl[byte & 0x0F] * x_qs[b * 32 + j];
      sumi1 += kvalues_iq4nl[byte >> 4] * x_qs[b * 32 + j + 16];
    }

    int sumi = sumi0 + sumi1;
    sum += sumi * (fp16_to_fp32(wd[b]) * x_d[b]);
  }
  y[row] = sum;
}

__kernel void mxfp4_scalar(kernel_arg_t *__UNIFORM__ arg) {
  auto w_qs = reinterpret_cast<const uint8_t *>(arg->w_qs_addr); // [M][K/2]
  auto w_e = reinterpret_cast<const uint8_t *>(arg->w_d_addr);   // [M][K/32], E8M0
  auto x_qs = reinterpret_cast<const int8_t *>(arg->x_qs_addr);  // [K]
  auto x_d = reinterpret_cast<const float *>(arg->x_d_addr);     // [K/32], fp32
  auto y = reinterpret_cast<float *>(arg->y_addr);
  uint32_t M = arg->M, K = arg->K, nb = K / 32;

  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M)
    return;

  const uint8_t *wq = w_qs + row * (K / 2);
  const uint8_t *we = w_e + row * nb;
  float sum = 0.0f;
  for (uint32_t b = 0; b < nb; ++b) {
    int32_t sumi0 = 0;
    int32_t sumi1 = 0;
#pragma unroll 4
    for (int j = 0; j < 16; ++j) {
      const int byte = wq[b * 16 + j];
      sumi0 += kvalues_mxfp4[byte & 0x0F] * x_qs[b * 32 + j];
      sumi1 += kvalues_mxfp4[byte >> 4] * x_qs[b * 32 + j + 16];
    }

    int sumi = sumi0 + sumi1;
    sum += sumi * (e8m0_to_fp32_half(we[b]) * x_d[b]);
  }
  y[row] = sum;
}

// Q4_K weights with Q8_K activations, ported from ggml's vec_dot_q4_K_q8_K. Per 256-weight
// block, sum_k w*x = xd * (d * sum_j sc_j * sum(q*xq) - dmin * sum_j m_j * sum(xq)), where
// j runs over the 8 sub-blocks of 32. The integer sums are exact; bsums supplies sum(xq)
// per 16 quants, so the min term costs 16 multiply-adds instead of 256.
__kernel void q4_k_scalar(kernel_arg_t *__UNIFORM__ arg) {
  auto w_qs = reinterpret_cast<const uint8_t *>(arg->w_qs_addr);     // [M][K/2]
  auto w_s = reinterpret_cast<const Q4K_Scales *>(arg->w_d_addr);    // [M][K/256]
  auto x_qs = reinterpret_cast<const int8_t *>(arg->x_qs_addr);      // [K]
  auto x_d = reinterpret_cast<const float *>(arg->x_d_addr);         // [K/256], fp32
  auto x_bsums = reinterpret_cast<const int16_t *>(arg->x_bsums_addr); // [K/16]
  auto y = reinterpret_cast<float *>(arg->y_addr);
  uint32_t M = arg->M, K = arg->K, nb = K / 256;

  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M)
    return;

  const uint8_t *wq = w_qs + row * (K / 2);
  const Q4K_Scales *ws = w_s + row * nb;
  float sum = 0.0f;
  for (uint32_t b = 0; b < nb; ++b) {
    const uint8_t *q4 = wq + b * 128;
    const int8_t *q8 = x_qs + b * 256;
    const int16_t *bs = x_bsums + b * 16;

    uint8_t sc[8], mn[8];
    for (int j = 0; j < 8; ++j)
      get_scale_min_k4(j, ws[b].scales, &sc[j], &mn[j]);

    // Sub-blocks 2g and 2g+1 share 32 bytes: the low nibbles, then the high nibbles
    int32_t sumi = 0;
    for (int g = 0; g < 4; ++g) {
      int32_t lo = 0, hi = 0;
#pragma unroll 4
      for (int l = 0; l < 32; ++l) {
        const int byte = q4[g * 32 + l];
        lo += (byte & 0x0F) * q8[g * 64 + l];
        hi += (byte >> 4) * q8[g * 64 + 32 + l];
      }
      sumi += sc[2 * g] * lo + sc[2 * g + 1] * hi;
    }

    // Sub-block j covers bsums 2j and 2j+1
    int32_t summ = 0;
    for (int j = 0; j < 16; ++j)
      summ += bs[j] * mn[j / 2];

    const float xd = x_d[b];
    sum += (fp16_to_fp32(ws[b].d) * xd) * sumi - (fp16_to_fp32(ws[b].dmin) * xd) * summ;
  }
  y[row] = sum;
}

// Q3_K weights with Q8_K activations, ported from ggml's vec_dot_q3_K_q8_K. Per block,
// sum_k w*x = xd * d * sum_j (sc_j - 32) * sum(q3 * xq), with j over the 16 sub-blocks of
// 16 and q3 = q2 - (hbit ? 0 : 4) in -4..3. There is no min term, so bsums go unused.
__kernel void q3_k_scalar(kernel_arg_t *__UNIFORM__ arg) {
  auto w_qs = reinterpret_cast<const uint8_t *>(arg->w_qs_addr);   // [M][K/4]
  auto w_m = reinterpret_cast<const Q3K_Meta *>(arg->w_d_addr);    // [M][K/256]
  auto x_qs = reinterpret_cast<const int8_t *>(arg->x_qs_addr);    // [K]
  auto x_d = reinterpret_cast<const float *>(arg->x_d_addr);       // [K/256], fp32
  auto y = reinterpret_cast<float *>(arg->y_addr);
  uint32_t M = arg->M, K = arg->K, nb = K / 256;

  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M)
    return;

  const uint8_t *wq = w_qs + row * (K / 4);
  const Q3K_Meta *wm = w_m + row * nb;
  float sum = 0.0f;
  for (uint32_t b = 0; b < nb; ++b) {
    const uint8_t *hm = wm[b].hmask;
    const uint8_t *sc = wm[b].scales;
    const int8_t *q8 = x_qs + b * 256;

    int32_t sumi = 0;
    for (int h = 0; h < 2; ++h) {        // 128-weight halves
      const uint8_t *q3 = wq + b * 64 + h * 32;
      for (int s = 0; s < 4; ++s) {      // 2-bit field, weights 32s..32s+31 of the half
        const int shift = 2 * s;
        const uint8_t m = 1u << (4 * h + s);
        for (int t = 0; t < 2; ++t) {    // two 16-weight sub-blocks
          const int base = t * 16;
          const int8_t *x = q8 + h * 128 + s * 32 + base;
          int32_t sub = 0;
#pragma unroll 4
          for (int l = 0; l < 16; ++l) {
            const int q = ((q3[base + l] >> shift) & 3) - ((hm[base + l] & m) ? 0 : 4);
            sub += q * x[l];
          }
          sumi += get_scale_k3(h * 8 + s * 2 + t, sc) * sub;
        }
      }
    }
    sum += (fp16_to_fp32(wm[b].d) * x_d[b]) * sumi;
  }
  y[row] = sum;
}
