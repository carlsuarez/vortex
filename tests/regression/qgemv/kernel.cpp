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
