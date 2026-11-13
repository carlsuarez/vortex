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

#ifdef VX_CFG_EXT_TCU_ENABLE
#include <vx_intrinsics.h>
#include <vx_tensor.h>

// tcu and dcmp engines: one warp per tileM rows. Each K tile of weights is decoded to fp16
// straight into fragment A, already scaled, and multiplied on the TCU against the fp16
// activations in fragment B; the accumulator stays fp32 across the whole row. The tcu engine
// decodes in software, the dcmp engine with DCMP.LD; both compute fp16(LUT[q] * scale) with
// the same fp32 product and RNE rounding, so their outputs are bit-identical.
namespace vt = vortex::tensor;
using tcu_ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, false, 8, 0, false>;
static_assert(32 % tcu_ctx::tileK == 0, "a K tile must not straddle a 32-weight block");

enum class wfmt_t { q8_0, q4_0, iq4_nl, mxfp4 };

// Logical index of every element, which load_matrix_sync scatters exactly as it would the
// data: the lane learns which (row, k) each of its fragment registers holds.
template <typename T, uint32_t N>
struct coord_table_t {
  T v[N];
  constexpr coord_table_t() : v() {
    for (uint32_t i = 0; i < N; ++i)
      v[i] = T(i);
  }
};
static constexpr coord_table_t<uint16_t, tcu_ctx::tileM * tcu_ctx::tileK> a_coords{};
static constexpr coord_table_t<float, tcu_ctx::tileM * tcu_ctx::tileN> c_coords{};

static inline float pack_fp16x2(float lo, float hi) {
  _Float16 h[2] = {(_Float16)lo, (_Float16)hi};
  float f;
  __builtin_memcpy(&f, h, sizeof f);
  return f;
}

template <wfmt_t F>
static inline float weight_scale(const uint8_t *w_d, uint32_t i) {
  if constexpr (F == wfmt_t::mxfp4)
    return e8m0_to_fp32_half(w_d[i]);
  else
    return fp16_to_fp32(reinterpret_cast<const uint16_t *>(w_d)[i]);
}

template <wfmt_t F>
static inline int32_t lut(uint32_t q) {
  if constexpr (F == wfmt_t::q4_0)
    return int32_t(q) - 8;
  else if constexpr (F == wfmt_t::iq4_nl)
    return kvalues_iq4nl[q];
  else
    return kvalues_mxfp4[q];
}

// p points at the register's two weights: two int8 for Q8_0, else one byte of nibbles
template <wfmt_t F, bool dcmp>
static inline float decode_pair(const uint8_t *p, float s) {
  if constexpr (F == wfmt_t::q8_0) {
    return pack_fp16x2(float(int8_t(p[0])) * s, float(int8_t(p[1])) * s);
  } else if constexpr (dcmp) {
#ifdef VX_CFG_EXT_DCMP_ENABLE
    return vx_dcmp_ld(p, s);
#endif
  } else {
    uint32_t byte = *p;
    return pack_fp16x2(float(lut<F>(byte & 0xf)) * s, float(lut<F>(byte >> 4)) * s);
  }
}

template <wfmt_t F, bool dcmp>
static inline void qgemv_tcu(kernel_arg_t *arg) {
  constexpr uint32_t TM = tcu_ctx::tileM, TN = tcu_ctx::tileN, TK = tcu_ctx::tileK;
  constexpr uint32_t NRA = tcu_ctx::fragment_a::NR, NRC = tcu_ctx::fragment_acc::NR;
  constexpr bool nibbles = (F != wfmt_t::q8_0);
  auto w_qs = reinterpret_cast<const uint8_t *>(arg->w_qs_addr);
  auto w_d = reinterpret_cast<const uint8_t *>(arg->w_d_addr);
  auto x16 = reinterpret_cast<const uint16_t *>(arg->x16_addr);
  auto y = reinterpret_cast<float *>(arg->y_addr);
  uint32_t M = arg->M, K = arg->K, nb = K / 32;
  uint32_t tile_row = blockIdx.x * TM;

  tcu_ctx::fragment_a fa;
  tcu_ctx::fragment_b fb;
  tcu_ctx::fragment_acc fc;

  // Per A register: the weight row's offset in w_d, and its byte offset in w_qs at k = 0.
  // Rows past M read row M-1 with a zero scale.
  uint32_t d_off[NRA], q_off[NRA];
  uint32_t valid = 0;
  tcu_ctx::load_matrix_sync(fa, a_coords.v, TK);
  for (uint32_t i = 0; i < NRA; ++i) {
    uint32_t idx = fp32_to_bits(fa.data[i]) & 0xffff;
    uint32_t row = tile_row + idx / TK, col = idx % TK;
    valid |= uint32_t(row < M) << i;
    row = (row < M) ? row : M - 1;
    d_off[i] = row * nb;
    q_off[i] = nibbles ? (row * (K / 2) + col / 2) : (row * K + col);
  }

  tcu_ctx::fill_fragment(fc, 0);
  for (uint32_t b = 0; b < nb; ++b) {
    float s[NRA];
    for (uint32_t i = 0; i < NRA; ++i)
      s[i] = ((valid >> i) & 1) ? weight_scale<F>(w_d, d_off[i] + b) : 0.0f;
    for (uint32_t t = 0; t < 32 / TK; ++t) {
      uint32_t k0 = b * 32 + t * TK;
      uint32_t q0 = nibbles ? (k0 / 2) : k0;
      for (uint32_t i = 0; i < NRA; ++i)
        fa.data[i] = decode_pair<F, dcmp>(w_qs + q_off[i] + q0, s[i]);
      // ldm 0 repeats x in every column of B; column 0 of the result is y
      tcu_ctx::load_matrix_sync<vt::col_major>(fb, x16 + k0, 0);
      tcu_ctx::mma_sync(fc, fa, fb, fc);
    }
  }

  tcu_ctx::fragment_acc pc;
  tcu_ctx::load_matrix_sync(pc, c_coords.v, TN);
  for (uint32_t i = 0; i < NRC; ++i) {
    uint32_t idx = uint32_t(pc.data[i]);
    uint32_t row = tile_row + idx / TN;
    if (idx % TN == 0 && row < M)
      y[row] = fc.data[i];
  }
}

__kernel void q8_0_tcu(kernel_arg_t *__UNIFORM__ arg) { qgemv_tcu<wfmt_t::q8_0, false>(arg); }
__kernel void q4_0_tcu(kernel_arg_t *__UNIFORM__ arg) { qgemv_tcu<wfmt_t::q4_0, false>(arg); }
__kernel void iq4_nl_tcu(kernel_arg_t *__UNIFORM__ arg) { qgemv_tcu<wfmt_t::iq4_nl, false>(arg); }
__kernel void mxfp4_tcu(kernel_arg_t *__UNIFORM__ arg) { qgemv_tcu<wfmt_t::mxfp4, false>(arg); }

#ifdef VX_CFG_EXT_DCMP_ENABLE
__kernel void q4_0_dcmp(kernel_arg_t *__UNIFORM__ arg) { qgemv_tcu<wfmt_t::q4_0, true>(arg); }
__kernel void iq4_nl_dcmp(kernel_arg_t *__UNIFORM__ arg) { qgemv_tcu<wfmt_t::iq4_nl, true>(arg); }
__kernel void mxfp4_dcmp(kernel_arg_t *__UNIFORM__ arg) { qgemv_tcu<wfmt_t::mxfp4, true>(arg); }
#endif
#endif // VX_CFG_EXT_TCU_ENABLE
