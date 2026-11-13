#include "repack.h"
#include <string.h>
#include <ggml.h>

// The formats that pair one fp16 scale with a run of quant bytes split the same way
template <typename Block, typename Q>
static void split_blocks(const Block* src, size_t nblocks, Q* qs, uint16_t* d) {
  const size_t qk = sizeof(src->qs);
  for (size_t b = 0; b < nblocks; ++b) {
    d[b] = src[b].d;
    memcpy(qs + qk * b, src[b].qs, qk);
  }
}

void repack_q8_0(const Q8_Block* src, size_t nblocks, int8_t* qs, uint16_t* d) {
  split_blocks(src, nblocks, qs, d);
}

void repack_q4_0(const Q4_Block* src, size_t nblocks, uint8_t* qs, uint16_t* d) {
  split_blocks(src, nblocks, qs, d);
}

void repack_iq4_nl(const IQ4NL_Block* src, size_t nblocks, uint8_t* qs, uint16_t* d) {
  split_blocks(src, nblocks, qs, d);
}

void repack_mxfp4(const MXFP4_Block* src, size_t nblocks, uint8_t* qs, uint8_t* e) {
  const size_t qk = sizeof(src->qs);
  for (size_t b = 0; b < nblocks; ++b) {
    e[b] = src[b].e;
    memcpy(qs + qk * b, src[b].qs, qk);
  }
}

void repack_q4_k(const Q4K_Block* src, size_t nblocks, uint8_t* qs, Q4K_Scales* s) {
  const size_t qk = sizeof(src->qs);
  for (size_t b = 0; b < nblocks; ++b) {
    s[b].d = src[b].d;
    s[b].dmin = src[b].dmin;
    memcpy(s[b].scales, src[b].scales, sizeof(s[b].scales));
    memcpy(qs + qk * b, src[b].qs, qk);
  }
}

void repack_q3_k(const Q3K_Block* src, size_t nblocks, uint8_t* qs, Q3K_Meta* meta) {
  const size_t qk = sizeof(src->qs);
  for (size_t b = 0; b < nblocks; ++b) {
    memcpy(meta[b].hmask, src[b].hmask, sizeof(meta[b].hmask));
    memcpy(meta[b].scales, src[b].scales, sizeof(meta[b].scales));
    meta[b].d = src[b].d;
    memcpy(qs + qk * b, src[b].qs, qk);
  }
}

void repack_q8_0_act(const Q8_Block* src, size_t nblocks, int8_t* qs, float* d) {
  const size_t qk = sizeof(src->qs);
  for (size_t b = 0; b < nblocks; ++b) {
    d[b] = ggml_fp16_to_fp32(src[b].d);
    memcpy(qs + qk * b, src[b].qs, qk);
  }
}

void repack_q8_k_act(const Q8K_Block* src, size_t nblocks, int8_t* qs, float* d,
                     int16_t* bsums) {
  const size_t qk = sizeof(src->qs);
  const size_t ns = sizeof(src->bsums) / sizeof(src->bsums[0]);
  for (size_t b = 0; b < nblocks; ++b) {
    d[b] = src[b].d;
    memcpy(qs + qk * b, src[b].qs, qk);
    memcpy(bsums + ns * b, src[b].bsums, sizeof(src->bsums));
  }
}
