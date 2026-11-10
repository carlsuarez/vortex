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

void repack_q8_0_act(const Q8_Block* src, size_t nblocks, int8_t* qs, float* d) {
  const size_t qk = sizeof(src->qs);
  for (size_t b = 0; b < nblocks; ++b) {
    d[b] = ggml_fp16_to_fp32(src[b].d);
    memcpy(qs + qk * b, src[b].qs, qk);
  }
}
