#ifndef _REPACK_H_
#define _REPACK_H_

#include <stddef.h>
#include <stdint.h>
#include "../blocks.h"

// The device reads each format as planes instead of ggml's array of blocks: the quants
// form a plain, aligned matrix and the scales get their own array. Every row is a whole
// number of blocks, so block b of a tensor lands at scale b and at b times the block's
// quant bytes in qs, with no per-row handling. Row r starts at scale r*K/32, and at
// qs[r*K] for Q8_0 or qs[r*K/2] for the 4-bit formats.

// Weights. Scales stay fp16, so the bytes per weight match the format.
void repack_q8_0(const Q8_Block* src, size_t nblocks, int8_t* qs, uint16_t* d);

// Q4_0 weights. The nibbles keep ggml's order: element j of a block is the low nibble of
// its byte j, and element j+16 the high nibble.
void repack_q4_0(const Q4_Block* src, size_t nblocks, uint8_t* qs, uint16_t* d);

// IQ4_NL weights: Q4_0's nibble order, but the nibbles index kvalues_iq4nl (blocks.h).
void repack_iq4_nl(const IQ4NL_Block* src, size_t nblocks, uint8_t* qs, uint16_t* d);

// MXFP4 weights: Q4_0's nibble order with values from kvalues_mxfp4 (blocks.h). The E8M0
// scales stay one byte each, so the bytes per weight match the format.
void repack_mxfp4(const MXFP4_Block* src, size_t nblocks, uint8_t* qs, uint8_t* e);

// Activations. Widening the scales to fp32 is exact, and leaves the kernel converting
// only the weight scales.
void repack_q8_0_act(const Q8_Block* src, size_t nblocks, int8_t* qs, float* d);

#endif
