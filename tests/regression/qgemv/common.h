#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// y = W x, with W and x as planes (host/repack.h)
typedef struct {
  uint64_t w_qs_addr;  // int8 [M][K] (Q8_0) or nibble pairs [M][K/2] (4-bit formats)
  uint64_t w_d_addr;   // scales [M][K/32]: fp16 bits, or E8M0 bytes for MXFP4
  uint64_t x_qs_addr;  // int8 [K]
  uint64_t x_d_addr;   // fp32 [K/32]
  uint64_t y_addr;     // fp32 [M]
  uint32_t M;
  uint32_t K;
} kernel_arg_t;

#endif
