#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t num_points;
  uint64_t src_addr;   // uint8_t [num_points], two 4-bit codes each
  uint64_t scale_addr; // float [num_points]
  uint64_t dst_addr;   // uint32_t [num_points], two fp16 each
} kernel_arg_t;

#endif
