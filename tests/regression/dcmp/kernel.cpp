#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto src   = reinterpret_cast<const uint8_t*>(arg->src_addr);
  auto scale = reinterpret_cast<const float*>(arg->scale_addr);
  auto dst   = reinterpret_cast<float*>(arg->dst_addr);
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= arg->num_points) {
    return;
  }
  dst[idx] = vx_dcmp_ld(src + idx, scale[idx]);
}
