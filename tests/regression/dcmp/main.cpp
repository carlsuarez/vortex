// dcmp — DCMP.LD unit test: each thread decompresses one byte through the
// LUT and the result must match the host reference bit for bit, for the
// q4_0, iq4_nl and mxfp4 LUTs and scales that reach fp16 subnormals and
// overflow.

#include <vortex2.h>
#include <VX_types.h>
#include "common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <unistd.h>
#include <vector>

#define CHECK(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n", \
                     __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        std::exit(1); \
    } \
} while (0)

namespace {
const char* kernel_file = "kernel.vxbin";
uint32_t    size        = 1024;

struct Lut { const char* name; int8_t v[16]; };
const Lut luts[] = {
  {"q4_0",   {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}},
  {"iq4_nl", {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}},
  {"mxfp4",  {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}},
};

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
      case 'n': size        = std::atoi(optarg); break;
      case 'k': kernel_file = optarg;            break;
      default:
        std::cout << "Usage: [-k kernel] [-n bytes] [-h]" << std::endl;
        std::exit(c == 'h' ? 0 : -1);
    }
  }
}

// fp32 -> fp16, round to nearest even; inputs are finite
uint16_t f32_to_f16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t  exp  = int32_t((x >> 23) & 0xff);
  uint32_t mant = x & 0x7fffff;
  if (exp == 0)
    return uint16_t(sign); // fp32 subnormals are far below fp16's range
  int32_t e = exp - 127 + 15;
  if (e >= 31)
    return uint16_t(sign | 0x7c00);
  mant |= 0x800000;
  uint32_t shift = (e > 0) ? 13 : uint32_t(14 - e);
  if (shift > 24)
    return uint16_t(sign);
  uint32_t half = mant >> shift;
  uint32_t rem  = mant & ((1u << shift) - 1);
  uint32_t mid  = 1u << (shift - 1);
  if (e > 0)
    half = (uint32_t(e) << 10) | (half & 0x3ff);
  if (rem > mid || (rem == mid && (half & 1)))
    ++half;
  return uint16_t(sign | half);
}

uint32_t ref_decode(const Lut& lut, uint8_t byte, float scale) {
  uint32_t lo = f32_to_f16(float(lut.v[byte & 0xf]) * scale);
  uint32_t hi = f32_to_f16(float(lut.v[byte >> 4]) * scale);
  return (hi << 16) | lo;
}
} // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);
  const uint32_t n = size;
  std::cout << "dcmp: n=" << n << std::endl;

  std::mt19937 rng(7);
  std::vector<uint8_t> h_src(n);
  std::vector<float> h_scale(n);
  for (uint32_t i = 0; i < n; ++i) {
    h_src[i] = uint8_t(rng());
    // magnitudes 2^-30..2^14 cover fp16 subnormals, flush to zero and overflow
    int e = int(rng() % 45) - 30;
    float m = 1.0f + float(rng() & 0xffff) / 65536.0f;
    h_scale[i] = std::ldexp(m, e) * ((rng() & 1) ? -1.0f : 1.0f);
  }

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  vx_buffer_h src_buf = nullptr, scale_buf = nullptr, dst_buf = nullptr;
  CHECK(vx_buffer_create(dev, n, VX_MEM_READ, &src_buf));
  CHECK(vx_buffer_create(dev, n * sizeof(float), VX_MEM_READ, &scale_buf));
  CHECK(vx_buffer_create(dev, n * sizeof(uint32_t), VX_MEM_WRITE, &dst_buf));

  vx_module_h mod = nullptr;
  vx_kernel_h kern = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kern));

  kernel_arg_t arg{};
  arg.num_points = n;
  CHECK(vx_buffer_address(src_buf, &arg.src_addr));
  CHECK(vx_buffer_address(scale_buf, &arg.scale_addr));
  CHECK(vx_buffer_address(dst_buf, &arg.dst_addr));

  CHECK(vx_enqueue_write(q, src_buf, 0, h_src.data(), n, 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(q, scale_buf, 0, h_scale.data(), n * sizeof(float), 0, nullptr, nullptr));

  uint32_t grid[1], block[1];
  CHECK(vx_device_max_occupancy_grid(dev, 1, &n, grid, block));

  int errors = 0;
  std::vector<uint32_t> h_dst(n);
  for (const auto& lut : luts) {
    for (uint32_t w = 0; w < 4; ++w) {
      uint32_t word = 0;
      for (uint32_t j = 0; j < 4; ++j)
        word |= uint32_t(uint8_t(lut.v[4 * w + j])) << (8 * j);
      CHECK(vx_enqueue_dcr_write(q, VX_DCR_DCMP_LUT0 + w, word, 0, nullptr, nullptr));
    }
    vx_launch_info_t li{};
    li.struct_size = sizeof(li);
    li.kernel      = kern;
    li.args_host   = &arg;
    li.args_size   = sizeof(arg);
    li.ndim        = 1;
    li.grid_dim[0] = grid[0];
    li.block_dim[0]= block[0];
    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, h_dst.data(), dst_buf, 0, n * sizeof(uint32_t), 1, &launch_ev, &read_ev));
    CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    int lut_errors = 0;
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t ref = ref_decode(lut, h_src[i], h_scale[i]);
      if (h_dst[i] != ref) {
        if (lut_errors < 8)
          std::printf("*** %s [%u] byte=0x%02x scale=%g expected=0x%08x actual=0x%08x\n",
                      lut.name, i, h_src[i], h_scale[i], ref, h_dst[i]);
        ++lut_errors;
      }
    }
    std::printf("%s: %d/%u mismatches\n", lut.name, lut_errors, n);
    errors += lut_errors;
  }

  vx_buffer_release(dst_buf);
  vx_buffer_release(scale_buf);
  vx_buffer_release(src_buf);
  vx_kernel_release(kern);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);

  if (errors) {
    std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
