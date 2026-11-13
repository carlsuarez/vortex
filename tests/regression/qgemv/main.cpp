#include <iostream>
#include <cstring>
#include <getopt.h>
#include <random>
#include <string>
#include <vector>
#include <vortex2.h>
#include <VX_types.h>
#include <tensor_cfg.h>
#include <ggml.h>
#include "common.h"
#include "host/repack.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

// test-backend-ops' MUL_MAT bound
const double MAX_NMSE = 5e-4;

// Weight formats with a kernel so far, by --type name
static const struct {
  const char* name;
  ggml_type type;
} formats[] = {
  {"q8_0", GGML_TYPE_Q8_0},
  {"q4_0", GGML_TYPE_Q4_0},
  {"iq4_nl", GGML_TYPE_IQ4_NL},
  {"mxfp4", GGML_TYPE_MXFP4},
  {"q4_k", GGML_TYPE_Q4_K},
  {"q3_k", GGML_TYPE_Q3_K},
};

// ggml's CPU path pairs the 32-weight formats with Q8_0 activations and the K-quants with
// Q8_K (its vec_dot_type); parse_args picks it from the weight type
ggml_type xtype = GGML_TYPE_Q8_0;

// Q8_K is only a vec_dot_type, so ggml's type traits leave its from_float_ref and to_float
// null. libggml-base still exports both; they're declared in the internal ggml-quants.h.
extern "C" {
void quantize_row_q8_K_ref(const float* x, void* y, int64_t k);
void dequantize_row_q8_K(const void* x, float* y, int64_t k);
}

static void quantize_act(const float* x, void* y, int64_t k) {
  if (xtype == GGML_TYPE_Q8_K)
    quantize_row_q8_K_ref(x, y, k);
  else
    ggml_get_type_traits(xtype)->from_float_ref(x, y, k);
}

static void dequantize_act(const void* x, float* y, int64_t k) {
  if (xtype == GGML_TYPE_Q8_K)
    dequantize_row_q8_K(x, y, k);
  else
    ggml_get_type_traits(xtype)->to_float(x, y, k);
}

const char* kernel_file = "kernel.vxbin";
std::string type = "q8_0";
ggml_type wtype = GGML_TYPE_Q8_0;
std::string engine = "scalar";
uint32_t M = 256;   // rows (output vector size)
uint32_t K = 1024;  // row length, a whole number of blocks
uint32_t N = 1;     // activation vectors
uint32_t seed = 1;

vx_device_h device = nullptr;
vx_buffer_h w_qs_buffer = nullptr;
vx_buffer_h w_d_buffer = nullptr;
vx_buffer_h x_qs_buffer = nullptr;
vx_buffer_h x_d_buffer = nullptr;
vx_buffer_h x_bsums_buffer = nullptr;
vx_buffer_h x16_buffer = nullptr;
vx_buffer_h y_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex QGEMV (quantized matrix-vector multiplication)." << std::endl;
   std::cout << "Usage: [--type q8_0|q4_0|iq4_nl|mxfp4|q4_k|q3_k] [--engine scalar|tcu|dcmp] [-m rows] [-k row length] [-n vectors] [--seed S] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  static const struct option long_options[] = {
    {"type",   required_argument, nullptr, 't'},
    {"engine", required_argument, nullptr, 'e'},
    {"seed",   required_argument, nullptr, 's'},
    {"help",   no_argument,       nullptr, 'h'},
    {nullptr,  0,                 nullptr, 0},
  };
  int c;
  while ((c = getopt_long(argc, argv, "m:k:n:h", long_options, nullptr)) != -1) {
    switch (c) {
    case 't': type = optarg; break;
    case 'e': engine = optarg; break;
    case 's': seed = atoi(optarg); break;
    case 'm': M = atoi(optarg); break;
    case 'k': K = atoi(optarg); break;
    case 'n': N = atoi(optarg); break;
    case 'h':
      show_usage();
      exit(0);
    default:
      show_usage();
      exit(-1);
    }
  }
  bool known = false;
  for (const auto& f : formats) {
    if (type == f.name) {
      wtype = f.type;
      known = true;
    }
  }
  // tcu decodes in software, dcmp with DCMP.LD (4-bit LUT formats only)
  const bool lut4 = (wtype == GGML_TYPE_Q4_0 || wtype == GGML_TYPE_IQ4_NL || wtype == GGML_TYPE_MXFP4);
  const bool supported = (engine == "scalar") || (engine == "tcu" && (lut4 || wtype == GGML_TYPE_Q8_0)) ||
                         (engine == "dcmp" && lut4);
  if (!known || !supported) {
    std::cout << "Error: implemented so far: --engine scalar for q8_0|q4_0|iq4_nl|mxfp4|q4_k|q3_k, "
                 "tcu for q8_0|q4_0|iq4_nl|mxfp4, dcmp for q4_0|iq4_nl|mxfp4" << std::endl;
    exit(-1);
  }
  if (wtype == GGML_TYPE_Q4_K || wtype == GGML_TYPE_Q3_K) {
    xtype = GGML_TYPE_Q8_K;
  }
  if (N != 1) {
    std::cout << "Error: only -n 1 (GEMV) is implemented so far" << std::endl;
    exit(-1);
  }
  const uint32_t qk = ggml_blck_size(wtype);
  if (M == 0 || K == 0 || K % qk != 0) {
    std::cout << "Error: -m must be positive and -k a positive multiple of " << qk
              << ", since a " << type << " row is whole blocks" << std::endl;
    exit(-1);
  }
}

void cleanup() {
  if (device) {
    if (w_qs_buffer) vx_buffer_release(w_qs_buffer);
    if (w_d_buffer)  vx_buffer_release(w_d_buffer);
    if (x_qs_buffer) vx_buffer_release(x_qs_buffer);
    if (x_d_buffer)  vx_buffer_release(x_d_buffer);
    if (x_bsums_buffer) vx_buffer_release(x_bsums_buffer);
    if (x16_buffer)  vx_buffer_release(x16_buffer);
    if (y_buffer)    vx_buffer_release(y_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static void upload(vx_buffer_h* buffer, uint64_t* addr, const void* data, uint64_t size) {
  RT_CHECK(vx_buffer_create(device, size, VX_MEM_READ, buffer));
  RT_CHECK(vx_buffer_address(*buffer, addr));
  RT_CHECK(vx_enqueue_write(queue, *buffer, 0, data, size, 0, nullptr, nullptr));
}

// Split ggml's weight blocks into the planes the kernels read (host/repack.h). The scale
// plane is raw bytes: fp16 bits for most formats, one E8M0 byte per block for MXFP4.
static void repack_weights(const void* blocks, size_t nblocks, std::vector<uint8_t>& qs,
                           std::vector<uint8_t>& d) {
  auto fp16_scales = [&] {
    d.resize(nblocks * sizeof(uint16_t));
    return reinterpret_cast<uint16_t*>(d.data());
  };
  switch (wtype) {
  case GGML_TYPE_Q8_0: {
    const auto* src = static_cast<const Q8_Block*>(blocks);
    qs.resize(nblocks * sizeof(src->qs));
    repack_q8_0(src, nblocks, reinterpret_cast<int8_t*>(qs.data()), fp16_scales());
    break;
  }
  case GGML_TYPE_Q4_0: {
    const auto* src = static_cast<const Q4_Block*>(blocks);
    qs.resize(nblocks * sizeof(src->qs));
    repack_q4_0(src, nblocks, qs.data(), fp16_scales());
    break;
  }
  case GGML_TYPE_IQ4_NL: {
    const auto* src = static_cast<const IQ4NL_Block*>(blocks);
    qs.resize(nblocks * sizeof(src->qs));
    repack_iq4_nl(src, nblocks, qs.data(), fp16_scales());
    break;
  }
  case GGML_TYPE_MXFP4: {
    const auto* src = static_cast<const MXFP4_Block*>(blocks);
    qs.resize(nblocks * sizeof(src->qs));
    d.resize(nblocks);
    repack_mxfp4(src, nblocks, qs.data(), d.data());
    break;
  }
  case GGML_TYPE_Q4_K: {
    const auto* src = static_cast<const Q4K_Block*>(blocks);
    qs.resize(nblocks * sizeof(src->qs));
    d.resize(nblocks * sizeof(Q4K_Scales));
    repack_q4_k(src, nblocks, qs.data(), reinterpret_cast<Q4K_Scales*>(d.data()));
    break;
  }
  case GGML_TYPE_Q3_K: {
    const auto* src = static_cast<const Q3K_Block*>(blocks);
    qs.resize(nblocks * sizeof(src->qs));
    d.resize(nblocks * sizeof(Q3K_Meta));
    repack_q3_k(src, nblocks, qs.data(), reinterpret_cast<Q3K_Meta*>(d.data()));
    break;
  }
  default:
    break;  // parse_args accepts only the formats above
  }
}

// The ggml reference: weights and activations dequantized with ggml's to_float for their
// types, dot products in double. The kernel's per-block integer sums are exact, so the
// two differ only by float rounding.
static void gemv_ref(const void* w, const void* x, float* y) {
  std::vector<float> wf(size_t(M) * K), xf(K);
  ggml_get_type_traits(wtype)->to_float(w, wf.data(), wf.size());
  dequantize_act(x, xf.data(), xf.size());
  for (uint32_t m = 0; m < M; ++m) {
    double sum = 0.0;
    for (uint32_t k = 0; k < K; ++k) {
      sum += double(wf[size_t(m) * K + k]) * xf[k];
    }
    y[m] = float(sum);
  }
}

// test-backend-ops' metric: squared error normalized by the reference's energy
static double nmse(const float* ref, const float* out, size_t n) {
  double err = 0.0, norm = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = double(ref[i]) - out[i];
    err += d * d;
    norm += double(ref[i]) * ref[i];
  }
  return err / norm;
}

// The fp16 activations the tcu and dcmp engines multiply: ggml's dequantized Q8_0 vector. For
// the nibble formats each 32-block is interleaved as (j, j+16), the order a byte's two nibbles
// land in a fragment register, so the weights keep their ggml layout.
static std::vector<uint16_t> make_x16(const std::vector<float>& x) {
  const bool nibbles = (wtype != GGML_TYPE_Q8_0);
  std::vector<uint16_t> x16(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    size_t b = i / 32, j = i % 32;
    size_t dst = nibbles ? (b * 32 + (j < 16 ? 2 * j : 2 * (j - 16) + 1)) : i;
    x16[dst] = ggml_fp32_to_fp16(x[i]);
  }
  return x16;
}

// DCMP.LD's LUT for the weight format: the same int8 tables the kernels decode with
static const int8_t* dcmp_lut() {
  static const int8_t q4_0[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
  static const int8_t iq4_nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  static const int8_t mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  return (wtype == GGML_TYPE_Q4_0) ? q4_0 : (wtype == GGML_TYPE_IQ4_NL) ? iq4_nl : mxfp4;
}

// FNV-1a over y's bits: equal for two engines only if every output matches bit for bit
static uint32_t fnv1a(const std::vector<float>& y) {
  uint32_t h = 2166136261u;
  for (float v : y) {
    uint32_t w;
    memcpy(&w, &v, 4);
    for (int i = 0; i < 4; ++i) {
      h ^= (w >> (8 * i)) & 0xff;
      h *= 16777619u;
    }
  }
  return h;
}

int main(int argc, char *argv[]) {
  // Parse command arguments
  parse_args(argc, argv);

  std::cout << "type=" << type << " engine=" << engine << " M=" << M << " K=" << K
            << " N=" << N << " seed=" << seed << std::endl;

  // Generate synthetic data, quantize it with ggml and split the blocks into planes
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> h_w(size_t(M) * K), h_x(K);
  for (auto& v : h_w) v = dist(rng);
  for (auto& v : h_x) v = dist(rng);

  std::vector<uint8_t> w_blk(M * ggml_row_size(wtype, K)), x_blk(ggml_row_size(xtype, K));
  ggml_get_type_traits(wtype)->from_float_ref(h_w.data(), w_blk.data(), h_w.size());
  quantize_act(h_x.data(), x_blk.data(), h_x.size());

  std::vector<uint8_t> w_qs;
  std::vector<uint8_t> w_d;
  repack_weights(w_blk.data(), h_w.size() / ggml_blck_size(wtype), w_qs, w_d);
  std::vector<int8_t> x_qs(K);
  std::vector<float> x_d(K / ggml_blck_size(xtype));
  std::vector<int16_t> x_bsums;
  if (xtype == GGML_TYPE_Q8_K) {
    x_bsums.resize(K / 16);
    repack_q8_k_act(reinterpret_cast<const Q8K_Block*>(x_blk.data()), x_d.size(),
                    x_qs.data(), x_d.data(), x_bsums.data());
  } else {
    repack_q8_0_act(reinterpret_cast<const Q8_Block*>(x_blk.data()), x_d.size(),
                    x_qs.data(), x_d.data());
  }

  // Open device connection
  std::cout << "Opening device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Initialize kernel arguments
  kernel_arg.M = M;
  kernel_arg.K = K;

  // Allocate device memory and upload the planes
  std::cout << "Uploading planes" << std::endl;
  upload(&w_qs_buffer, &kernel_arg.w_qs_addr, w_qs.data(), w_qs.size());
  upload(&w_d_buffer, &kernel_arg.w_d_addr, w_d.data(), w_d.size());
  upload(&x_qs_buffer, &kernel_arg.x_qs_addr, x_qs.data(), x_qs.size());
  upload(&x_d_buffer, &kernel_arg.x_d_addr, x_d.data(), x_d.size() * sizeof(float));
  // vx_enqueue_write reads the host data when it runs, so x16 must outlive the queue
  const bool on_tcu = (engine != "scalar");
  std::vector<uint16_t> x16;
  if (on_tcu) {
    std::vector<float> x_deq(K);
    dequantize_act(x_blk.data(), x_deq.data(), K);
    x16 = make_x16(x_deq);
    upload(&x16_buffer, &kernel_arg.x16_addr, x16.data(), x16.size() * sizeof(uint16_t));
  }
  if (engine == "dcmp") {
    const int8_t* lut = dcmp_lut();
    for (uint32_t w = 0; w < 4; ++w) {
      uint32_t word = 0;
      for (uint32_t j = 0; j < 4; ++j)
        word |= uint32_t(uint8_t(lut[4 * w + j])) << (8 * j);
      RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_DCMP_LUT0 + w, word, 0, nullptr, nullptr));
    }
  }
  if (!x_bsums.empty()) {
    upload(&x_bsums_buffer, &kernel_arg.x_bsums_addr, x_bsums.data(),
           x_bsums.size() * sizeof(int16_t));
  }
  const uint32_t y_size = M * sizeof(float);
  RT_CHECK(vx_buffer_create(device, y_size, VX_MEM_WRITE, &y_buffer));
  RT_CHECK(vx_buffer_address(y_buffer, &kernel_arg.y_addr));

  // Load kernel module
  std::cout << "Loading kernel" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  const std::string kernel_name = type + "_" + engine;
  if (vx_module_get_kernel(module_, kernel_name.c_str(), &kernel) != VX_SUCCESS) {
    printf("Error: no kernel '%s' in %s (not written yet, or not built under these CONFIGS)\n",
           kernel_name.c_str(), kernel_file);
    cleanup();
    exit(-1);
  }

  // Execute kernel, one thread per output row
  std::cout << "Launching kernel" << std::endl;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint32_t grid_dim[1], block_dim[1];
    if (on_tcu) {
      // one warp per tileM rows
      using cfg = vortex::tensor::wmma_config_t<VX_CFG_NUM_THREADS, vortex::tensor::fp16, vortex::tensor::fp32>;
      uint64_t nt;
      RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &nt));
      if (nt != VX_CFG_NUM_THREADS) {
        printf("Error: device warp size %u differs from VX_CFG_NUM_THREADS=%u\n", (uint32_t)nt, VX_CFG_NUM_THREADS);
        cleanup();
        exit(-1);
      }
      grid_dim[0] = (M + cfg::tileM - 1) / cfg::tileM;
      block_dim[0] = VX_CFG_NUM_THREADS;
    } else {
      RT_CHECK(vx_device_max_occupancy_grid(device, 1, &M, grid_dim, block_dim));
    }
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = grid_dim[0];
    li.block_dim[0] = block_dim[0];
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  // Download results
  std::cout << "Downloading results" << std::endl;
  std::vector<float> h_y(M, 0.0f);
  RT_CHECK(vx_enqueue_read(queue, h_y.data(), y_buffer, 0, y_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  // Verify results against the ggml reference
  std::cout << "Verifying results" << std::endl;
  std::vector<float> y_ref(M);
  gemv_ref(w_blk.data(), x_blk.data(), y_ref.data());
  const double err = nmse(y_ref.data(), h_y.data(), M);
  const bool passed = err <= MAX_NMSE;

  // Cleanup
  std::cout << "Cleaning up" << std::endl;
  cleanup();

  printf("QGEMV_RESULT type=%s engine=%s M=%u K=%u N=%u seed=%u nmse=%.3e yhash=%08x\n",
         type.c_str(), engine.c_str(), M, K, N, seed, err, fnv1a(h_y));
  std::cout << (passed ? "PASSED!" : "FAILED!") << std::endl;
  return passed ? 0 : 1;
}
