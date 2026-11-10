#include <iostream>
#include <getopt.h>
#include <random>
#include <string>
#include <vector>
#include <vortex2.h>
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
};

// ggml's CPU path pairs the 32-weight formats with Q8_0 activations
const ggml_type xtype = GGML_TYPE_Q8_0;

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
vx_buffer_h y_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex QGEMV (quantized matrix-vector multiplication)." << std::endl;
   std::cout << "Usage: [--type q8_0|q4_0|iq4_nl|mxfp4] [--engine scalar] [-m rows] [-k row length] [-n vectors] [--seed S] [-h: help]" << std::endl;
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
  if (!known || engine != "scalar") {
    std::cout << "Error: only --type q8_0|q4_0|iq4_nl|mxfp4 --engine scalar are implemented so far" << std::endl;
    exit(-1);
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
  ggml_get_type_traits(xtype)->to_float(x, xf.data(), xf.size());
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
  ggml_get_type_traits(xtype)->from_float_ref(h_x.data(), x_blk.data(), h_x.size());

  std::vector<uint8_t> w_qs;
  std::vector<uint8_t> w_d;
  repack_weights(w_blk.data(), h_w.size() / ggml_blck_size(wtype), w_qs, w_d);
  std::vector<int8_t> x_qs(K);
  std::vector<float> x_d(K / ggml_blck_size(xtype));
  repack_q8_0_act(reinterpret_cast<const Q8_Block*>(x_blk.data()), x_d.size(), x_qs.data(),
                  x_d.data());

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
    RT_CHECK(vx_device_max_occupancy_grid(device, 1, &M, grid_dim, block_dim));
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

  printf("QGEMV_RESULT type=%s engine=%s M=%u K=%u N=%u seed=%u nmse=%.3e\n",
         type.c_str(), engine.c_str(), M, K, N, seed, err);
  std::cout << (passed ? "PASSED!" : "FAILED!") << std::endl;
  return passed ? 0 : 1;
}
