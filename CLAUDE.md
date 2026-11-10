@AGENTS.md

# llama.cpp + DECA on Vortex (CS254A Project 12)

Two-person course project for CS254A, Fall 2026 (Prof. Blaise Tine). Reproduce DECA (Gerogiannis, Eyerman, Georganas, Heirman & Torrellas, "DECA: A Near-Core LLM Decompression Accelerator Grounded on a 3D Roofline Model", MICRO 2025, arXiv:2505.19349) on Vortex v3.0. The plan: run llama.cpp decode with quantized weights on Vortex, reproduce DECA's 3D roofline, then add a near-core decompressor that feeds the tensor unit (TCU).

The question to answer: **at what bit width does software unpacking stall decode, and how much does a decompressor buy back?**

Deliverables:
1. llama.cpp decode running on Vortex with quantized weights
2. DECA's 3D roofline reproduced for Vortex
3. A near-core decompressor (SimX, then RTL) with tokens/s and area
4. Report: what was reproduced, adapted and new

## Workstreams

- **A: kernels and llama.cpp.** Quantized GEMV kernels, the `ggml-vortex` backend, the roofline. No edits under `hw/`, `sim/` or `sw/`.
- **B: the decompressor.** Design proposal, SimX model, RTL, parity tests, area. Touches `hw/`, `sim/`, a new `sw/kernel/include/vx_dcmp.h` and a new flag in `VX_config.toml`, following AGENTS.md's hardware rules.
- Nobody edits `sw/runtime/` or the stock tests.
- The contract between A and B is the intrinsic in `vx_dcmp.h`, specified in `docs/proposals/dcmp_proposal.md`. Change it only with both owners agreeing.
- Say which workstream a change belongs to, and flag any edit to the other one's files.

## Layout

Two forks side by side: `vortex/` (this repo) and `llama.cpp/`. llama.cpp is pinned; its commit is recorded in `tests/regression/qgemv/LLAMA_COMMIT`.

In `vortex/`:
- `tests/regression/qgemv/` holds the GEMV harness and **every device kernel**. One `kernel.cpp` builds one `kernel.vxbin`, which both the harness and the llama.cpp backend load.
  - `main.cpp` + `host/`: shapes, formats, reference check, `QGEMV_RESULT` output. `blocks.h`: device-side block structs.
  - `configs.sh` (CONFIGS presets) and `run.sh` (clean, build and run with one preset; `run.sh smoke` runs the smoke set).
  - `roofline3d.py`: the 3D roofline. It reuses `ci/roofline.py`'s counters rather than forking it.
  - `results/`: generated; never hand-edit.
  - `NOTES.md`: one line per deviation from the paper, under Reproduced / Adapted / New. It feeds the report.
- Workstream B's files: `hw/rtl/dcmp/`, `sim/simx/dcmp/`, `hw/unittest/dcmp/`, `sw/kernel/include/vx_dcmp.h`, and the flag `VX_CFG_EXT_DCMP_ENABLE` (default false).
- Patterns to copy, in `tests/regression/`: `sgemv` (GEMV), `multikernel` (several kernels in one module), `sgemm_tcu` (WMMA), `sgemm_tcu_mx` (block-scaled 4-bit on the TCU), `sgemm_tcu_wg_dxa` (WGMMA fed from shared memory).

In `llama.cpp/`:
- The backend lives in `ggml/src/ggml-vortex/` and `ggml/include/ggml-vortex.h`. It also needs three touch points:
  - `option(GGML_VORTEX …)` in `ggml/CMakeLists.txt`
  - `ggml_add_backend(Vortex)` in `ggml/src/CMakeLists.txt`
  - `register_backend(ggml_backend_vortex_reg())` under `#ifdef GGML_USE_VORTEX` in `ggml/src/ggml-backend-reg.cpp`
- Skeleton: copy `ggml/src/ggml-blas`, the smallest backend (registry, device, `graph_compute` for MUL_MAT). For a device buffer type that repacks weights at load time, follow `ggml/src/ggml-zdnn`.
- The backend's CMake finds Vortex with `pkg_check_modules(VORTEX REQUIRED vortex-runtime)`, the route Vortex's `.pc` file documents.

## Commands

Vortex commands run from `vortex/build/`; llama.cpp commands run from `llama.cpp/`.

Everything here runs inside the `vortex-deca` Docker container (Ubuntu 22.04, as in Vortex's CI). The container mounts the folder holding `vortex/` and `llama.cpp/`, plus `~/tools`, at their host paths, and runs as you, so paths match on both sides. `docker/vx CMD…` runs CMD in the container in the current directory (in `build/` from the top folder or anywhere in the Vortex tree outside `build/`), passing `CONFIGS`, `LD_LIBRARY_PATH`, `PKG_CONFIG_PATH`, `VORTEX_DRIVER` and `GGML_VORTEX_*` through. `docker/vx` alone opens a shell. From the top folder or `llama.cpp/`, call it as `vortex/docker/vx` or `../vortex/docker/vx`.

```bash
# Vortex, one-time; docker/setup.sh runs this block (safe to re-run). `make install` provides the pkg-config files llama.cpp builds against
mkdir -p build && cd build && ../configure --xlen=32 --tooldir=$HOME/tools
./ci/toolchain_install.sh && make -s && make install

# harness. The app and the driver must see identical CONFIGS; run.sh wraps exactly these lines
../configure        # after creating qgemv/ or editing its Makefile
source ../tests/regression/qgemv/configs.sh
make -C tests/regression/qgemv clean
CONFIGS="$QDEC_WMMA" make -C tests/regression/qgemv
CONFIGS="$QDEC_WMMA" ./ci/blackbox.sh --driver=simx --app=qgemv --args="--type q4_0 --engine scalar -m 256 -k 1024"

# llama.cpp: a CPU-only reference build and a Vortex build of the same commit
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-cpu -j --target llama-simple llama-quantize test-backend-ops
PKG_CONFIG_PATH=$PWD/../vortex/build/install/lib/pkgconfig cmake -B build-vx -DCMAKE_BUILD_TYPE=Release -DGGML_VORTEX=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-vx -j --target llama-simple test-backend-ops

# model: the F32 TinyStories model llama.cpp's own tests use. --pure keeps every tensor in one format
mkdir -p models && wget -O models/stories15M.gguf https://huggingface.co/ggml-org/models/resolve/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories15M.gguf
for t in Q8_0 Q4_0 Q4_K Q3_K IQ4_NL; do build-cpu/bin/llama-quantize --pure models/stories15M.gguf models/stories15M-$t.gguf $t; done

# run on Vortex. This uses the driver and kernel.vxbin that run.sh last built, so run the harness with the same preset first
export LD_LIBRARY_PATH=$PWD/../vortex/build/sw/runtime VORTEX_DRIVER=simx \
       GGML_VORTEX_KERNEL=$PWD/../vortex/build/tests/regression/qgemv/kernel.vxbin
build-vx/bin/llama-simple -m models/stories15M-Q4_0.gguf -n 16 "Once upon a time"
build-vx/bin/test-backend-ops -b VORTEX0 -o MUL_MAT -p 'type_a=q4_0'
build-cpu/bin/llama-simple -m models/stories15M-Q4_0.gguf -n 16 "Once upon a time"   # reference tokens
```

- `run.sh smoke`: small shapes for every format and every engine built so far. Keep it under ~2 min and run it after every kernel change.
- Area (B): once `dcmp` is in both `hw/syn/{yosys,xilinx}/dut/catalog.mk`, run `make -C hw/syn/yosys/dut dcmp TARGET=synthesis`. `make -C hw/syn/yosys/dut list` shows every DUT.
- `ci/roofline.py` defaults to `--driver=rtlsim`, so always pass `--driver=simx`. For TCU kernels, pass `--peak-flops`.
- `--perf=<class>` adds detail: 1 core, 7 memory, 11 TCU, 16 DXA. blackbox appends `-DPERF_ENABLE` when `--perf` is set, so `run.sh` must build with it too.
- Correctness-only runs may add `-DSIMX_FUNCTIONAL` to CONFIGS: much faster, but its cycle counts are fake.
- Editor (clangd on the host): `docker/compdb.sh` records `build/compile_commands.json` from a clean rebuild. Re-record qgemv alone with `docker/compdb.sh tests/regression/qgemv` after changing its Makefile or adding source files. The top-level `.clangd` points llama.cpp files at `build-cpu/`; switch it to `build-vx/` once the backend builds.

Presets (`configs.sh`). Compare engines only within one preset, and pick engine and format with `--args`:

```bash
# scalar + WMMA with fp16 (default), int8 and MXFP4 datapaths. int8 and mxfp4 each have NUM_THREADS=4 CI cases; together they're untested
QDEC_WMMA="-DVX_CFG_NUM_THREADS=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_INT8_ENABLE -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_MXFP4_ENABLE"
# shared-memory path: WGMMA + DXA, fp16 (matches the model_parity/perf_gate-wgmma-dxa CI cases)
QDEC_WGMMA="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_NUM_ALU_BLOCKS=1 -DVX_CFG_NUM_LSU_BLOCKS=1 -DVX_CFG_NUM_FPU_BLOCKS=1"
# decompressor runs: the same preset + -DVX_CFG_EXT_DCMP_ENABLE, so before/after differ by one flag
```

If a flag combination that isn't in `ci/testcases/` misbehaves, reproduce it on the stock `sgemm_tcu*` test before debugging qgemv.

## Conventions (change them here first)

Formats, all defined in `ggml/src/ggml-common.h`. Device code mirrors them in `blocks.h` with `static_assert` on every size:

| type | bits/weight | block | layout |
|---|---|---|---|
| Q8_0 | 8.5 | 32 | fp16 d + int8[32] (34 B) |
| Q4_0 | 4.5 | 32 | fp16 d + 16 B of nibbles, w = d·(q−8) (18 B) |
| IQ4_NL | 4.5 | 32 | like Q4_0, but w = d·kvalues_iq4nl[q]: the lookup-table format |
| MXFP4 | 4.25 | 32 | E8M0 scale + 16 B of e2m1 nibbles (17 B): DECA's own format |
| Q4_K | 4.5 | 256 | 8 sub-blocks of 32, 6-bit scales and mins (144 B) |
| Q3_K | 3.4375 | 256 | 16 sub-blocks of 16, 6-bit scales (110 B): the only 3-bit format |

- Reference semantics are ggml's. Use `ggml_get_type_traits(t)->from_float_ref` and `->to_float` from libggml-base, linked from `build-cpu`, and include ggml headers with `-isystem`. For K-quants, port `dequantize_row_q3_K/q4_K` from `ggml-quants.c` literally.
- Activations follow ggml's CPU path, which is what makes token matching possible:
  - Q8_0 activations for Q8_0, Q4_0, IQ4_NL and MXFP4 weights; Q8_K for K-quants.
  - An integer dot product per block, then an fp32 scale.
- Weights may be re-laid-out at upload (in `set_tensor`) but never expanded: bytes per weight stay the format's.
- Backend: device name `VORTEX0`, type `GGML_BACKEND_DEVICE_TYPE_ACCEL`, with its own buffer type (like zDNN). llama.cpp then puts only the weights of supported ops in Vortex memory, with no `-ngl`, and the scheduler runs everything else on the CPU.
- `supports_op`: MUL_MAT only, with src0 a contiguous 2D tensor in one of the formats above and src1 contiguous F32. Return false for everything else.
- Env vars: `GGML_VORTEX_KERNEL` (absolute path to `kernel.vxbin`) and `GGML_VORTEX_ENGINE=scalar|tcu|dcmp`.
- Harness CLI (extend it, don't rename): `--type q8_0|q4_0|iq4_nl|mxfp4|q4_k|q3_k  --engine scalar|tcu|dcmp  -m M -k K -n N  --seed S`
- Output:
  - The harness prints `PASSED!` or `FAILED!`, plus one `QGEMV_RESULT key=value …` line.
  - The backend prints one `VXGGML_RESULT key=value …` line and calls `vx_device_dump_perf` when freed.
  - Analysis parses only those lines and the `PERF:` lines.

## DECA in brief

- Platform: a simulated 56-core Sapphire Rapids with HBM. The GEMM runs on AMX; decompression runs in AVX (libxsmm). Formats: BF8 and MXFP4 with group scales, plus 50–95% unstructured (bitmask) sparsity.
- 3D roofline:
  - Axes: x = AI_XM (matrix ops per DRAM byte), y = AI_XV (matrix ops per vector instruction).
  - Roofs: MEM (bandwidth), VEC (vector throughput) and MTX (matrix throughput). Attainable = min(MBW·AI_XM, VOS·AI_XV, MOS).
  - With HBM, most compressed kernels are VEC-bound: cores spend over 95% of their instructions decompressing.
- DECA itself: one unit per core, reading through the L2. LUT dequantization, then bitmask expansion, then group scaling, then output tile registers. A new instruction, TEPL, launches it out of order with up to 2 in flight.
- Results: compressed GEMM up to 4× faster than libxsmm, and next-token time 1.6–2.6× faster than software-only (Llama2-70B and OPT-66B, batch 1–16).
- Vortex mapping:
  - MEM = DRAM (`VX_CAPS_PEAK_MEM_BW`).
  - VEC = SIMT core instructions (cores × issue width × clock).
  - MTX = TCU peak, from the tile geometry in `docs/designs/tensor_core_wgmma_engine.md` §3.
- Closest existing pieces for B:
  - DXA, the async copy into shared memory (`docs/designs/dxa_async_copy_multicast.md`)
  - TCU_LD's custom load
  - the launch/wait pattern in `docs/designs/custom_accelerator_isa_extensions.md`; read that guide before choosing the intrinsic
- One 16-entry LUT stage, programmed per dispatch, covers Q4_0 (q−8), IQ4_NL and MXFP4. Only the scale decode differs (fp16 vs E8M0).

## Correctness contract

1. Harness: every format × engine matches the ggml reference with NMSE ≤ 5e-4 (test-backend-ops' MUL_MAT bound).
2. Engines agree: `dcmp` output is bit-identical to `tcu` on the same inputs, because the decompressor does exactly what software unpacking does.
3. Backend: `test-backend-ops -b VORTEX0 -o MUL_MAT` passes for each supported type. Filter with `-p`; the full sweep takes hours in SimX.
4. End to end: on Vortex, `llama-simple`'s greedy output matches the CPU-only build for the first 16 tokens. On a mismatch, compare logits before assuming a bug, since near-ties can flip.
5. Hardware (B): SimX/RTL parity per AGENTS.md (same retired instructions, cycles within 5%), through a `model_parity` case in `ci/testcases/`.

## Measurement rules

- Time comes from SimX's `PERF: … cycles=`. For per-phase splits, take differences of `VX_CSR_MCYCLE` read via `vx_device_mpm_query`. Never use wall-clock time, and never use cycles from `SIMX_FUNCTIONAL` runs.
- DRAM bytes come from the `--perf=7` line `memory: reqs=… (r=…, w=…)`, × 64 B. AI_XV uses the `PERF:` line's `instrs` count.
- Count useful MACs (M·K·N) for throughput, and report TCU utilization separately. At N=1, a tile wastes (tileN−1)/tileN of its MACs.
- tokens/s = tokens ÷ (Vortex cycles for that token's MUL_MATs ÷ clock in MHz). Host-side ops aren't simulated, so also report the share of decode MACs that ran on Vortex.
- Record both repos' git hashes, the preset, CONFIGS and args with every result.
- Start timed runs small (stories15M, ≤ 16 tokens; GEMV with K ≤ 4096). Ask before launching anything expected to take more than ~15 min.

## Gotchas

- Nibble order: llama.cpp puts element j in the low nibble of `qs[j]` and element j+16 in the high nibble. Vortex's MX path packs adjacent pairs (2i low, 2i+1 high). Repack; don't reinterpret.
- Vortex has no misaligned loads (the LSU asserts). Q4_0 and IQ4_NL blocks are 18 B and MXFP4 blocks 17 B, so word loads inside a row are misaligned. Use byte or halfword loads, or split scales and quants into aligned planes at upload.
- The TCU takes one input format for both operands, so the weight path's output must match the activations:
  - int8 follows ggml's own algorithm and is needed for token matching; fp16 is the alternative.
  - MXFP4 weights feed the TCU's native mxfp4 after a repack, but then the activations must be mxfp4 too. Use that as the no-decompression bound, not for token matching.
- K-quants need row lengths divisible by 256. stories15M's 288-wide rows silently fall back (Q3_K→Q4_0, Q4_K→Q5_0), and llama-quantize prints "required fallback quantization". Get clean 3-bit and K-quant numbers from harness shapes.
- llama.cpp loads whichever `libvortex-simx.so` comes first on `LD_LIBRARY_PATH`. If that driver wasn't built with the kernel's preset, results are silently wrong, so run the harness with that preset first.
- ggml dimensions are innermost-first. Weights are ne=[K, M] with K contiguous, and `ggml_mul_mat(W, x)` gives ne=[M, N]. That makes W the TCU's row-major A and x its column-major B.
- ggml's backend interface changes often. Stay on the pinned commit, and rebase only deliberately.
- Custom opcode space is scarce (4 opcodes × 8 funct3 rows). Find a free encoding in `hw/rtl/core/VX_decode.sv` before writing `vx_dcmp.h`. Intrinsics are `.insn` inline asm, so no compiler change is needed.
- `-Werror` is on for host and kernel code, and kernels also build with `-fno-exceptions -fno-rtti -nostdlib`.
- Never build or run on the host. `configure` rejects Debian 13, and its GCC 14 breaks simx under `-Werror` (`-Wdangling-reference` in ramulator's bundled fmt). Go through `docker/vx`.
- The container holds no state. `docker/setup.sh` recreates it when `docker/Dockerfile` changes, so add system packages there, not with `sudo apt` in the container.

## Plan

1. Baseline.
   - A: the harness with scalar software-dequant GEMV for all six formats, checked against the ggml reference; presets, `run.sh` and the smoke set.
   - B, in parallel: read DECA and the accelerator guide, then write `docs/proposals/dcmp_proposal.md`.
2. llama.cpp on Vortex (A): the `ggml-vortex` backend. Done when checks 3–4 pass on stories15M Q8_0 and Q4_0.
3. 3D roofline.
   - A: run the scalar engine and the TCU-with-software-unpack engine for every format, place each on the MEM/VEC/MTX roofs, and answer the bit-width question.
   - B: the SimX model of the decompressor and its intrinsic.
4. Decompressor.
   - A switches the kernels to `vx_dcmp.h` (check 2).
   - B does the RTL, parity and area.
   - Start with the 32-element formats, then add Q3_K's two-level scales; it's the only 3-bit point.
5. Evaluate and report: tokens/s and area at 8, 4 and 3 bits, the roofline before and after, then turn `NOTES.md` into the report.

Stretch, only after 1–4: bitmask sparsity (DECA's de-sparsification), decompressor placement (L1 vs core vs TCU), KV-cache compression, and a batch-size sweep to find where decode turns compute-bound.

## Working with us

- Two people on a short timeline: finish the default scope first, nothing speculative.
- Be direct and brief. Before multi-file changes, give a short plan.
- Show evidence: the command you ran, the `PASSED!`/`FAILED!` line, and the key `PERF:` and `*_RESULT` numbers.
