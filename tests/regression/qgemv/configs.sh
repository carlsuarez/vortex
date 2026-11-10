# CONFIGS presets for qgemv; source from build/ (run.sh does). Compare engines only within
# one preset, and pick engine and format with --args.

# scalar + WMMA with fp16 (default), int8 and MXFP4 datapaths. int8 and mxfp4 each have
# NUM_THREADS=4 CI cases; together they pass the stock sgemm_tcu (int8) and sgemm_tcu_mx (mxfp4) tests
QDEC_WMMA="-DVX_CFG_NUM_THREADS=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_INT8_ENABLE -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_MXFP4_ENABLE"
# shared-memory path: WGMMA + DXA, fp16 (matches the model_parity/perf_gate-wgmma-dxa CI cases)
QDEC_WGMMA="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_NUM_ALU_BLOCKS=1 -DVX_CFG_NUM_LSU_BLOCKS=1 -DVX_CFG_NUM_FPU_BLOCKS=1"
# decompressor runs: the same preset + -DVX_CFG_EXT_DCMP_ENABLE, so before/after differ by one flag
