// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

// What DCMP adds to a core with one LSU block: the LUT latch and the LSU
// block's DCMP path, at the configured lane count and pending-request depth.
module VX_dcmp_top import VX_gpu_pkg::*; #(
    parameter NUM_LANES = `VX_CFG_NUM_LSU_LANES,
    parameter SIZE      = `VX_CFG_LSU_PENDING_SIZE,
    parameter TAGW      = $bits(lsu_header_t),
    parameter ADDRW     = `LOG2UP(SIZE)
) (
    input wire                          clk,
    input wire                          reset,

    input wire                          dcr_req_valid,
    input dcr_req_t                     dcr_req_data,

    input wire                          req_fire,
    input wire [NUM_LANES-1:0][31:0]    req_scale,
    output wire [ADDRW-1:0]             req_slot,
    output wire                         req_full,

    input wire                          rsp_valid_in,
    output wire                         rsp_ready_in,
    input wire [ADDRW-1:0]              rsp_slot,
    input wire                          rsp_eop,
    input wire [NUM_LANES-1:0][7:0]     rsp_bytes,
    input wire [TAGW-1:0]               rsp_tag_in,

    output wire                         rsp_valid_out,
    input wire                          rsp_ready_out,
    output wire [NUM_LANES-1:0][31:0]   rsp_data_out,
    output wire [TAGW-1:0]              rsp_tag_out
);
    wire [DCMP_LUT_BITS-1:0] lut;

    VX_dcmp_dcr dcmp_dcr (
        .clk           (clk),
        .dcr_req_valid (dcr_req_valid),
        .dcr_req_data  (dcr_req_data),
        .lut           (lut)
    );

    VX_dcmp_unit #(
        .NUM_LANES (NUM_LANES),
        .SIZE      (SIZE),
        .TAGW      (TAGW)
    ) dcmp_unit (
        .clk           (clk),
        .reset         (reset),
        .lut           (lut),
        .req_fire      (req_fire),
        .req_scale     (req_scale),
        .req_slot      (req_slot),
        .req_full      (req_full),
        .rsp_valid_in  (rsp_valid_in),
        .rsp_ready_in  (rsp_ready_in),
        .rsp_slot      (rsp_slot),
        .rsp_eop       (rsp_eop),
        .rsp_bytes     (rsp_bytes),
        .rsp_tag_in    (rsp_tag_in),
        .rsp_valid_out (rsp_valid_out),
        .rsp_ready_out (rsp_ready_out),
        .rsp_data_out  (rsp_data_out),
        .rsp_tag_out   (rsp_tag_out)
    );

endmodule
