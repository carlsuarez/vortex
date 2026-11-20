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

// The LSU block's DCMP.LD path. A request keeps its lanes' rs2 scales in a slot
// whose index the caller carries to the response; the response's bytes then
// go through VX_dcmp_decode, and the slot is freed on the request's last
// response. The whole pipeline stalls together, so the caller must drain its
// output before accepting another input.
module VX_dcmp_unit import VX_gpu_pkg::*; #(
    parameter NUM_LANES = 1,
    parameter SIZE      = 1,
    parameter TAGW      = 1,
    parameter LATENCY   = `VX_CFG_DCMP_LATENCY,
    parameter ADDRW     = `LOG2UP(SIZE)
) (
    input wire                          clk,
    input wire                          reset,

    input wire [DCMP_LUT_BITS-1:0]      lut,

    // request: one slot per DCMP.LD memory request
    input wire                          req_fire,
    input wire [NUM_LANES-1:0][31:0]    req_scale,
    output wire [ADDRW-1:0]             req_slot,
    output wire                         req_full,

    // response from memory
    input wire                          rsp_valid_in,
    output wire                         rsp_ready_in,
    input wire [ADDRW-1:0]              rsp_slot,
    input wire                          rsp_eop,
    input wire [NUM_LANES-1:0][7:0]     rsp_bytes,
    input wire [TAGW-1:0]               rsp_tag_in,

    // decoded response: two fp16 per lane
    output wire                         rsp_valid_out,
    input wire                          rsp_ready_out,
    output wire [NUM_LANES-1:0][31:0]   rsp_data_out,
    output wire [TAGW-1:0]              rsp_tag_out
);
    wire enable = ~(rsp_valid_out && ~rsp_ready_out);
    wire rsp_fire = rsp_valid_in && enable;

    VX_allocator #(
        .SIZE (SIZE)
    ) allocator (
        .clk          (clk),
        .reset        (reset),
        .acquire_en   (req_fire),
        .acquire_addr (req_slot),
        .release_en   (rsp_fire && rsp_eop),
        .release_addr (rsp_slot),
        `UNUSED_PIN   (empty),
        .full         (req_full)
    );

    wire [NUM_LANES-1:0][31:0] rsp_scale;

    VX_dp_ram #(
        .DATAW    (NUM_LANES * 32),
        .SIZE     (SIZE),
        .OUT_REG  (0),
        .LUTRAM   (1),
        .RDW_MODE ("R")
    ) scale_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (req_fire),
        .wren  (1'b1),
        .waddr (req_slot),
        .wdata (req_scale),
        .raddr (rsp_slot),
        .rdata (rsp_scale)
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_decode
        VX_dcmp_decode #(
            .LATENCY (LATENCY)
        ) decode (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .lut    (lut),
            .data   (rsp_bytes[i]),
            .scale  (rsp_scale[i]),
            .result (rsp_data_out[i])
        );
    end

    VX_pipe_register #(
        .DATAW  (1 + TAGW),
        .RESETW (1),
        .DEPTH  (LATENCY)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({rsp_valid_in, rsp_tag_in}),
        .data_out ({rsp_valid_out, rsp_tag_out})
    );

    assign rsp_ready_in = enable;

endmodule
