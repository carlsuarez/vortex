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

// DCMP.LD datapath: one byte of two 4-bit codes -> two fp16 in one 32-bit word,
//   result[16h +: 16] = fp16_rne(fp32_rne(LUT[data[4h +: 4]] * scale))
// The product is rounded to fp32 before fp16, as an fmul then fcvt.h.s would:
// the scale carries 24 bits, so a single rounding of the exact product differs
// from that in rare ties, and the dcmp and tcu engines must agree bit for bit.
// Stage 1 looks the codes up and rounds the exact product (int8 x 24 bit) to a
// 24-bit significand; stage 2 normalizes and rounds it to fp16.
module VX_dcmp_decode #(
    parameter LATENCY = 2
) (
    input wire         clk,
    input wire         reset,
    input wire         enable,
    input wire [127:0] lut,     // 16 x int8, entry i in bits [8i+7:8i]
    input wire [7:0]   data,
    input wire [31:0]  scale,   // fp32
    output wire [31:0] result
);
    `STATIC_ASSERT (LATENCY >= 2, ("invalid parameter"))

    localparam [15:0] FP16_QNAN = 16'h7e00;
    localparam [14:0] FP16_INF  = 15'h7c00;

    // stage 1: lookup, exact product, round to fp32's 24-bit significand

    wire        s_sign = scale[31];
    wire [7:0]  s_exp  = scale[30:23];
    wire [22:0] s_frac = scale[22:0];

    wire s_is_nan  = (s_exp == 8'hff) && (s_frac != 0);
    wire s_is_inf  = (s_exp == 8'hff) && (s_frac == 0);
    // zero or subnormal scale: the product is below 2^-119 and rounds to fp16 zero
    wire s_is_tiny = (s_exp == 8'h00);

    wire [1:0]       st1_sign, st1_nan, st1_inf, st1_zero;
    wire [1:0][31:0] st1_prod;

    for (genvar h = 0; h < 2; ++h) begin : g_lookup
        wire [3:0] code  = data[4*h +: 4];
        wire [7:0] entry = lut[8*code +: 8];
        wire [7:0] mag   = entry[7] ? (~entry + 8'd1) : entry; // |entry| <= 128

        // a non-zero entry puts the exact product's leading one in bits [30:23]
        wire [30:0] prod = 31'(mag * {1'b1, s_frac}); // < 128 * 2^24
        wire [2:0] lz;
        wire       lz_valid;
        VX_lzc #(
            .N (8)
        ) lzc (
            .data_in   (prod[30:23]),
            .data_out  (lz),
            .valid_out (lz_valid)
        );
        `UNUSED_VAR (lz_valid)

        // keep 24 significant bits; a carry can reach 2^24 << drop = 2^31
        wire [2:0]  drop     = 3'd7 - lz;
        wire [23:0] kept     = 24'(prod >> drop); // < 2^24
        wire [30:0] dropped  = prod & ~(31'h7fffffff << drop);
        wire [30:0] halfway  = (drop == 0) ? 31'b0 : (31'b1 << (drop - 3'd1));
        wire        round_up = (drop != 0) && ((dropped > halfway) || ((dropped == halfway) && kept[0]));
        wire [24:0] sig      = 25'(kept) + 25'(round_up);

        assign st1_sign[h] = entry[7] ^ s_sign;
        assign st1_nan[h]  = s_is_nan || (s_is_inf && (mag == 0));
        assign st1_inf[h]  = s_is_inf && (mag != 0);
        assign st1_zero[h] = (mag == 0) || s_is_tiny;
        assign st1_prod[h] = 32'(sig) << drop;
    end

    wire [1:0]       st2_sign, st2_nan, st2_inf, st2_zero;
    wire [1:0][31:0] st2_prod;
    wire [7:0]       st2_exp;

    VX_pipe_register #(
        .DATAW (2 * (4 + 32) + 8),
        .DEPTH (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({st1_sign, st1_nan, st1_inf, st1_zero, st1_prod, s_exp}),
        .data_out ({st2_sign, st2_nan, st2_inf, st2_zero, st2_prod, st2_exp})
    );

    // stage 2: normalize and round to fp16, nearest even

    wire [31:0] st2_result;

    for (genvar h = 0; h < 2; ++h) begin : g_round
        wire [31:0] prod = st2_prod[h];

        // the rounded product's leading one sits in bits [31:23]
        wire [3:0] lz;
        wire       lz_valid;
        VX_lzc #(
            .N (9)
        ) lzc (
            .data_in   (prod[31:23]),
            .data_out  (lz),
            .valid_out (lz_valid)
        );
        `UNUSED_VAR (lz_valid)

        // value = prod * 2^(exp - 150), leading one at bit msb
        wire [4:0] msb = 5'd31 - 5'(lz);
        // fp16 biased exponent if normal: msb + exp - 150 + 15
        wire signed [10:0] e16 = 11'(msb) + 11'(st2_exp) - 11'sd135;
        wire is_normal = (e16 >= 1);
        wire is_ovf    = (e16 >= 31);

        // a normal keeps msb-10 bits below its 11-bit significand; a subnormal's
        // unit is 2^-24, so it drops 126 - exp bits
        wire [7:0] sub_shift = 8'd126 - st2_exp;
        wire [5:0] shift = is_normal ? 6'(msb - 5'd10)
                                     : ((sub_shift > 8'd32) ? 6'd32 : 6'(sub_shift));
        wire [63:0] shifted  = {prod, 32'b0} >> shift;
        wire [14:0] kept     = shifted[46:32]; // < 2^11
        wire        guard    = shifted[31];
        wire        sticky   = (shifted[30:0] != 0);
        wire        round_up = guard && (sticky || kept[0]);
        `UNUSED_VAR (shifted[63:47])

        // a normal's kept bits include the hidden 1 (2^10), so (e16 - 1) << 10 adds
        // the exponent; a carry out of the mantissa bumps it, up to infinity
        wire [14:0] base    = is_normal ? {5'(e16 - 11'sd1), 10'b0} : 15'b0;
        wire [14:0] rounded = base + kept + 15'(round_up);

        reg [15:0] half;
        always @(*) begin
            if (st2_nan[h]) begin
                half = FP16_QNAN;
            end else if (st2_inf[h] || (!st2_zero[h] && is_ovf)) begin
                half = {st2_sign[h], FP16_INF};
            end else if (st2_zero[h]) begin
                half = {st2_sign[h], 15'b0};
            end else begin
                half = {st2_sign[h], rounded};
            end
        end
        assign st2_result[16*h +: 16] = half;
    end

    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (LATENCY - 1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (st2_result),
        .data_out (result)
    );

endmodule
