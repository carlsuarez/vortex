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

// DCMP.LD's weight LUT: VX_DCR_DCMP_LUT0..3, entry 4i+k in byte k of LUTi.
// It snoops the core's DCR requests and never answers them. The host writes it
// before a launch, so reset does not clear it, as in SimX.
module VX_dcmp_dcr import VX_gpu_pkg::*; (
    input wire                      clk,
    input wire                      dcr_req_valid,
    input dcr_req_t                 dcr_req_data,
    output wire [DCMP_LUT_BITS-1:0] lut
);
    reg [3:0][31:0] lut_r;

    wire is_lut_write = dcr_req_valid
                     && dcr_req_data.rw
                     && (dcr_req_data.addr >= VX_DCR_ADDR_WIDTH'(`VX_DCR_DCMP_LUT0))
                     && (dcr_req_data.addr <= VX_DCR_ADDR_WIDTH'(`VX_DCR_DCMP_LUT3));

    always @(posedge clk) begin
        if (is_lut_write) begin
            lut_r[dcr_req_data.addr[1:0]] <= dcr_req_data.data[31:0];
        end
    end

    assign lut = lut_r;

endmodule
