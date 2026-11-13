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

// Functional test for VX_dcmp_decode: every result must match, bit for bit, what
// SimX's DCMP.LD computes -- an fp32 product rounded to fp16 by softfloat -- over
// random LUTs, bytes and scales, plus scales at fp16's subnormal, overflow and
// special-value edges.

#include "VVX_dcmp_decode.h"
#include "vl_simulator.h"
#include "rvfloats.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <random>
#include <vector>

#ifndef LATENCY
#define LATENCY 2
#endif

using Device = VVX_dcmp_decode;

namespace {

struct TV { int8_t lut[16]; uint8_t data; uint32_t scale; };

const uint32_t kScales[] = {
    0x00000000, 0x80000000, 0x00000001, 0x007fffff, 0x00800000, 0x3f800000, 0xbf800000,
    0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0x7f7fffff, 0x477fe000, 0x477ff000,
    0x47800000, 0x43ff0000, 0x38800000, 0x33800000, 0x33000000, 0x33000001, 0x32ffffff,
    0x2f800000, 0x3c000000, 0x3d000000, 0x3e800000,
};

uint32_t golden(const TV& t) {
    uint32_t out = 0;
    for (int h = 0; h < 2; ++h) {
        float s;
        std::memcpy(&s, &t.scale, 4);
        float v = float(t.lut[(t.data >> (4 * h)) & 0xf]) * s;
        uint32_t vb;
        std::memcpy(&vb, &v, 4);
        uint32_t ff = 0;
        out |= uint32_t(rv_ftoh_s(vb, 0, &ff)) << (16 * h);
    }
    return out;
}

void drive(Device& dut, const TV& t) {
    for (int w = 0; w < 4; ++w) {
        uint32_t word = 0;
        for (int j = 0; j < 4; ++j)
            word |= uint32_t(uint8_t(t.lut[4 * w + j])) << (8 * j);
        dut.lut[w] = word;
    }
    dut.data  = t.data;
    dut.scale = t.scale;
}

} // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    size_t num_random = 200000;
    if (argc > 1) {
        long v = std::strtol(argv[1], nullptr, 10);
        if (v > 0) num_random = static_cast<size_t>(v);
    }

    std::mt19937 rng(0xDC3Bu);
    auto random_lut = [&](TV& t) {
        for (auto& e : t.lut) e = int8_t(rng());
        if (rng() % 8 == 0) t.lut[rng() % 16] = -128;
        if (rng() % 8 == 0) t.lut[rng() % 16] = 0;
    };

    std::vector<TV> tests;
    for (uint32_t s : kScales) {
        for (int i = 0; i < 64; ++i) {
            TV t;
            random_lut(t);
            t.data = uint8_t(rng());
            t.scale = s ^ ((i & 1) ? 0x80000000u : 0);
            tests.push_back(t);
        }
    }
    for (size_t i = 0; i < num_random; ++i) {
        TV t;
        random_lut(t);
        t.data = uint8_t(rng());
        uint32_t s = uint32_t(rng());
        if (i % 2 == 0) {
            // exponents around fp16's range: subnormal results through overflow
            s = (s & 0x807fffffu) | ((90u + (s >> 24) % 60u) << 23);
        }
        t.scale = s;
        tests.push_back(t);
    }

    const size_t N = tests.size();
    vl_simulator<Device> sim;
    uint64_t ticks = 0;
    sim->enable = 1;
    for (int w = 0; w < 4; ++w) sim->lut[w] = 0;
    sim->data = 0;
    sim->scale = 0;
    ticks = sim.reset(ticks);

    std::deque<std::pair<uint32_t, size_t>> fifo;
    size_t feed = 0, checked = 0, errors = 0;
    for (size_t c = 0; c < N + LATENCY; ++c) {
        if (feed < N) {
            drive(*sim.operator->(), tests[feed]);
            fifo.push_back({golden(tests[feed]), feed});
            ++feed;
        }
        ticks = sim.step(ticks, 2);
        if (c >= (size_t)(LATENCY - 1) && !fifo.empty()) {
            auto g = fifo.front();
            fifo.pop_front();
            uint32_t got = sim->result;
            if (got != g.first) {
                const TV& t = tests[g.second];
                if (errors < 24)
                    printf("MISMATCH data=%02x scale=%08x lut[lo]=%d lut[hi]=%d : got %08x exp %08x\n",
                           t.data, t.scale, t.lut[t.data & 0xf], t.lut[t.data >> 4], got, g.first);
                ++errors;
            }
            ++checked;
        }
    }

    printf("dcmp_decode: checked=%zu errors=%zu\n%s\n", checked, errors, errors ? "FAILED" : "PASSED");
    return errors ? 1 : 0;
}
