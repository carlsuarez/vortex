Throughput in weights per cycle (higher is better); best per bandwidth in bold.

| format | engine | instrs/weight | 64 B/c | 6.4 B/c | 3.2 B/c | 1.6 B/c | 0.8 B/c | 0.4 B/c |
|---|---|---|---|---|---|---|---|---|
| q8_0 | scalar | 1.40 | 2.38 | 2.08 | 1.68 | 0.95 | 0.50 | 0.25 |
| q4_0 | scalar | 1.80 | 1.90 | 1.82 | 1.70 | 1.37 | 0.80 | 0.41 |
| iq4_nl | scalar | 2.05 | 1.67 | 1.60 | 1.49 | 1.25 | 0.79 | 0.42 |
| mxfp4 | scalar | 1.94 | 1.78 | 1.71 | 1.61 | 1.35 | 0.85 | 0.45 |
| q4_k | scalar | 1.53 | 2.23 | 2.11 | 1.89 | 1.41 | 0.80 | 0.39 |
| q3_k | scalar | 3.54 | 0.98 | 0.96 | 0.94 | 0.88 | 0.76 | **0.52** |
| q8_0 | tcu | 2.44 | 1.06 | 1.04 | 0.98 | 0.79 | 0.47 | 0.25 |
| q4_0 | tcu | 2.81 | 0.93 | 0.92 | 0.90 | 0.84 | 0.66 | 0.41 |
| iq4_nl | tcu | 3.15 | 0.84 | 0.83 | 0.80 | 0.75 | 0.61 | 0.40 |
| mxfp4 | tcu | 3.26 | 0.82 | 0.81 | 0.79 | 0.74 | 0.62 | 0.42 |
| q4_0 | dcmp | 0.85 | 3.54 | **3.21** | **2.65** | 1.60 | 0.89 | 0.44 |
| iq4_nl | dcmp | 0.85 | **3.54** | 3.19 | 2.58 | 1.65 | 0.86 | 0.46 |
| mxfp4 | dcmp | 0.92 | 3.41 | 3.09 | 2.63 | **1.75** | **0.94** | 0.49 |

Fastest kernel at each bandwidth, and its measured DRAM traffic as a share of peak:

| peak B/cycle | fastest | weights/cycle | DRAM use |
|---|---|---|---|
| 64 | iq4_nl dcmp | 3.54 | 3% |
| 6.4 | q4_0 dcmp | 3.21 | 29% |
| 3.2 | q4_0 dcmp | 2.65 | 48% |
| 1.6 | mxfp4 dcmp | 1.75 | 60% |
| 0.8 | mxfp4 dcmp | 0.94 | 64% |
| 0.4 | q3_k scalar | 0.52 | 58% |
