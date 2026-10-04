# GPT-OSS-120B one-layer decode results

Full attention; 2,048 cached tokens plus one new token. FP16 tensors, reproducible synthetic parameters, 1 GHz simulator clock.

The timed region includes all 14 stages, DMA, and synchronization. Direct ELF initialization, output dumping, and reporting are outside it. Latency is for the complete batch; all sequences decode one token concurrently.

| Batch | Cycles | Layer latency (ms) | Distinct experts | Output max absolute error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3,432,185 | 3.432185 | 4 | 0.00036621094 |
| 8 | 11,745,299 | 11.745299 | 32 | 0.00390625 |
| 64 | 60,598,176 | 60.598176 | 115 | 0.00390625 |

Stage times in microseconds at 1 GHz:

| Stage | Batch 1 | Batch 8 | Batch 64 |
| --- | ---: | ---: | ---: |
| norm1 | 222.187 | 222.474 | 887.974 |
| qkv | 146.224 | 153.551 | 246.236 |
| rope_cache | 286.617 | 287.472 | 1150.391 |
| gqa | 994.599 | 4001.633 | 31823.444 |
| out_projection | 112.194 | 114.799 | 154.740 |
| attention_residual | 1.216 | 1.597 | 6.472 |
| norm2 | 222.478 | 222.802 | 889.345 |
| router_gemm | 26.211 | 26.786 | 55.216 |
| topk | 13.390 | 91.956 | 720.754 |
| dispatch | 0.704 | 1.903 | 12.512 |
| expert_up | 464.086 | 3617.464 | 11457.447 |
| swiglu | 295.531 | 592.347 | 4728.336 |
| expert_down | 271.435 | 2034.652 | 6953.228 |
| combine | 375.313 | 375.863 | 1512.081 |

Common decoder ELF SHA-256: `dcf1bd1922f3bbf0125ffda176a1207d3e4663163de75ed6787371b599996f32`.

Shared sequences produce byte-identical simulator outputs across these batch sizes.

All 128 experts are preloaded; the packed weight payload is 6,903,704,960 bytes. The synthetic router uses distinct, reproducible top-k choices. Latency depends on expert reuse and is not a prediction for a trained checkpoint.

This baseline uses one controlling scalar/vector core and one RedMule per cluster across 16 clusters. It does not exploit all four vector cores per cluster. FP32 reductions and FP32 accelerator accumulation are narrowed to FP16 at tensor/tile boundaries.

The complete run artifacts are in `build/gpt_oss_decode/full` in the evaluation GVSOC checkout. `results/measurements.json` archives the timing and validation records; `SETUP.md` records the simulator revisions and rebuild instructions.
