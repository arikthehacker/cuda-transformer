| run | tok/s | MFU | peak |
|---|---|---|---|
| PyTorch exact (fp32) | 55,180 | 40.9% | 10 TFLOPS |
| PyTorch fast (fp16 autocast + SDPA) | 181,885 | 33.7% | 40 TFLOPS |
| stage 3 (fp32 CUDA) | 34,055 | 25.2% | 10 TFLOPS |
| stage 4 fp32 build (simt GEMMs, stage 3 attention, graph) | 38,550 | 28.6% | 10 TFLOPS |
| fp16 + loss scaling, simt GEMMs | 39,087 | 7.2% | 40 TFLOPS |
| + wmma tensor core GEMMs | 94,564 | 17.5% | 40 TFLOPS |
| + mma.sync + ldmatrix GEMMs | 112,718 | 20.9% | 40 TFLOPS |
| + flash attention | 193,578 | 35.8% | 40 TFLOPS |
| + CUDA Graph (all of stage 4) | 194,250 | 36.0% | 40 TFLOPS |

300 steps per run, measured 2026-10-26 on NVIDIA GeForce RTX 2080.
The fused small kernels (residual + LayerNorm, in place cross entropy, one pass unscale + norm,
one launch AdamW, GELU recompute) are part of every train_fast row.
