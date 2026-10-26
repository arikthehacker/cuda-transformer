| GEMM | M x N x K | cuBLAS | mma.sync | wmma | simt |
|---|---|---|---|---|---|
| c_attn fwd | 4096 x 1152 x 384 | 26.8 TFLOPS | 21.7 TFLOPS, 81% | 15.0 TFLOPS, 56% | 3.4 TFLOPS, 13% |
| attn c_proj fwd | 4096 x 384 x 384 | 22.6 TFLOPS | 15.8 TFLOPS, 70% | 15.0 TFLOPS, 66% | 3.4 TFLOPS, 15% |
| c_fc fwd | 4096 x 1536 x 384 | 29.8 TFLOPS | 25.7 TFLOPS, 86% | 20.0 TFLOPS, 67% | 4.0 TFLOPS, 14% |
| mlp c_proj fwd | 4096 x 384 x 1536 | 33.3 TFLOPS | 22.2 TFLOPS, 67% | 18.2 TFLOPS, 54% | 3.8 TFLOPS, 11% |
| logits fwd | 4096 x 4160 x 384 | 35.7 TFLOPS | 28.9 TFLOPS, 81% | 18.8 TFLOPS, 53% | 4.1 TFLOPS, 11% |
| c_attn dinp | 4096 x 384 x 1152 | 31.6 TFLOPS | 24.4 TFLOPS, 77% | 14.0 TFLOPS, 44% | 4.0 TFLOPS, 13% |
| attn c_proj dinp | 4096 x 384 x 384 | 25.1 TFLOPS | 20.7 TFLOPS, 82% | 14.0 TFLOPS, 56% | 3.9 TFLOPS, 16% |
| c_fc dinp | 4096 x 384 x 1536 | 30.9 TFLOPS | 23.5 TFLOPS, 76% | 13.8 TFLOPS, 45% | 4.0 TFLOPS, 13% |
| mlp c_proj dinp | 4096 x 1536 x 384 | 31.7 TFLOPS | 27.2 TFLOPS, 86% | 16.7 TFLOPS, 53% | 4.2 TFLOPS, 13% |
| logits dinp | 4096 x 384 x 4160 | 37.6 TFLOPS | 24.0 TFLOPS, 64% | 14.0 TFLOPS, 37% | 4.0 TFLOPS, 11% |
| c_attn dW | 1152 x 384 x 4096 | 25.3 TFLOPS | 23.8 TFLOPS, 94% | 12.3 TFLOPS, 49% | 3.2 TFLOPS, 13% |
| attn c_proj dW | 384 x 384 x 4096 | 21.0 TFLOPS | 18.4 TFLOPS, 88% | 8.8 TFLOPS, 42% | 2.5 TFLOPS, 12% |
| c_fc dW | 1536 x 384 x 4096 | 32.3 TFLOPS | 26.8 TFLOPS, 83% | 15.9 TFLOPS, 49% | 3.3 TFLOPS, 10% |
| mlp c_proj dW | 384 x 1536 x 4096 | 29.6 TFLOPS | 27.0 TFLOPS, 91% | 16.1 TFLOPS, 55% | 3.3 TFLOPS, 11% |
| logits dW | 4160 x 384 x 4096 | 27.4 TFLOPS | 24.4 TFLOPS, 89% | 13.0 TFLOPS, 47% | 4.0 TFLOPS, 15% |
| one training step | | 10.08 ms | 12.43 ms (81%) | 19.75 ms (51%) | 82.11 ms (12%) |
