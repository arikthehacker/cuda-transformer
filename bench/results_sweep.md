Every run: 12,200 steps of 16 x 512 tokens (100M tokens), the same batches and init, warmup 2,000, WSD with the last 20% decaying, min lr = lr / 10.

| lr | beta2 | final val loss | final val bpb | bpb vs baseline | first train loss | skipped steps |
|---|---|---|---|---|---|---|
| 6e-4 | 0.95 | 3.8869 | 1.3045 | +0.0000 (baseline) | 9.826203 | 1 |
| 6e-4 | 0.99 | 3.8192 | 1.2818 | -0.0227 | 9.826203 | 0 |
| 2e-3 | 0.95 | 3.9004 | 1.3090 | +0.0045 | 9.826203 | 1 |
| 2e-3 | 0.99 | 3.8161 | 1.2807 **best** | -0.0238 | 9.826203 | 1 |
| 6e-3 | 0.95 | 4.0100 | 1.3458 | +0.0413 | 9.826203 | 2 |
| 6e-3 | 0.99 | 3.8847 | 1.3037 | -0.0008 tie | 9.826203 | 0 |
| 6e-4 | 0.95 | 3.8896 | 1.3054 | +0.0009 (A/A: the baseline again) | 9.826203 | 1 |

Noise floor (the two baseline runs): 0.0009 bpb. A difference under about twice that is a tie.
Every run's first step had the same train loss, so the batches and the init match.
Best: lr 2e-3, beta2 0.99.
