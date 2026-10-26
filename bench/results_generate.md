| generation (fp16, 500 new tokens, temperature 0.8, top_k 40) | tok/s | ms |
|---|---|---|
| KV cache + CUDA Graph | 3,259 | 153.4 |
| KV cache, plain launches | 2,981 | 167.7 |
| no cache (full forward per token) | 734 | 681.1 |
