| run | steps | tokens | val at start | best val | final val | final val bpb | tok/s | GPU hours | counted per step |
|---|---|---|---|---|---|---|---|---|---|
| pretrain (FineWeb-Edu) | 122,000 | 999M | 9.826 | 3.443 (step 122,000) | 3.443 | 1.1554 | 91,408 | 3.04 | 8,192 of 8,192 |
| SFT (smol-smoltalk, answers only) | 26,000 | 213M | 3.384 | 1.757 (step 26,000) | 1.757 | 0.5799 | 91,222 | 0.65 | 4,798 of 8,192 |

The SFT val loss counts only the answer tokens of smol-smoltalk's test split, so it is not on the same scale as the pretraining val loss. Bits per byte leave the tags out and do not depend on the tokenizer.
