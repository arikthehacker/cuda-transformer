"""Every setting for the reference model, in one place.

Both configs are frozen, and both get saved inside every checkpoint and every
dump, so each number below is only ever written here.
"""
from __future__ import annotations

from dataclasses import dataclass

TOKENIZER_PATH = "data/tokenizer.bpe"
TRAIN_PATH = "data/train.bin"
VAL_PATH = "data/val.bin"
REF_DIR = "data/ref"


# Model sizes by name: (n_layer, n_head, n_embd, n_ctx). "small" is the default
# ModelConfig below (stages 1 to 5), "chat" is stage 6.
CONFIGS = {"small": (6, 6, 384, 256), "chat": (8, 8, 512, 512)}


def sizes(name: str) -> dict[str, int]:
    """The ModelConfig keyword arguments of a preset, for ModelConfig.for_vocab(V, **sizes(name))."""
    if name not in CONFIGS:
        raise ValueError(f"no model size {name!r} (have: {', '.join(CONFIGS)})")
    n_layer, n_head, n_embd, n_ctx = CONFIGS[name]
    return {"n_layer": n_layer, "n_head": n_head, "n_embd": n_embd, "n_ctx": n_ctx}


def pad_to(n: int, multiple: int = 64) -> int:
    """Smallest multiple of `multiple` that is >= n."""
    return ((n + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int            # real ids: 256 bytes + merges + tags, from tokenizer.bpe
    padded_vocab: int          # vocab_size rounded up to a multiple of 64
    n_ctx: int = 256           # longest sequence the model sees, in tokens
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384          # head size 384 / 6 = 64
    ln_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.padded_vocab < self.vocab_size or self.padded_vocab % 64 != 0:
            raise ValueError(f"padded_vocab {self.padded_vocab} must be a multiple of 64 and >= vocab_size {self.vocab_size}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd {self.n_embd} must divide evenly into n_head {self.n_head} heads")

    @property
    def head_size(self) -> int:
        """D = C / H."""
        return self.n_embd // self.n_head

    def same_size(self, other: ModelConfig) -> bool:
        """Every size equal. ln_eps is left out: weights.bin stores it as float32,
        so 1e-5 comes back as 9.99999975e-06 (the same number once it is float32)."""
        return (self.vocab_size, self.padded_vocab, self.n_ctx, self.n_layer, self.n_head, self.n_embd) == \
            (other.vocab_size, other.padded_vocab, other.n_ctx, other.n_layer, other.n_head, other.n_embd)

    @classmethod
    def for_vocab(cls, vocab_size: int, **sizes: int) -> ModelConfig:
        """Config for a tokenizer with `vocab_size` ids, padded_vocab filled in."""
        return cls(vocab_size=vocab_size, padded_vocab=pad_to(vocab_size), **sizes)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16       # 16 x 256 = 4,096 tokens per step
    max_steps: int = 5000      # about 7.7 passes over 2.66M training tokens
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    eval_every: int = 250      # steps
    eval_batches: int = 20
    seed: int = 1337
    patience: int = 4          # evals with no new best val loss before stopping, 0 = never stop early
    sample_every: int = 1000   # steps between printed samples, 0 = never
    sample_tokens: int = 200   # tokens per printed sample
    timing_skip: int = 10      # first steps of a run left out of tokens per second
    # GPU peaks for MFU, in TFLOPS. Each mode is measured against the hardware it
    # runs on: exact mode against the fp32 cores, fast mode against the fp16 tensor
    # cores (fp32 accumulate, which is what PyTorch uses for fp16 matmuls).
    peak_tflops_fp32: float = 10.0     # RTX 2080 fp32, about 10
    peak_tflops_fp16: float = 40.0     # RTX 2080 fp16 tensor cores, about 40


# --sft (reference/train.py, cuda/train_fast): fine-tuning a pretrained model
# on answers wants a smaller step than pretraining, with a short warmup.
SFT_TRAIN = {"lr": 1e-4, "min_lr": 1e-5, "warmup_steps": 100}


# The answer key model (reference/dump.py), also used by the tests.
DUMP_BATCH = 4                 # B for the dump; T is n_ctx


def tiny_model_config(vocab_size: int) -> ModelConfig:
    """L 2, H 4, C 64, n_ctx 64: small enough that a wrong number is easy to trace."""
    return ModelConfig.for_vocab(vocab_size, n_ctx=64, n_layer=2, n_head=4, n_embd=64)
