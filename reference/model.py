"""The reference GPT (GPT-2 layout, shrunk). It is the answer key for the CUDA port.

Layout rules (the CUDA code follows these exactly):
  Linear stores w as (out, in) and computes y = x @ w.T + b, same as nn.Linear.
  c_attn output columns are [q | k | v], each C wide. Head h of q owns columns
    h*D .. h*D + D - 1, and the same goes for k and v inside their own C block.
  wte is (Vp, C) and wpe is (n_ctx, C). The output layer reuses wte:
    logits = x @ wte.T. Rows V .. Vp-1 of wte are padding and stay 0.
  All tensors are row-major float32 in exact mode.

Names: every parameter is named the way the dump and the CUDA buffers name it,
  "wte", "wpe", "h.{l}.ln1.w", "h.{l}.attn.c_attn.b", "h.{l}.mlp.c_proj.w",
  "ln_f.w", ... That is why Linear and LayerNorm are small modules here with
  parameters called w and b (nn.Linear would call them weight and bias).

Two modes:
  exact  float32, LayerNorm and attention written out by hand, every
         intermediate can be recorded. The answer key.
  fast   autocast float16 (bfloat16 on CPU), F.layer_norm and
         F.scaled_dot_product_attention. The baseline stage 4 races.

Shape letters: B batch, T tokens, C n_embd, H n_head, D head size,
V vocab_size, Vp padded_vocab, L n_layer.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig


class Recorder:
    """Collects named intermediate values during one forward pass.
    record(name, t) stores t and calls t.retain_grad(), so after backward
    every recorded value also has its gradient in t.grad."""

    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}

    def record(self, name: str, t: torch.Tensor) -> torch.Tensor:
        if name in self.values:
            raise KeyError(f"{name} recorded twice")
        if t.requires_grad:
            t.retain_grad()
        self.values[name] = t
        return t


def _rec(rec: Recorder | None, name: str, t: torch.Tensor) -> torch.Tensor:
    return t if rec is None else rec.record(name, t)


def _fast_dtype(device: torch.device) -> torch.dtype:
    """float16 on the GPU. CPU autocast only does bfloat16 well."""
    return torch.float16 if device.type == "cuda" else torch.bfloat16


class Linear(nn.Module):
    """Same layout and math as nn.Linear: w is (out, in), y = x @ w.T + b."""

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_out, n_in))     # (out, in)
        self.b = nn.Parameter(torch.zeros(n_out))           # (out,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.w, self.b)


class LayerNorm(nn.Module):
    """Biased variance (divide by C), eps inside the square root.
    Exact mode records out, mean and rstd = 1 / sqrt(var + eps), because the
    CUDA backward pass needs mean and rstd."""

    def __init__(self, n: int, eps: float) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(n))                # (C,)
        self.b = nn.Parameter(torch.zeros(n))               # (C,)
        self.eps = eps

    def forward(self, x: torch.Tensor, rec: Recorder | None = None, name: str = "",
                fast: bool = False) -> torch.Tensor:
        if fast:
            return F.layer_norm(x, (x.shape[-1],), self.w, self.b, self.eps)
        mean = x.mean(dim=-1)                               # (B, T)
        xc = x - mean.unsqueeze(-1)                         # (B, T, C)
        var = (xc * xc).mean(dim=-1)                        # (B, T)
        rstd = torch.rsqrt(var + self.eps)                  # (B, T)
        out = xc * rstd.unsqueeze(-1) * self.w + self.b     # (B, T, C)
        if rec is not None:
            rec.record(name, out)
            rec.record(name + ".mean", mean)
            rec.record(name + ".rstd", rstd)
        return out


class CausalSelfAttention(nn.Module):
    """c_attn: Linear(C, 3C), c_proj: Linear(C, C)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.c_attn = Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = Linear(cfg.n_embd, cfg.n_embd)
        mask = torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool))   # (n_ctx, n_ctx), True = may look
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, rec: Recorder | None = None, prefix: str = "",
                fast: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        H = self.n_head
        D = C // H

        qkv = _rec(rec, prefix + "qkv", self.c_attn(x))     # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)                       # (B, T, C) each, columns [q | k | v]
        q = q.view(B, T, H, D).transpose(1, 2)              # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)              # (B, H, T, D)
        v = v.view(B, T, H, D).transpose(1, 2)              # (B, H, T, D)

        if fast:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # (B, H, T, D)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)                    # (B, H, T, T)
            scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))      # (B, H, T, T) future = -inf
            _rec(rec, prefix + "scores", scores)
            probs = torch.softmax(scores, dim=-1)           # (B, H, T, T) subtracts each row's max first
            _rec(rec, prefix + "probs", probs)
            y = probs @ v                                   # (B, H, T, D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)    # (B, T, C) heads joined back
        _rec(rec, prefix + "atty", y)
        return _rec(rec, prefix + "attn_out", self.c_proj(y))   # (B, T, C)


class MLP(nn.Module):
    """c_fc: Linear(C, 4C), GELU tanh version, c_proj: Linear(4C, C)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.c_fc = Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor, rec: Recorder | None = None, prefix: str = "") -> torch.Tensor:
        fc = _rec(rec, prefix + "fc", self.c_fc(x))         # (B, T, 4C)
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        gelu = _rec(rec, prefix + "gelu", F.gelu(fc, approximate="tanh"))       # (B, T, 4C)
        return _rec(rec, prefix + "mlp_out", self.c_proj(gelu))                 # (B, T, C)


class Block(nn.Module):
    """x = x + attn(ln1(x)), then x = x + mlp(ln2(x))."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, cfg.ln_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, cfg.ln_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, rec: Recorder | None = None, prefix: str = "",
                fast: bool = False) -> torch.Tensor:
        ln1 = self.ln1(x, rec, prefix + "ln1", fast)                            # (B, T, C)
        res1 = _rec(rec, prefix + "res1", x + self.attn(ln1, rec, prefix, fast))   # (B, T, C)
        ln2 = self.ln2(res1, rec, prefix + "ln2", fast)                         # (B, T, C)
        return _rec(rec, prefix + "res2", res1 + self.mlp(ln2, rec, prefix))    # (B, T, C)


class GPT(nn.Module):
    """tokens (B, T) -> logits (B, T, Vp), and the loss when targets are given.

    Recorded names, in order: emb, then for each layer h.{l}.ln1, .ln1.mean,
    .ln1.rstd, .qkv, .scores, .probs, .atty, .attn_out, .res1, .ln2, .ln2.mean,
    .ln2.rstd, .fc, .gelu, .mlp_out, .res2, then ln_f, ln_f.mean, ln_f.rstd,
    logits, loss. Recording only works in exact mode."""

    def __init__(self, cfg: ModelConfig, fast: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.fast = fast
        self.wte = nn.Parameter(torch.empty(cfg.padded_vocab, cfg.n_embd))     # (Vp, C)
        self.wpe = nn.Parameter(torch.empty(cfg.n_ctx, cfg.n_embd))            # (n_ctx, C)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, cfg.ln_eps)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        """Linear and embedding weights normal(0, 0.02), biases 0, every c_proj
        weight normal(0, 0.02 / sqrt(2 L)), LayerNorm weight 1 and bias 0,
        padded wte rows 0. Draws happen in parameter order, so the same seed
        gives the same weights. Build on the CPU, then move, and every device
        starts from identical weights."""
        std = 0.02
        proj_std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        nn.init.normal_(self.wte, 0.0, std)
        nn.init.normal_(self.wpe, 0.0, std)
        for block in self.h:
            for lin, s in ((block.attn.c_attn, std), (block.attn.c_proj, proj_std),
                           (block.mlp.c_fc, std), (block.mlp.c_proj, proj_std)):
                nn.init.normal_(lin.w, 0.0, s)
                nn.init.zeros_(lin.b)
            for ln in (block.ln1, block.ln2):
                nn.init.ones_(ln.w)
                nn.init.zeros_(ln.b)
        nn.init.ones_(self.ln_f.w)
        nn.init.zeros_(self.ln_f.b)
        self.wte[self.cfg.vocab_size:].zero_()

    def hidden(self, idx: torch.Tensor, rec: Recorder | None = None) -> torch.Tensor:
        """Everything before the output layer: tokens (B, T) -> ln_f output (B, T, C)."""
        B, T = idx.shape
        if T > self.cfg.n_ctx:
            raise ValueError(f"sequence of {T} tokens is longer than n_ctx {self.cfg.n_ctx}")
        emb = F.embedding(idx, self.wte) + self.wpe[:T]                         # (B, T, C)
        x = _rec(rec, "emb", emb)
        for l, block in enumerate(self.h):
            x = block(x, rec, f"h.{l}.", self.fast)                             # (B, T, C)
        return self.ln_f(x, rec, "ln_f", self.fast)                             # (B, T, C)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                rec: Recorder | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (logits (B, T, Vp), loss or None). The loss is the mean over
        all B*T tokens and only looks at the V real columns."""
        V = self.cfg.vocab_size
        if rec is not None and self.fast:
            raise ValueError("recording needs exact mode: fast mode has no scores, probs or LayerNorm stats")
        with torch.autocast(device_type=idx.device.type, dtype=_fast_dtype(idx.device), enabled=self.fast):
            x = self.hidden(idx, rec)                                           # (B, T, C)
            logits = _rec(rec, "logits", F.linear(x, self.wte))                 # (B, T, Vp) tied: x @ wte.T
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits[..., :V].reshape(-1, V), targets.reshape(-1))   # scalar
                loss = _rec(rec, "loss", loss)
        return logits, loss

    def num_params(self) -> int:
        """Every parameter once (wte counts once even though it is used twice)."""
        return sum(p.numel() for p in self.parameters())

    def optimizer(self, tc: TrainConfig) -> torch.optim.AdamW:
        """AdamW with weight decay on every 2D weight (Linear w, wte, wpe) and
        none on biases or LayerNorm. Per element, with step t starting at 1:
            p = p - lr * wd * p                       (decay first)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            p = p - lr * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)
        Fast mode on the GPU uses the fused kernel, exact mode the plain loop."""
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": tc.weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        on_gpu = self.wte.device.type == "cuda"
        extra = {"fused": True} if (self.fast and on_gpu) else {"foreach": False}
        return torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2), eps=tc.eps, **extra)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, temperature: float, top_k: int,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Append max_new sampled tokens to idx (B, t). Crops to the last n_ctx
        tokens and never samples a padded id (>= V). temperature 0 = greedy."""
        V = self.cfg.vocab_size
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.cfg.n_ctx:])                          # (B, t, Vp)
            logits = logits[:, -1, :V].float()                                  # (B, V) padded ids cut off
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)                       # (B, 1)
            else:
                logits = logits / temperature
                if top_k > 0:
                    kth = torch.topk(logits, min(top_k, V), dim=-1).values[:, -1:]   # (B, 1)
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)                           # (B, V)
                nxt = torch.multinomial(probs, 1, generator=generator)          # (B, 1)
            idx = torch.cat([idx, nxt], dim=1)                                  # (B, t + 1)
        return idx
