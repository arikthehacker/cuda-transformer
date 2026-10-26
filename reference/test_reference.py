"""Sanity checks for the reference model. From the repo root:

    python -m pytest reference

Uses data/tokenizer.bpe and data/val.bin when they exist, random tokens otherwise.
Runs on the GPU when there is one.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")    # before CUDA starts

import copy
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from . import dump
from .config import DUMP_BATCH, TOKENIZER_PATH, VAL_PATH, ModelConfig, TrainConfig, tiny_model_config
from .data import Batches, fixed_batches, load_tokens
from .model import GPT
from .tokenizer import load_bpe
from .train import exact_mode, lr_at, make_scaler, train_step

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FALLBACK_VOCAB = 4100


def vocab_size() -> int:
    if os.path.exists(TOKENIZER_PATH):
        return len(load_bpe(TOKENIZER_PATH)[0])
    return FALLBACK_VOCAB


def tokens() -> np.ndarray:
    """Real val tokens, or 100,000 random ones."""
    if os.path.exists(VAL_PATH):
        return load_tokens(VAL_PATH)
    rng = np.random.default_rng(0)
    return rng.integers(0, vocab_size(), size=100_000).astype(np.uint16)


def one_batch(cfg: ModelConfig, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    return fixed_batches(tokens(), batch_size, cfg.n_ctx, 1, DEVICE)[0]        # (B, T), (B, T)


def fresh(cfg: ModelConfig, seed: int = 0) -> GPT:
    exact_mode(seed)
    return GPT(cfg).to(DEVICE)


def test_param_count():
    model = GPT(ModelConfig.for_vocab(4100))
    assert model.num_params() == 6 * 1_774_464 + 1_597_440 + 98_304 + 768


def test_initial_loss():
    cfg = ModelConfig.for_vocab(vocab_size())
    model = fresh(cfg)
    x, y = one_batch(cfg, 8)
    with torch.no_grad():
        _, loss = model(x, y)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.1, f"loss {loss.item():.3f}, ln(V) {math.log(cfg.vocab_size):.3f}"


def test_causal():
    cfg = tiny_model_config(vocab_size())
    model = fresh(cfg)
    x, _ = one_batch(cfg, 2)
    t = 20
    x2 = x.clone()
    x2[:, t] = (x2[:, t] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x2)
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6, rtol=0), "a later token changed an earlier logit"
    assert not torch.allclose(a[:, t:], b[:, t:], atol=1e-6, rtol=0), "the changed token had no effect at all"


def test_overfit_one_batch():
    cfg = ModelConfig.for_vocab(vocab_size(), n_ctx=64, n_layer=2, n_head=4, n_embd=128)
    model = fresh(cfg)
    tc = TrainConfig(lr=3e-3, weight_decay=0.0)
    opt = model.optimizer(tc)
    scaler = make_scaler(False)
    x, y = one_batch(cfg, 4)
    for _ in range(200):
        loss, _ = train_step(model, opt, scaler, x, y, tc.lr, tc.grad_clip)
    assert loss.item() < 0.1, f"loss after 200 steps on one batch: {loss.item():.3f}"


def test_padded_vocab():
    cfg = tiny_model_config(vocab_size())
    V = cfg.vocab_size
    model = fresh(cfg)
    x, y = one_batch(cfg, 2)
    assert torch.all(model.wte[V:] == 0), "padded rows should start at 0"

    _, loss = model(x, y)
    loss.backward()
    assert torch.all(model.wte.grad[V:] == 0), "padded rows got a gradient"

    with torch.no_grad():
        model.wte[V:] = torch.randn_like(model.wte[V:])       # padded logits change now
        _, loss2 = model(x, y)
    assert abs(loss2.item() - loss.item()) < 1e-6, "padded logits changed the loss"

    model = fresh(cfg)                                        # one real step keeps them at 0
    opt = model.optimizer(TrainConfig())
    train_step(model, opt, make_scaler(False), x, y, 6e-4, 1.0)
    assert torch.all(model.wte[V:] == 0), "an optimizer step moved the padded rows"


def test_tied_grad():
    cfg = tiny_model_config(vocab_size())
    V = cfg.vocab_size
    model = fresh(cfg)
    x, y = one_batch(cfg, 2)
    _, loss = model(x, y)
    loss.backward()

    untied = copy.deepcopy(model)
    untied.zero_grad(set_to_none=True)
    head = torch.nn.Parameter(untied.wte.detach().clone())   # separate output layer, same values
    logits = F.linear(untied.hidden(x), head)                # (B, T, Vp)
    F.cross_entropy(logits[..., :V].reshape(-1, V), y.reshape(-1)).backward()

    lookup, output = untied.wte.grad, head.grad
    assert lookup.abs().sum() > 0 and output.abs().sum() > 0
    assert torch.allclose(model.wte.grad, lookup + output, atol=1e-6, rtol=1e-5), \
        "wte.grad is not lookup part + output part"


def test_gradcheck():
    cfg = ModelConfig(vocab_size=50, padded_vocab=64, n_ctx=8, n_layer=1, n_head=2, n_embd=8)
    torch.manual_seed(0)
    model = GPT(cfg).double()                                # CPU, float64
    names = [n for n, _ in model.named_parameters()]
    params = tuple(p.detach().clone().requires_grad_(True) for p in model.parameters())
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (2, 8), generator=g)
    y = torch.randint(0, cfg.vocab_size, (2, 8), generator=g)

    def loss_of(*ps: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(model, dict(zip(names, ps)), (x, y))[1]

    assert torch.autograd.gradcheck(loss_of, params, eps=1e-6, atol=1e-5, rtol=1e-3)


def test_repeatable():
    def run() -> list[float]:
        cfg = tiny_model_config(vocab_size())
        tc = TrainConfig(batch_size=4)
        model = fresh(cfg, tc.seed)
        opt = model.optimizer(tc)
        scaler = make_scaler(False)
        batches = Batches(tokens(), tc.batch_size, cfg.n_ctx, DEVICE, tc.seed)
        losses = []
        for step in range(20):
            x, y = batches.next()
            loss, _ = train_step(model, opt, scaler, x, y, lr_at(step, tc), tc.grad_clip)
            losses.append(loss.item())
        return losses

    assert run() == run()


def test_exact_vs_fast():
    cfg = tiny_model_config(vocab_size())
    model = fresh(cfg)
    x, _ = one_batch(cfg, 4)
    with torch.no_grad():
        exact, _ = model(x)
        model.fast = True
        fast, _ = model(x)
        model.fast = False
    diff = (exact - fast.float()).abs().max().item()
    assert diff < 1e-2, f"fast logits are {diff:.2e} away from exact"


def test_dump_round_trip(tmp_path):
    cfg = tiny_model_config(vocab_size())
    header, records = dump.build_records(cfg, TrainConfig(), tokens(), DUMP_BATCH, "cpu")
    path = str(tmp_path / "debug.bin")
    dump.write_dump(path, header, records)
    header2, back = dump.read_dump(path)

    for k in dump.HEADER_INTS:
        assert header2[k] == header[k], k
    for k in dump.HEADER_FLOATS:
        assert header2[k] == np.float32(header[k]), k
    assert list(back) == list(records), "records came back in a different order"
    for name, t in records.items():
        a = t.cpu().numpy()
        assert a.dtype == back[name].dtype and a.shape == back[name].shape, name
        assert np.array_equal(a, back[name]), name

    # the names and order the stage 3 reader expects
    params = [n for n, _ in GPT(cfg).named_parameters()]
    layer = ["ln1", "ln1.mean", "ln1.rstd", "qkv", "scores", "probs", "atty", "attn_out", "res1",
             "ln2", "ln2.mean", "ln2.rstd", "fc", "gelu", "mlp_out", "res2"]
    forward = ["emb"] + [f"h.{l}.{n}" for l in range(cfg.n_layer) for n in layer] + \
              ["ln_f", "ln_f.mean", "ln_f.rstd", "logits", "loss"]
    expected = (["config.ln_eps"] + params + ["tokens", "targets"] + forward
                + ["grad." + n for n in params + forward] + ["grad_norm"] + ["step1." + n for n in params])
    assert list(records) == expected
    assert params[:4] == ["wte", "wpe", "h.0.ln1.w", "h.0.ln1.b"] and params[-2:] == ["ln_f.w", "ln_f.b"]


def test_lr_schedule():
    tc = TrainConfig()
    assert lr_at(0, tc) == pytest.approx(tc.lr / tc.warmup_steps)
    assert lr_at(tc.warmup_steps - 1, tc) == pytest.approx(tc.lr)
    assert lr_at(tc.warmup_steps, tc) == pytest.approx(tc.lr)
    assert lr_at(tc.max_steps, tc) == pytest.approx(tc.min_lr)
    middle = (tc.warmup_steps + tc.max_steps) // 2
    assert lr_at(middle, tc) == pytest.approx((tc.lr + tc.min_lr) / 2, rel=1e-3)


def test_load_tokens_checks(tmp_path):
    good = tmp_path / "good.bin"
    ids = np.array([1, 2, 65535], dtype="<u2")
    good.write_bytes(np.array([0x31455042, 1, 3], dtype="<u4").tobytes() + ids.tobytes())
    assert load_tokens(str(good)).tolist() == [1, 2, 65535]

    short = tmp_path / "short.bin"
    short.write_bytes(np.array([0x31455042, 1, 4], dtype="<u4").tobytes() + ids.tobytes())
    with pytest.raises(ValueError, match="header says 4 tokens"):
        load_tokens(str(short))
