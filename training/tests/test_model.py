
import pytest
import torch

from model import AnuTransformer, ModelConfig, num_parameters


@pytest.fixture(scope="module")
def full_config():
    return ModelConfig()


def test_config_values(full_config):
    assert full_config.vocab_size == 12_000
    assert full_config.context_length == 512
    assert full_config.n_layer == 8
    assert full_config.n_embd == 512
    assert full_config.n_head == 8
    assert full_config.ffn_dim == 2_048
    assert full_config.head_dim == 64


def test_config_json_roundtrip(full_config, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(full_config.to_json())
    restored = ModelConfig.from_json_file(str(path))
    assert restored == full_config
    assert restored.to_dict() == full_config.to_dict()


def test_config_rejects_indivisible_head_dim():
    with pytest.raises(ValueError):
        ModelConfig(n_embd=100, n_head=8)


def test_forward_shape(full_config):
    torch.manual_seed(0)
    model = AnuTransformer(full_config)
    ids = torch.randint(0, full_config.vocab_size, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, full_config.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_parameter_count_matches_estimate(full_config):
    torch.manual_seed(0)
    model = AnuTransformer(full_config)
    actual = num_parameters(model)
    estimate = full_config.num_params_estimate
    assert actual == estimate
    assert 30_000_000 < actual < 33_000_000  # ~31M per Architecture §4.1


def test_weight_tying(full_config):
    torch.manual_seed(0)
    model = AnuTransformer(full_config)
    assert model.lm_head.weight is model.token_embedding.weight


def test_loss_dtype_and_finiteness(full_config):
    torch.manual_seed(0)
    model = AnuTransformer(full_config)
    ids = torch.randint(0, full_config.vocab_size, (2, 16))
    loss = model.compute_loss(ids, ids)
    assert loss.ndim == 0
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert loss.item() > 1.0  # random init: near ln(vocab) ~ 9.4


def test_causality(tiny_config):
    """Position j must not influence outputs at positions < j."""
    torch.manual_seed(0)
    model = AnuTransformer(tiny_config)
    model.eval()
    with torch.no_grad():
        base = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.context_length))
        changed = base.clone()
        changed[0, 8] = (changed[0, 8] + 1) % tiny_config.vocab_size
        logits_base = model(base)
        logits_changed = model(changed)
    assert torch.allclose(logits_base[0, :8], logits_changed[0, :8], atol=1e-5)
    assert not torch.allclose(logits_base[0, 8:], logits_changed[0, 8:], atol=1e-5)


def test_rope_rotation_properties():
    """Layer-level RoPE checks: pos 0 is identity; pairs rotate; dot products invariant."""
    from model import RotaryEmbedding

    rope = RotaryEmbedding(dim=64, max_seq_len=8)
    torch.manual_seed(0)
    x = torch.randn(1, 1, 8, 64)
    rotated = rope(x)

    # position 0: cos(0)=1, sin(0)=0 -> identity
    assert torch.allclose(rotated[0, 0, 0], x[0, 0, 0], atol=1e-6)
    # different positions rotate differently
    assert not torch.allclose(rotated[0, 0, 1], rotated[0, 0, 2], atol=1e-6)

    # rotation is length-preserving: ||rot(q)|| == ||q||
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)

    # same-position rotations preserve query-key dot products
    q, k = torch.randn(1, 1, 4, 64), torch.randn(1, 1, 4, 64)
    rq, rk = rope(q), rope(k)
    for i in range(4):
        before = (q[0, 0, i] * k[0, 0, i]).sum()
        after = (rq[0, 0, i] * rk[0, 0, i]).sum()
        assert torch.allclose(after, before, atol=1e-4)

    # cross-position dot products depend on relative position (this is what
    # makes attention position-aware): rotating k by a different offset changes it
    diff_before = (rq[0, 0, 2] * k[0, 0, 0]).sum()  # k at position 0: identity
    diff_after = (rq[0, 0, 2] * rk[0, 0, 1]).sum()
    assert not torch.allclose(diff_after, diff_before, atol=1e-3)


def test_overfit_single_batch(tiny_config):
    """Core Phase 2 gate: loss on a single repeated batch collapses to ~0."""
    torch.manual_seed(0)
    model = AnuTransformer(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)

    batch = torch.randint(0, tiny_config.vocab_size, (8, tiny_config.context_length))
    targets = torch.roll(batch, -1, dims=1)  # next-token prediction

    initial = model.compute_loss(batch, targets).item()
    assert initial > 3.0  # well above the log2 vocab floor, model not already converged

    for _ in range(300):
        optimizer.zero_grad()
        loss = model.compute_loss(batch, targets)
        loss.backward()
        optimizer.step()

    final = model.compute_loss(batch, targets).item()
    assert final < 0.05, f"overfit loss too high: {final}"
    assert final < initial / 10
