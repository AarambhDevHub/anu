
import pytest
import torch

from model import AnuTransformer, ModelConfig
from train import (
    build_lr_schedule,
    build_optimizer,
    compute_fast_loss,
    find_latest_checkpoint,
    generate,
    load_checkpoint,
    run_training,
    save_checkpoint,
)


def test_fast_loss_matches_standard_ce(tiny_config):
    torch.manual_seed(0)
    model = AnuTransformer(tiny_config)
    model.eval()
    x = torch.randint(0, tiny_config.vocab_size, (4, 16))
    y = torch.roll(x, -1, dims=1)
    standard = model.compute_loss(x, y)
    fast = compute_fast_loss(model, x, y)
    assert torch.allclose(fast, standard, atol=1e-2)
    assert fast.requires_grad  # must stay differentiable


def test_sdpa_matches_manual_attention(tiny_config):
    torch.manual_seed(0)
    manual = AnuTransformer(tiny_config)
    sdpa = AnuTransformer(tiny_config, use_sdpa=True)
    sdpa.load_state_dict(manual.state_dict())
    manual.eval()
    sdpa.eval()
    x = torch.randint(0, tiny_config.vocab_size, (2, 16))
    with torch.no_grad():
        a = manual(x)
        b = sdpa(x)
    assert torch.allclose(a, b, atol=1e-3)


def test_lr_schedule_warmup_and_decay():
    lr_at = build_lr_schedule(total_steps=1000, warmup_steps=100, max_lr=1e-3, min_lr=1e-5)
    assert lr_at(0) == pytest.approx(1e-3 / 100)  # linear warmup start
    assert lr_at(99) == pytest.approx(1e-3, rel=1e-6)  # warmup end
    assert lr_at(100) == pytest.approx(1e-3, rel=1e-6)  # decay starts
    assert lr_at(550) == pytest.approx((1e-3 + 1e-5) / 2, rel=1e-3)  # cosine midpoint
    assert lr_at(999) == pytest.approx(1e-5, rel=1e-2)  # near decay end
    assert lr_at(100_000) == pytest.approx(1e-5, rel=1e-2)  # clamped past end


def test_optimizer_decay_split(tiny_config):
    torch.manual_seed(0)
    model = AnuTransformer(tiny_config)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    # norms (1-D) in no-decay group
    norm_params = {id(p) for p in model.final_rmsnorm.parameters()}
    no_decay_ids = {id(p) for p in no_decay_group["params"]}
    assert norm_params <= no_decay_ids
    # tied embedding/head is 2-D -> decay group, counted once
    assert id(model.token_embedding.weight) in {id(p) for p in decay_group["params"]}


def test_checkpoint_roundtrip_and_resume(tmp_path, tiny_config):
    torch.manual_seed(0)
    model = AnuTransformer(tiny_config)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=build_lr_schedule(1000, 100, 1e-3, 1e-5)
    )
    optimizer.step()
    scheduler.step()
    path = save_checkpoint(tmp_path, step=42, model=model, optimizer=optimizer,
                           lr_scheduler=scheduler, metrics=[{"step": 42, "train_loss": 1.0}])

    assert find_latest_checkpoint(tmp_path) == path

    restored = AnuTransformer(tiny_config)
    restored_opt = build_optimizer(restored, lr=1e-3, weight_decay=0.1)
    restored_sched = torch.optim.lr_scheduler.LambdaLR(
        restored_opt, lr_lambda=build_lr_schedule(1000, 100, 1e-3, 1e-5)
    )
    step = load_checkpoint(path, restored, restored_opt, restored_sched, torch.device("cpu"))
    assert step == 42
    for p, q in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(p, q)
    assert restored_opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_find_latest_checkpoint_picks_highest_step(tmp_path):
    for step in [100, 1000, 10]:
        (tmp_path / f"ckpt-{step:07d}.pt").touch()
    assert find_latest_checkpoint(tmp_path).name == "ckpt-0001000.pt"
    assert find_latest_checkpoint(tmp_path / "missing") is None


def test_generate_respects_token_budget(mini_tokenizer, tiny_config):
    torch.manual_seed(0)
    model = AnuTransformer(tiny_config)
    text, ids = generate(model, mini_tokenizer, "Tom", max_new_tokens=50,
                         temperature=0.8, device=torch.device("cpu"))
    assert text.startswith("Tom")
    assert len(ids) <= len(mini_tokenizer.encode("Tom").ids) + 50


class _EosArgmaxModel:
    """Fake model whose logits always point at <|endoftext|> (id 0)."""

    def __init__(self, vocab_size: int):
        self.config = ModelConfig(vocab_size=vocab_size, context_length=16)
        logits = torch.zeros(vocab_size)
        logits[0] = 10.0
        self._logits = logits

    @torch.no_grad()
    def __call__(self, window):
        return self._logits.unsqueeze(0).unsqueeze(0).repeat(window.shape[0], 1, 1)


def test_generate_stops_at_eos(mini_tokenizer):
    fake = _EosArgmaxModel(mini_tokenizer.get_vocab_size())
    _, ids = generate(fake, mini_tokenizer, "Tom", max_new_tokens=50,
                      temperature=0.0, device=torch.device("cpu"))
    eos = mini_tokenizer.token_to_id("<|endoftext|>")
    assert ids[-1] == eos  # stopped because EOS was sampled
    assert len(ids) <= len(mini_tokenizer.encode("Tom").ids) + 2


def test_generate_is_deterministic_greedy(mini_tokenizer, tiny_config):
    torch.manual_seed(0)
    model = AnuTransformer(tiny_config)
    model.eval()
    a, _ = generate(model, mini_tokenizer, "Tom", max_new_tokens=10,
                    temperature=0.0, device=torch.device("cpu"))
    b, _ = generate(model, mini_tokenizer, "Tom", max_new_tokens=10,
                    temperature=0.0, device=torch.device("cpu"))
    assert a == b




def test_smoke_training_run_and_resume(smoke_bins, tmp_path):
    from train import TrainConfig

    torch.manual_seed(0)
    base = TrainConfig(
        data_dir=smoke_bins,
        ckpt_dir=tmp_path / "ckpts",
        tokenizer_path=tmp_path / "no_tokenizer.json",
        total_steps=20,
        batch_size=2,
        lr=5e-3,
        warmup_steps=5,
        checkpoint_interval=10,
        val_interval=10,
        val_batches=2,
        log_every=5,
        sample_prompts=[],
        seed=0,
    )
    first = run_training(base)
    steps = [m["step"] for m in first]
    assert 20 in steps and 10 in steps
    losses = [m["train_loss"] for m in first if "train_loss" in m]
    assert len(losses) >= 2
    assert losses[-1] < losses[0]
    assert (tmp_path / "ckpts" / "ckpt-0000020.pt").exists()
    assert (tmp_path / "ckpts" / "metrics.jsonl").exists()
    assert any("val_loss" in m for m in first)

    # resume: run again with a higher budget, must continue from step 20 to 30
    second = run_training(
        TrainConfig(
            data_dir=smoke_bins,
            ckpt_dir=tmp_path / "ckpts",
            tokenizer_path=tmp_path / "no_tokenizer.json",
            total_steps=30,
            batch_size=2,
            lr=5e-3,
            warmup_steps=5,
            checkpoint_interval=10,
            val_interval=10,
            val_batches=2,
            log_every=5,
            sample_prompts=[],
            seed=0,
        )
    )
    steps2 = [m["step"] for m in second]
    assert min(steps2) > 20  # resumed after where the first run left off
    assert max(steps2) == 30
    assert (tmp_path / "ckpts" / "ckpt-0000030.pt").exists()


def test_smoke_run_samples_with_real_tokenizer(smoke_bins, tmp_path):
    from train import PROJECT_ROOT, TrainConfig

    tokenizer = PROJECT_ROOT.parent / "weights" / "tokenizer.json"
    if not tokenizer.exists():
        pytest.skip("weights/tokenizer.json not present")
    cfg = TrainConfig(
        data_dir=smoke_bins,
        ckpt_dir=tmp_path / "ckpts2",
        tokenizer_path=tokenizer,
        total_steps=10,
        batch_size=2,
        warmup_steps=2,
        checkpoint_interval=10,
        val_interval=10,
        val_batches=1,
        log_every=5,
        max_new_tokens=10,
        seed=1,
    )
    metrics = run_training(cfg)
    sample_entry = next(m for m in metrics if "samples" in m)
    from train import DEFAULT_SAMPLE_PROMPTS

    assert len(sample_entry["samples"]) == len(DEFAULT_SAMPLE_PROMPTS)
