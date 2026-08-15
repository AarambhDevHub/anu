"""Training loop: cosine LR + warmup, mixed precision, checkpoint/resume, val loss, sampling.

Run on a Kaggle GPU for the real training run; local CPU runs are only for
smoke tests (use --steps/--batch-size tiny values). Checkpoints are saved
every --checkpoint-interval steps and auto-resumed from the latest one in
--ckpt-dir, so training can span multiple 12-hour Kaggle sessions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TokenDataset, load_meta  # noqa: E402
from model import AnuTransformer, ModelConfig, num_parameters  # noqa: E402

DEFAULT_SAMPLE_PROMPTS = [
    "Once upon a time",
    "Tom and his friends",
    "The little girl",
    "In the forest",
]


@dataclass
class TrainConfig:
    data_dir: Path = PROJECT_ROOT / "data" / "tokenized"
    ckpt_dir: Path = PROJECT_ROOT / "data" / "checkpoints"
    tokenizer_path: Path = PROJECT_ROOT / ".." / "weights" / "tokenizer.json"
    total_steps: int = 25_000
    batch_size: int = 32
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    checkpoint_interval: int = 2_500
    val_interval: int = 2_500
    val_batches: int = 128
    log_every: int = 25
    max_new_tokens: int = 128
    temperature: float = 0.8
    sample_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_SAMPLE_PROMPTS))
    seed: int = 0


def build_lr_schedule(total_steps: int, warmup_steps: int, max_lr: float, min_lr: float):
    """Cosine decay with linear warmup; returns lr_at(step) for step in [0, total_steps)."""

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        if step >= total_steps:
            return min_lr
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    return lr_at


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        (decay if param.ndim >= 2 else no_decay).append(param)  # norms (1-D) get no decay
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("ckpt-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
    return candidates[-1] if candidates else None


def save_checkpoint(
    ckpt_dir: Path, step: int, model, optimizer, lr_scheduler, metrics: list[dict]
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt-{step:07d}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "metrics": metrics,
            "config": model.config.to_dict(),
        },
        path,
    )
    return path


def load_checkpoint(path: Path, model, optimizer, lr_scheduler, device: torch.device) -> int:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
    return state["step"]


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> tuple[str, list[int]]:
    """Greedy or temperature-sampled continuation, stopping at <|endoftext|>.

    Returns (text, full_id_sequence) so callers can reason in token space
    (decoded garbage bytes can re-encode to a different token count).
    """
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    ids = list(tokenizer.encode(prompt).ids)
    context_length = model.config.context_length
    for _ in range(max_new_tokens):
        window = torch.tensor(ids[-context_length:], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(window)[0, -1]
        if temperature > 0:
            scaled = logits / temperature
            probs = torch.softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
        else:
            next_id = int(logits.argmax().item())
        ids.append(next_id)
        if next_id == eos_id:
            break
    return tokenizer.decode(ids), ids


def evaluate(
    model: torch.nn.Module, valid_data, context_length: int, num_batches: int, device: torch.device
) -> float:
    """Mean cross-entropy over fixed (deterministic) windows of the valid split."""
    model.eval()
    total, count = 0.0, 0
    max_offset = len(valid_data) - context_length - 1
    with torch.no_grad():
        for i in range(num_batches):
            offset = (i * context_length) % max_offset
            x = torch.from_numpy(
                valid_data.data[offset : offset + context_length].astype(np.int64)
            )
            y = torch.from_numpy(
                valid_data.data[offset + 1 : offset + context_length + 1].astype(np.int64)
            )
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            total += model.compute_loss(x, y).item()
            count += 1
    model.train()
    return total / count


def compute_fast_loss(
    model: torch.nn.Module, input_ids: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy computed in the autocast dtype (fp16/bf16).

    torch's cross_entropy upcasts logits to fp32 internally, which at
    vocab_size=12000 dominates wall time on a T4 (the whole (B*T, 12000)
    logits tensor materialized in fp32). Max-subtraction keeps exp() in
    range, so this is numerically safe. Training-only — sample.py uses the
    exact fp32 CE for parity.
    """
    logits = model(input_ids).view(-1, model.config.vocab_size)
    targets = targets.view(-1)
    max_logit = logits.max(dim=-1, keepdim=True).values
    log_sum_exp = (logits - max_logit).exp().sum(dim=-1).log() + max_logit.squeeze(-1)
    gathered = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - gathered).mean()


def run_training(cfg: TrainConfig, device: torch.device | None = None) -> list[dict]:
    torch.manual_seed(cfg.seed)
    meta = load_meta(cfg.data_dir / "meta.json")
    context_length = meta["context_length"]
    config = ModelConfig(vocab_size=meta["vocab_size"], context_length=context_length)

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    use_cuda = device.type == "cuda"
    model = AnuTransformer(config, use_sdpa=use_cuda)
    model.to(device)

    if use_cuda and torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(0)[0] >= 8:
        dtype = torch.bfloat16
    elif use_cuda:
        dtype = torch.float16
    else:
        dtype = torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and dtype == torch.float16))
    autocast = torch.autocast(device_type=device.type, dtype=dtype) if use_cuda else None

    if use_cuda:
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"GPU: {name} (cc {capability[0]}.{capability[1]}) dtype={dtype}")
    print(
        f"model: {num_parameters(model) / 1e6:.1f}M params (tied), "
        f"tokens/step: {cfg.batch_size * context_length:,}"
    )

    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay)
    lr_at = build_lr_schedule(cfg.total_steps, cfg.warmup_steps, cfg.lr, cfg.min_lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_at)

    step = 0
    metrics: list[dict] = []
    ckpt_path = find_latest_checkpoint(cfg.ckpt_dir)
    if ckpt_path is not None:
        step = load_checkpoint(ckpt_path, model, optimizer, lr_scheduler, device)
        print(f"resumed from {ckpt_path} at step {step}")

    tokenizer = None
    if cfg.tokenizer_path.exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(cfg.tokenizer_path))

    train_data = TokenDataset(cfg.data_dir / "train.bin", context_length, seed=cfg.seed)
    loader = DataLoader(train_data, batch_size=cfg.batch_size, drop_last=True, num_workers=0)
    valid_data = TokenDataset(cfg.data_dir / "valid.bin", context_length, seed=cfg.seed)
    data_iter = iter(loader)

    model.train()
    progress = tqdm(total=cfg.total_steps, initial=step, desc="training", unit="step")
    t_fwd = t_bwd = t_opt = 0.0
    while step < cfg.total_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        if autocast is None:
            loss = model.compute_loss(x, y)
        else:
            with autocast:
                loss = compute_fast_loss(model, x, y)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        lr_scheduler.step()
        t3 = time.perf_counter()
        t_fwd += (t1 - t0) * 1000
        t_bwd += (t2 - t1) * 1000
        t_opt += (t3 - t2) * 1000
        step += 1
        progress.update(1)

        if step % cfg.log_every == 0:
            metrics.append(
                {
                    "step": step,
                    "train_loss": loss.item(),
                    "lr": lr_at(min(step, cfg.total_steps - 1)),
                    "fwd_ms": t_fwd / cfg.log_every,
                    "bwd_ms": t_bwd / cfg.log_every,
                    "opt_ms": t_opt / cfg.log_every,
                }
            )
            print(
                f"step {step}: fwd+loss {t_fwd / cfg.log_every:.0f}ms "
                f"bwd {t_bwd / cfg.log_every:.0f}ms opt {t_opt / cfg.log_every:.0f}ms"
            )
            t_fwd = t_bwd = t_opt = 0.0

        if step % cfg.checkpoint_interval == 0:
            save_checkpoint(cfg.ckpt_dir, step, model, optimizer, lr_scheduler, metrics)
            print(f"checkpoint saved at step {step}")

        if step % cfg.val_interval == 0:
            val_loss = evaluate(model, valid_data, context_length, cfg.val_batches, device)
            entry = {"step": step, "val_loss": val_loss}
            if tokenizer is not None:
                samples = [
                    generate(model, tokenizer, p, cfg.max_new_tokens, cfg.temperature, device)[0]
                    for p in cfg.sample_prompts
                ]
                entry["samples"] = samples
            metrics.append(entry)
            print(f"step {step}: val_loss={val_loss:.4f}")

    save_checkpoint(cfg.ckpt_dir, step, model, optimizer, lr_scheduler, metrics)
    metrics_path = cfg.ckpt_dir / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")
    print(f"training done at step {step}; metrics -> {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=TrainConfig.data_dir)
    parser.add_argument("--ckpt-dir", type=Path, default=TrainConfig.ckpt_dir)
    parser.add_argument("--tokenizer", type=Path, default=TrainConfig.tokenizer_path)
    parser.add_argument("--steps", type=int, default=TrainConfig.total_steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--min-lr", type=float, default=TrainConfig.min_lr)
    parser.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--checkpoint-interval", type=int, default=TrainConfig.checkpoint_interval)
    parser.add_argument("--val-interval", type=int, default=TrainConfig.val_interval)
    parser.add_argument("--val-batches", type=int, default=TrainConfig.val_batches)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--max-new-tokens", type=int, default=TrainConfig.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=TrainConfig.temperature)
    parser.add_argument("--no-sample", action="store_true", help="skip generation sampling")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer,
        total_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        checkpoint_interval=args.checkpoint_interval,
        val_interval=args.val_interval,
        val_batches=args.val_batches,
        log_every=args.log_every,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        sample_prompts=[] if args.no_sample else list(DEFAULT_SAMPLE_PROMPTS),
        seed=args.seed,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
