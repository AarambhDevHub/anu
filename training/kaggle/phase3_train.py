"""Phase 3 on Kaggle: full training run with checkpoint/resume.

Self-contained: paste into a Kaggle notebook (GPU accelerator) and run.
Mirrors training/train.py + training/model/ — keep in sync with those files.

Inputs (upload the Phase 1 notebook's output as a dataset named "anu-data"):
  train.bin, valid.bin, meta.json, tokenizer.json   (from /kaggle/working)

Outputs written to /kaggle/working (auto-downloaded when the run finishes):
  checkpoints/ckpt-<step>.pt  — every --checkpoint-interval + final
  checkpoints/metrics.jsonl   — train/val loss + sampled generations

RESUME WORKFLOW (12h session cap):
  1. Run this notebook; outputs download to your machine when it finishes.
  2. Create a NEW dataset version "anu-data" containing the downloaded
     checkpoints/ + the original bins, or add the checkpoints to the
     notebook input.
  3. Rerun the notebook: it auto-detects the latest checkpoint and
     continues training exactly where it stopped.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CONTEXT_LENGTH = 512
TOTAL_STEPS = 50_000
BATCH_SIZE = 16
LR = 3e-4
MIN_LR = 1e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
CHECKPOINT_INTERVAL = 2_500
VAL_INTERVAL = 2_500
VAL_BATCHES = 128
LOG_EVERY = 25
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.8
SAMPLE_PROMPTS = ["Once upon a time", "Tom and his friends", "The little girl", "In the forest"]
SEED = 0


# ---------------------------------------------------------------- model mirror

class ModelConfig:
    def __init__(self, vocab_size=12_000, context_length=512, n_layer=8, n_embd=512,
                 n_head=8, ffn_dim=2_048, rms_norm_eps=1e-6):
        self.vocab_size, self.context_length = vocab_size, context_length
        self.n_layer, self.n_embd, self.n_head = n_layer, n_embd, n_head
        self.ffn_dim, self.rms_norm_eps = ffn_dim, rms_norm_eps
        self.head_dim = n_embd // n_head

    def to_dict(self):
        return {k: getattr(self, k) for k in
                ("vocab_size", "context_length", "n_layer", "n_embd", "n_head",
                 "ffn_dim", "rms_norm_eps")}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms.to(x.dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.einsum("i,j->ij", torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x):
        seq_len = x.size(2)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, rope):
        super().__init__()
        self.n_embd, self.n_head, self.head_dim = config.n_embd, config.n_head, config.head_dim
        self.rope = rope
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        causal = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(config.ffn_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config, rope):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config, rope)
        self.norm2 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AnuTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.rope = RotaryEmbedding(config.head_dim, config.context_length)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, self.rope) for _ in range(config.n_layer)]
        )
        self.final_rmsnorm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_rmsnorm(x))

    def compute_loss(self, input_ids, targets):
        logits = self(input_ids)
        return nn.functional.cross_entropy(
            logits.view(-1, self.config.vocab_size), targets.view(-1)
        )


# ---------------------------------------------------------------- data mirror

class TokenDataset(Dataset):
    def __init__(self, bin_path, context_length, seed=0):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.context_length = context_length
        self.rng = np.random.default_rng(seed)
        self.max_offset = max(0, len(self.data) - context_length - 1)

    def __len__(self):
        return self.max_offset

    def __getitem__(self, index):
        offset = int(self.rng.integers(0, self.max_offset + 1))
        x = self.data[offset: offset + self.context_length]
        y = self.data[offset + 1: offset + self.context_length + 1]
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


# ---------------------------------------------------------------- training loop

def find_data_dir() -> Path:
    """Locate bins: prefer an 'anu-data' dataset in /kaggle/input, else /kaggle/working."""
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not root.exists():
            continue
        candidates = list(root.rglob("train.bin"))
        if candidates:
            return candidates[0].parent
    raise FileNotFoundError("train.bin not found in /kaggle/input or /kaggle/working")


def find_latest_checkpoint(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("ckpt-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
    return candidates[-1] if candidates else None


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        (decay if param.ndim >= 2 else no_decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def lr_at(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    if step >= TOTAL_STEPS:
        return MIN_LR
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(ckpt_dir, step, model, optimizer, lr_scheduler, metrics):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt-{step:07d}.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "metrics": metrics,
        "config": model.config.to_dict(),
    }, path)
    return path


def load_checkpoint(path, model, optimizer, lr_scheduler, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
    return state["step"]


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, device):
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    ids = list(tokenizer.encode(prompt).ids)
    for _ in range(max_new_tokens):
        window = torch.tensor(ids[-CONTEXT_LENGTH:], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(window)[0, -1]
        if temperature > 0:
            next_id = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).item())
        else:
            next_id = int(logits.argmax().item())
        ids.append(next_id)
        if next_id == eos_id:
            break
    return tokenizer.decode(ids)


def evaluate(model, valid_data, num_batches, device):
    model.eval()
    total = 0.0
    max_offset = len(valid_data)
    with torch.no_grad():
        for i in range(num_batches):
            offset = (i * CONTEXT_LENGTH) % max_offset
            x = torch.from_numpy(
                valid_data.data[offset: offset + CONTEXT_LENGTH].astype(np.int64)
            )
            y = torch.from_numpy(
                valid_data.data[offset + 1: offset + CONTEXT_LENGTH + 1].astype(np.int64)
            )
            total += model.compute_loss(x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)).item()
    model.train()
    return total / num_batches


def main(data_dir: str | None = None, ckpt_dir: str | None = None):
    torch.manual_seed(SEED)
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else Path("/kaggle/working/checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} data_dir={data_dir}")

    meta = json.loads((data_dir / "meta.json").read_text())
    vocab_size = meta["vocab_size"]
    config = ModelConfig(vocab_size=vocab_size, context_length=CONTEXT_LENGTH)
    model = AnuTransformer(config).to(device)

    use_cuda = device.type == "cuda"
    if use_cuda and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif use_cuda:
        dtype = torch.float16
    else:
        dtype = torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and dtype == torch.float16))
    autocast = torch.autocast(device_type=device.type, dtype=dtype) if use_cuda else None

    optimizer = build_optimizer(model, LR, WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_at)

    step = 0
    metrics = []
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        for candidate in sorted(Path("/kaggle/input").rglob("ckpt-*.pt")):
            ckpt_path = candidate
    if ckpt_path is not None:
        step = load_checkpoint(ckpt_path, model, optimizer, lr_scheduler, device)
        print(f"resumed from {ckpt_path} at step {step}")

    tokenizer_path = data_dir / "tokenizer.json"
    if tokenizer_path.exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = None

    loader = DataLoader(TokenDataset(data_dir / "train.bin", CONTEXT_LENGTH, SEED),
                        batch_size=BATCH_SIZE, drop_last=True, num_workers=0)
    valid_data = TokenDataset(data_dir / "valid.bin", CONTEXT_LENGTH, SEED)
    data_iter = iter(loader)

    model.train()
    progress = tqdm(total=TOTAL_STEPS, initial=step, desc="training", unit="step")
    while step < TOTAL_STEPS:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        if autocast is None:
            loss = model.compute_loss(x, y)
        else:
            with autocast:
                loss = model.compute_loss(x, y)
        loss.backward()
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        lr_scheduler.step()
        step += 1
        progress.update(1)

        if step % LOG_EVERY == 0:
            metrics.append({"step": step, "train_loss": loss.item(),
                            "lr": lr_at(min(step, TOTAL_STEPS - 1))})

        if step % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(ckpt_dir, step, model, optimizer, lr_scheduler, metrics)
            print(f"checkpoint saved at step {step}")

        if step % VAL_INTERVAL == 0:
            val_loss = evaluate(model, valid_data, VAL_BATCHES, device)
            entry = {"step": step, "val_loss": val_loss}
            if tokenizer is not None:
                entry["samples"] = [
                    generate(model, tokenizer, p, MAX_NEW_TOKENS, TEMPERATURE, device)
                    for p in SAMPLE_PROMPTS
                ]
            metrics.append(entry)
            print(f"step {step}: val_loss={val_loss:.4f}")

    save_checkpoint(ckpt_dir, step, model, optimizer, lr_scheduler, metrics)
    with open(ckpt_dir / "metrics.jsonl", "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")
    print(f"done at step {step}. outputs in /kaggle/working/checkpoints/")


if __name__ == "__main__":
    main()
