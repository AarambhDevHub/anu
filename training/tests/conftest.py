import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from data.prepare_dataset import pack_stories
from model import ModelConfig
from tests.stories import SAMPLE_STORIES


@pytest.fixture(scope="session")
def mini_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=2048,
        special_tokens=["<|endoftext|>"],
        min_frequency=1,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(SAMPLE_STORIES, trainer=trainer)
    return tokenizer


@pytest.fixture(scope="module")
def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=300,
        context_length=16,
        n_layer=2,
        n_embd=64,
        n_head=4,
        ffn_dim=256,
    )


@pytest.fixture
def smoke_bins(tmp_path, mini_tokenizer):
    """Tiny deterministic packed bins + meta.json for a smoke training run."""
    rng = np.random.default_rng(7)
    train_stories = ["".join(chr(97 + rng.integers(0, 26)) for _ in range(120)) for _ in range(60)]
    valid_stories = ["".join(chr(97 + rng.integers(0, 26)) for _ in range(120)) for _ in range(20)]
    pack_stories(train_stories, mini_tokenizer, tmp_path / "train.bin")
    pack_stories(valid_stories, mini_tokenizer, tmp_path / "valid.bin")
    (tmp_path / "meta.json").write_text(
        f'{{"context_length": 32, "vocab_size": {mini_tokenizer.get_vocab_size()},'
        f' "train": {{"num_tokens": 10000}}, "valid": {{"num_tokens": 2000}}}}'
    )
    return tmp_path
