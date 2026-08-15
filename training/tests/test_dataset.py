import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import TokenDataset, load_meta
from data.prepare_dataset import pack_stories
from tests.stories import SAMPLE_STORIES


@pytest.fixture
def packed_bin(tmp_path, mini_tokenizer):
    out = tmp_path / "train.bin"
    pack_stories(SAMPLE_STORIES, mini_tokenizer, out)
    return out


def test_dataset_shapes(packed_bin, mini_tokenizer):
    ds = TokenDataset(packed_bin, context_length=16, seed=0)
    x, y = ds[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_target_is_shifted_input(packed_bin, mini_tokenizer):
    ds = TokenDataset(packed_bin, context_length=16, seed=1)
    x, y = ds[3]
    assert torch.equal(y[:-1], x[1:])


def test_dataloader_batch(packed_bin, mini_tokenizer):
    ds = TokenDataset(packed_bin, context_length=16, seed=2)
    loader = DataLoader(ds, batch_size=4)
    x, y = next(iter(loader))
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_meta_roundtrip(tmp_path):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text('{"context_length": 512, "vocab_size": 12000}')
    assert load_meta(meta_path)["context_length"] == 512
