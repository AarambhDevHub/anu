from pathlib import Path

import pytest
from tokenizers import Tokenizer

from tests.stories import SAMPLE_STORIES

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "text",
    [
        "Once upon a time, there was a little cat named Tom.",
        "It rained 99.5% of the time! 'Really?' asked Tom.",
        "  Indented   and   oddly spaced   text.",
        "Unicode: café, naïve, 你好, 🚀 and émigré.",
        "The end.",
    ],
)
def test_round_trip(mini_tokenizer, text):
    ids = mini_tokenizer.encode(text).ids
    decoded = mini_tokenizer.decode(ids)
    assert decoded == text


def test_round_trip_all_stories(mini_tokenizer):
    for story in SAMPLE_STORIES:
        assert mini_tokenizer.decode(mini_tokenizer.encode(story).ids) == story


def test_special_token_present(mini_tokenizer):
    assert mini_tokenizer.token_to_id("<|endoftext|>") is not None


def test_save_load(tmp_path, mini_tokenizer):
    path = tmp_path / "tokenizer.json"
    mini_tokenizer.save(str(path))
    loaded = Tokenizer.from_file(str(path))
    text = "Tom and the dog played in the garden."
    assert loaded.encode(text).ids == mini_tokenizer.encode(text).ids
    assert loaded.decode(loaded.encode(text).ids) == text


def test_real_tokenizer_round_trip():
    """Round-trip against the production weights/tokenizer.json if present."""
    path = PROJECT_ROOT / "weights" / "tokenizer.json"
    if not path.exists():
        pytest.skip("weights/tokenizer.json not built yet")
    tokenizer = Tokenizer.from_file(str(path))
    for text in ["Once upon a time", "The little cat slept", "hello world 42"]:
        assert tokenizer.decode(tokenizer.encode(text).ids) == text
