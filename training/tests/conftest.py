import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

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
