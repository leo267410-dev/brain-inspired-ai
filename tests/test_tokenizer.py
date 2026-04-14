"""Tests for the CodeLingual tokenizer."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.tokenizer.codelingual import CodeLingualTokenizer


def test_tokenizer_init():
    """Test tokenizer initialization."""
    tok = CodeLingualTokenizer(vocab_size=48000)
    assert tok.vocab_size == 48000
    assert tok.pad_token_id == 0
    assert tok.unk_token_id == 1
    assert tok.bos_token_id == 2
    assert tok.eos_token_id == 3


def test_encode_decode():
    """Test basic encode/decode roundtrip."""
    tok = CodeLingualTokenizer()
    text = "Hello World"
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert len(ids) > 0
    # First token should be BOS, last should be EOS
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id

    decoded = tok.decode(ids)
    assert decoded == text


def test_encode_without_special_tokens():
    """Test encoding without special tokens."""
    tok = CodeLingualTokenizer()
    text = "abc"
    ids = tok.encode(text, add_special_tokens=False)
    assert ids[0] != tok.bos_token_id


def test_encode_max_length():
    """Test max_length truncation."""
    tok = CodeLingualTokenizer()
    text = "a" * 1000
    ids = tok.encode(text, max_length=50)
    assert len(ids) <= 50


def test_encode_code():
    """Test code-aware encoding."""
    tok = CodeLingualTokenizer()
    code = "def foo():\n    return 42"
    ids = tok.encode_code(code)
    assert isinstance(ids, list)
    assert ids[0] == tok.bos_token_id
    assert ids[1] == tok.code_token_id
    assert ids[-1] == tok.eos_token_id


def test_save_load():
    """Test tokenizer save/load roundtrip."""
    tok = CodeLingualTokenizer(vocab_size=1000)
    text = "test"
    ids = tok.encode(text)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tok.save(f.name)
        loaded = CodeLingualTokenizer.load(f.name)

    assert loaded.vocab_size == 1000
    loaded_ids = loaded.encode(text)
    assert ids == loaded_ids


def test_decode_skip_special():
    """Test decode skips special tokens when requested."""
    tok = CodeLingualTokenizer()
    ids = [tok.bos_token_id, tok.pad_token_id, tok.eos_token_id]
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert decoded == ""


if __name__ == "__main__":
    test_tokenizer_init()
    test_encode_decode()
    test_encode_without_special_tokens()
    test_encode_max_length()
    test_encode_code()
    test_save_load()
    test_decode_skip_special()
    print("All tokenizer tests passed!")
