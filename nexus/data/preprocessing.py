"""Data preprocessing utilities."""

from __future__ import annotations

import re
from typing import Dict, List, Optional


def clean_code(code: str) -> str:
    """Clean and normalize code strings."""
    # Remove trailing whitespace
    lines = [line.rstrip() for line in code.split("\n")]
    # Remove excessive blank lines
    cleaned = []
    prev_blank = False
    for line in lines:
        is_blank = len(line.strip()) == 0
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank
    return "\n".join(cleaned).strip()


def clean_text(text: str) -> str:
    """Clean and normalize natural language text."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def chunk_text(
    text: str, chunk_size: int = 8192, overlap: int = 256,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: input text.
        chunk_size: maximum chunk size in characters.
        overlap: overlap between chunks.

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from markdown-formatted text.

    Args:
        text: markdown text with code blocks.

    Returns:
        List of dicts with 'language' and 'code' keys.
    """
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [{"language": lang or "unknown", "code": code.strip()} for lang, code in matches]


def create_training_pairs(
    code: str,
    context_ratio: float = 0.5,
) -> Dict[str, str]:
    """
    Create training pairs from code by splitting into context and target.

    Args:
        code: source code string.
        context_ratio: fraction of code to use as context.

    Returns:
        Dict with 'context' and 'target' keys.
    """
    lines = code.split("\n")
    split_point = int(len(lines) * context_ratio)
    return {
        "context": "\n".join(lines[:split_point]),
        "target": "\n".join(lines[split_point:]),
    }
