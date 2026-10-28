from __future__ import annotations

import re
from collections.abc import Callable, Iterable


Tokenize = Callable[[str], list[int]]
Decode = Callable[[list[int]], str]


class TokenChunker:
    def __init__(
        self,
        tokenize: Tokenize,
        decode: Decode,
        target_min: int = 280,
        target_max: int = 420,
        hard_max: int = 480,
        overlap: int = 48,
    ):
        if not 0 <= overlap < target_max <= hard_max:
            raise ValueError("分块参数不合法")
        self.tokenize = tokenize
        self.decode = decode
        self.target_min = target_min
        self.target_max = target_max
        self.hard_max = hard_max
        self.overlap = overlap

    def split_sections(self, sections: Iterable[tuple[str, str]]) -> list[tuple[str, str, int]]:
        output: list[tuple[str, str, int]] = []
        for heading, text in sections:
            pieces = self._split_text(text)
            output.extend((heading, piece, len(self.tokenize(piece))) for piece in pieces if piece.strip())
        return output

    def _split_text(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
        if not paragraphs:
            return []
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            for unit in self._fit_paragraph(paragraph):
                candidate = f"{current}\n\n{unit}".strip() if current else unit
                if len(self.tokenize(candidate)) <= self.target_max:
                    current = candidate
                    continue
                if current:
                    chunks.append(current)
                current = unit
        if current:
            chunks.append(current)
        return self._enforce_hard_limit(chunks)

    def _fit_paragraph(self, paragraph: str) -> list[str]:
        tokens = self.tokenize(paragraph)
        if len(tokens) <= self.target_max:
            return [paragraph]
        sentences = [s.strip() for s in re.split(r"(?<=[。！？；.!?;])", paragraph) if s.strip()]
        if len(sentences) > 1:
            results: list[str] = []
            current = ""
            for sentence in sentences:
                candidate = current + sentence
                if current and len(self.tokenize(candidate)) > self.target_max:
                    results.extend(self._token_windows(current))
                    current = sentence
                else:
                    current = candidate
            if current:
                results.extend(self._token_windows(current))
            return results
        return self._token_windows(paragraph)

    def _token_windows(self, text: str) -> list[str]:
        tokens = self.tokenize(text)
        if len(tokens) <= self.hard_max:
            return [text]
        step = self.target_max - self.overlap
        return [self.decode(tokens[start : start + self.target_max]).strip()
                for start in range(0, len(tokens), step)
                if tokens[start : start + self.target_max]]

    def _enforce_hard_limit(self, chunks: list[str]) -> list[str]:
        output: list[str] = []
        for chunk in chunks:
            if len(self.tokenize(chunk)) <= self.hard_max:
                output.append(chunk)
            else:
                output.extend(self._token_windows(chunk))
        if any(len(self.tokenize(chunk)) > self.hard_max for chunk in output):
            raise AssertionError("生成了超过 tokenizer 硬上限的分块")
        return output


class CharacterTokenizer:
    """Deterministic test fallback. Production must use the local BGE tokenizer."""

    @staticmethod
    def encode(text: str) -> list[int]:
        return [ord(ch) for ch in text]

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def load_bge_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return (lambda text: tokenizer.encode(text, add_special_tokens=False, verbose=False)), lambda ids: tokenizer.decode(ids, skip_special_tokens=True)

