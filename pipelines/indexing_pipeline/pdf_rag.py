from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import fitz
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    HuggingFaceInferenceEmbeddingProvider,
)


DEFAULT_MODEL_NAME = DEFAULT_EMBEDDING_MODEL_NAME


@dataclass
class ChunkRecord:
    chunk_id: int
    source_file: str
    page_number: int
    text: str


class PDFRAG:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        provider: str | None = None,
        use_fp16: bool = False,
        token: str | None = None,
    ) -> None:
        self.model_name = model_name
        # use_fp16 is kept for API compatibility with the previous local FlagEmbedding
        # provider but is ignored because inference providers handle precision server-side.
        self.use_fp16 = use_fp16
        self.provider = provider
        self._provider = HuggingFaceInferenceEmbeddingProvider(
            model_name=model_name,
            provider=provider,
            token=token,
        )

    def extract_pdf_pages(self, pdf_path: Path) -> list[tuple[int, str]]:
        document = fitz.open(pdf_path)
        pages: list[tuple[int, str]] = []

        for page_index in range(len(document)):
            text = document.load_page(page_index).get_text("text")
            normalized = self._normalize_whitespace(text)
            if normalized:
                pages.append((page_index + 1, normalized))

        document.close()
        return pages

    def chunk_text(
        self,
        text: str,
        *,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> list[str]:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )
        return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]

    def build_records_from_pdf(
        self,
        pdf_path: Path,
        *,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        next_chunk_id = 0

        for page_number, page_text in self.extract_pdf_pages(pdf_path):
            for chunk in self.chunk_text(
                page_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ):
                records.append(
                    ChunkRecord(
                        chunk_id=next_chunk_id,
                        source_file=str(pdf_path.resolve()),
                        page_number=page_number,
                        text=chunk,
                    )
                )
                next_chunk_id += 1

        return records

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        return self._provider.embed_texts(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self._provider.embed_query(query)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
