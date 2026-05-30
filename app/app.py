from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import faiss
import fitz
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.manifold import TSNE
from umap import UMAP


DEFAULT_METADATA_PATH = Path("storage/metadata.json")
DEFAULT_INDEX_PATH = Path("storage/faiss.index")
DEFAULT_PDF_PATH = Path("t130sp_na_operator_manual.pdf")


def load_chunk_records(metadata_path: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    raw_records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise ValueError("metadata.json must contain a JSON array of chunk records.")

    records: list[dict[str, Any]] = []
    for record in raw_records:
        if not isinstance(record, dict):
            continue
        normalized = dict(record)
        text = str(normalized.get("text", ""))
        normalized["text"] = text
        normalized["chunk_method"] = normalized.get("chunk_method", "RecursiveCharacterTextSplitter")
        normalized["char_count"] = len(text)
        normalized["word_count"] = len(text.split())
        normalized["line_count"] = len([line for line in text.splitlines() if line.strip()])
        normalized["source_name"] = Path(str(normalized.get("source_file", "unknown"))).name
        records.append(normalized)

    dataframe = pd.DataFrame(records)
    if dataframe.empty:
        return records, dataframe

    if "chunk_id" in dataframe.columns:
        dataframe["chunk_id"] = pd.to_numeric(dataframe["chunk_id"], errors="coerce")
    if "page_number" in dataframe.columns:
        dataframe["page_number"] = pd.to_numeric(dataframe["page_number"], errors="coerce")
    if "char_count" in dataframe.columns:
        dataframe["char_count"] = pd.to_numeric(dataframe["char_count"], errors="coerce")
    if "word_count" in dataframe.columns:
        dataframe["word_count"] = pd.to_numeric(dataframe["word_count"], errors="coerce")

    return records, dataframe


@st.cache_data(show_spinner=False)
def load_metadata_dataframe(metadata_path_str: str) -> pd.DataFrame:
    metadata_path = Path(metadata_path_str)
    _, dataframe = load_chunk_records(metadata_path)
    return dataframe


@st.cache_resource(show_spinner=False)
def load_faiss_index(index_path_str: str) -> faiss.Index:
    index_path = Path(index_path_str)
    return faiss.read_index(str(index_path))


def reconstruct_embeddings(index: faiss.Index) -> np.ndarray:
    if index.ntotal == 0:
        return np.empty((0, 0), dtype="float32")
    return np.asarray(index.reconstruct_n(0, index.ntotal), dtype="float32")


def normalize_raw_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long_unit(text: str, max_chunk_size: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []
    current_length = 0

    for word in words:
        proposed_length = current_length + len(word) + (1 if current_words else 0)
        if current_words and proposed_length > max_chunk_size:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
            continue

        current_words.append(word)
        current_length = proposed_length

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def merge_small_text_chunks(
    chunks: list[str],
    *,
    min_chunk_size: int,
    target_chunk_size: int,
    max_chunk_size: int,
) -> list[str]:
    merged = [chunk for chunk in chunks if chunk.strip()]
    index = 0

    while index < len(merged):
        current_length = len(merged[index])
        if current_length >= min_chunk_size or len(merged) == 1:
            index += 1
            continue

        left_length = len(merged[index - 1]) if index > 0 else None
        right_length = len(merged[index + 1]) if index + 1 < len(merged) else None
        candidates: list[tuple[str, int]] = []

        if left_length is not None and left_length + 2 + current_length <= max_chunk_size:
            candidates.append(("left", abs((left_length + 2 + current_length) - target_chunk_size)))
        if right_length is not None and current_length + 2 + right_length <= max_chunk_size:
            candidates.append(("right", abs((current_length + 2 + right_length) - target_chunk_size)))

        if candidates:
            direction = min(candidates, key=lambda item: item[1])[0]
            if direction == "left":
                merged[index - 1] = f"{merged[index - 1]}\n\n{merged[index]}".strip()
                merged.pop(index)
                index = max(0, index - 1)
                continue

            merged[index + 1] = f"{merged[index]}\n\n{merged[index + 1]}".strip()
            merged.pop(index)
            continue

        if 0 < index < len(merged) - 1:
            triple_length = len(merged[index - 1]) + len(merged[index]) + len(merged[index + 1]) + 4
            if triple_length <= max_chunk_size:
                merged[index - 1] = f"{merged[index - 1]}\n\n{merged[index]}\n\n{merged[index + 1]}".strip()
                merged.pop(index + 1)
                merged.pop(index)
                index = max(0, index - 1)
                continue

        index += 1

    return merged


def split_text_soft(
    text: str,
    *,
    min_chunk_size: int,
    target_chunk_size: int,
    max_chunk_size: int,
) -> list[str]:
    text = normalize_raw_text(text)
    if not text:
        return []
    if len(text) <= max_chunk_size:
        return [text]

    raw_units = re.split(r"(?<=[.!?])\s+|\n{1,}", text)
    units: list[str] = []
    for raw_unit in raw_units:
        unit = normalize_raw_text(raw_unit)
        if not unit:
            continue
        if len(unit) > max_chunk_size:
            units.extend(split_long_unit(unit, max_chunk_size))
            continue
        units.append(unit)

    chunks: list[str] = []
    current = ""

    for unit in units:
        separator = " " if current else ""
        proposed = f"{current}{separator}{unit}" if current else unit

        if not current:
            current = unit
            continue

        if len(proposed) <= target_chunk_size:
            current = proposed
            continue

        if len(current) < min_chunk_size and len(proposed) <= max_chunk_size:
            current = proposed
            continue

        chunks.append(current)
        current = unit

    if current:
        chunks.append(current)

    return merge_small_text_chunks(
        chunks,
        min_chunk_size=min_chunk_size,
        target_chunk_size=target_chunk_size,
        max_chunk_size=max_chunk_size,
    )


def extract_raw_segments(
    pdf_path: Path,
    *,
    min_chunk_size: int,
    target_chunk_size: int,
    max_chunk_size: int,
) -> list[dict[str, Any]]:
    document = fitz.open(pdf_path)
    segments: list[dict[str, Any]] = []

    try:
        for page_index in range(len(document)):
            page = document.load_page(page_index)
            blocks = page.get_text("blocks")
            for block_index, block in enumerate(blocks):
                text = normalize_raw_text(block[4])
                if not text:
                    continue

                split_blocks = split_text_soft(
                    text,
                    min_chunk_size=min_chunk_size,
                    target_chunk_size=target_chunk_size,
                    max_chunk_size=max_chunk_size,
                )
                for segment_index, segment_text in enumerate(split_blocks):
                    segments.append(
                        {
                            "page_number": page_index + 1,
                            "block_index": block_index,
                            "segment_index": segment_index,
                            "text": segment_text,
                            "char_count": len(segment_text),
                        }
                    )
    finally:
        document.close()

    return segments


def finalize_raw_chunk(
    chunk_id: int,
    source_file: Path,
    segments: list[dict[str, Any]],
    *,
    min_chunk_size: int,
    target_chunk_size: int,
    max_chunk_size: int,
) -> dict[str, Any]:
    chunk_text = "\n\n".join(segment["text"] for segment in segments).strip()
    page_numbers = sorted({int(segment["page_number"]) for segment in segments})
    block_indexes = [int(segment["block_index"]) for segment in segments]
    return {
        "chunk_id": chunk_id,
        "source_file": str(source_file.resolve()),
        "source_name": source_file.name,
        "page_number": page_numbers[0],
        "page_start": page_numbers[0],
        "page_end": page_numbers[-1],
        "page_span": f"{page_numbers[0]}-{page_numbers[-1]}" if len(page_numbers) > 1 else str(page_numbers[0]),
        "chunk_method": "PyMuPDF raw blocks",
        "min_chunk_size": min_chunk_size,
        "target_chunk_size": target_chunk_size,
        "max_chunk_size": max_chunk_size,
        "block_count": len(segments),
        "block_start": min(block_indexes),
        "block_end": max(block_indexes),
        "char_count": len(chunk_text),
        "word_count": len(chunk_text.split()),
        "line_count": len([line for line in chunk_text.splitlines() if line.strip()]),
        "text": chunk_text,
    }


def merge_small_chunk_records(
    chunks: list[dict[str, Any]],
    *,
    source_file: Path,
    min_chunk_size: int,
    target_chunk_size: int,
    max_chunk_size: int,
) -> list[dict[str, Any]]:
    merged = chunks[:]
    index = 0

    while index < len(merged):
        current_length = len("\n\n".join(segment["text"] for segment in merged[index]["segments"]).strip())
        if current_length >= min_chunk_size or len(merged) == 1:
            index += 1
            continue

        candidates: list[tuple[str, int]] = []

        if index > 0:
            left_length = len("\n\n".join(segment["text"] for segment in merged[index - 1]["segments"]).strip())
            combined_left = left_length + current_length + 2
            if combined_left <= max_chunk_size:
                candidates.append(("left", abs(combined_left - target_chunk_size)))

        if index + 1 < len(merged):
            right_length = len("\n\n".join(segment["text"] for segment in merged[index + 1]["segments"]).strip())
            combined_right = current_length + right_length + 2
            if combined_right <= max_chunk_size:
                candidates.append(("right", abs(combined_right - target_chunk_size)))

        if candidates:
            direction = min(candidates, key=lambda item: item[1])[0]
            if direction == "left":
                merged[index - 1]["segments"].extend(merged[index]["segments"])
                merged.pop(index)
                index = max(0, index - 1)
                continue

            merged[index + 1]["segments"] = merged[index]["segments"] + merged[index + 1]["segments"]
            merged.pop(index)
            continue

        index += 1

    finalized: list[dict[str, Any]] = []
    for chunk_id, item in enumerate(merged):
        finalized.append(
            finalize_raw_chunk(
                chunk_id,
                source_file,
                item["segments"],
                min_chunk_size=min_chunk_size,
                target_chunk_size=target_chunk_size,
                max_chunk_size=max_chunk_size,
            )
        )
    return finalized


def build_raw_chunks(
    pdf_path: Path,
    *,
    min_chunk_size: int,
    target_chunk_size: int,
    max_chunk_size: int,
) -> pd.DataFrame:
    segments = extract_raw_segments(
        pdf_path,
        min_chunk_size=min_chunk_size,
        target_chunk_size=target_chunk_size,
        max_chunk_size=max_chunk_size,
    )
    if not segments:
        return pd.DataFrame()

    chunk_buckets: list[dict[str, Any]] = []
    current_segments: list[dict[str, Any]] = []
    current_length = 0

    for segment in segments:
        segment_text = segment["text"]
        segment_length = len(segment_text)
        separator_length = 2 if current_segments else 0
        proposed_length = current_length + separator_length + segment_length

        if not current_segments:
            current_segments = [segment]
            current_length = segment_length
            continue

        if proposed_length <= target_chunk_size:
            current_segments.append(segment)
            current_length = proposed_length
            continue

        if current_length < min_chunk_size and proposed_length <= max_chunk_size:
            current_segments.append(segment)
            current_length = proposed_length
            continue

        chunk_buckets.append({"segments": current_segments})
        current_segments = [segment]
        current_length = segment_length

    if current_segments:
        chunk_buckets.append({"segments": current_segments})

    finalized = merge_small_chunk_records(
        chunk_buckets,
        source_file=pdf_path,
        min_chunk_size=min_chunk_size,
        target_chunk_size=target_chunk_size,
        max_chunk_size=max_chunk_size,
    )
    dataframe = pd.DataFrame(finalized)
    if not dataframe.empty:
        dataframe["chunk_id"] = pd.to_numeric(dataframe["chunk_id"], errors="coerce")
        dataframe["page_number"] = pd.to_numeric(dataframe["page_number"], errors="coerce")
        dataframe["char_count"] = pd.to_numeric(dataframe["char_count"], errors="coerce")
        dataframe["word_count"] = pd.to_numeric(dataframe["word_count"], errors="coerce")
    return dataframe


@st.cache_data(show_spinner=False)
def load_raw_chunk_dataframe(
    pdf_path_str: str,
    min_chunk_size: int,
    target_chunk_size: int,
    max_chunk_size: int,
) -> pd.DataFrame:
    return build_raw_chunks(
        Path(pdf_path_str),
        min_chunk_size=min_chunk_size,
        target_chunk_size=target_chunk_size,
        max_chunk_size=max_chunk_size,
    )


def build_histogram(dataframe: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(columns=["Length bin", "Chunks"])

    max_chars = int(dataframe["char_count"].max())
    start_values = range(0, max_chars + bin_size, bin_size)
    rows: list[dict[str, Any]] = []

    for start in start_values:
        end = start + bin_size - 1
        count = int(dataframe["char_count"].between(start, end, inclusive="both").sum())
        rows.append({"Length bin": f"{start}-{end}", "Chunks": count})

    return pd.DataFrame(rows)


def render_summary(dataframe: pd.DataFrame) -> None:
    total_chunks = int(len(dataframe))
    total_sources = int(dataframe["source_name"].nunique()) if "source_name" in dataframe else 0
    total_pages = int(dataframe["page_number"].nunique()) if "page_number" in dataframe else 0
    avg_chars = float(dataframe["char_count"].mean()) if total_chunks else 0.0
    avg_words = float(dataframe["word_count"].mean()) if total_chunks else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Chunks", total_chunks)
    col2.metric("Source files", total_sources)
    col3.metric("Pages", total_pages)
    col4.metric("Avg chars / chunk", f"{avg_chars:.1f}")
    col5.metric("Avg words / chunk", f"{avg_words:.1f}")


def filter_dataframe(
    dataframe: pd.DataFrame,
    selected_sources: list[str],
    selected_methods: list[str],
    char_range: tuple[int, int],
) -> pd.DataFrame:
    return dataframe[
        dataframe["source_name"].isin(selected_sources)
        & dataframe["chunk_method"].isin(selected_methods)
        & dataframe["char_count"].between(char_range[0], char_range[1], inclusive="both")
    ].copy()


def render_chunk_statistics(filtered: pd.DataFrame, bin_size: int) -> None:
    st.subheader("Overview")
    render_summary(filtered)

    st.subheader("Chunk Length Distribution")
    histogram_df = build_histogram(filtered, bin_size=bin_size)
    st.bar_chart(histogram_df.set_index("Length bin"))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Chunks By Source File")
        by_source = (
            filtered.groupby("source_name", dropna=False)
            .size()
            .reset_index(name="chunk_count")
            .sort_values("chunk_count", ascending=False)
        )
        st.dataframe(by_source, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Chunks By Page Number")
        by_page = (
            filtered.groupby("page_number", dropna=False)
            .size()
            .reset_index(name="chunk_count")
            .sort_values(["chunk_count", "page_number"], ascending=[False, True])
        )
        st.dataframe(by_page, use_container_width=True, hide_index=True)

    st.subheader("Chunk Browser")
    display_columns = [
        column
        for column in ["chunk_id", "source_name", "page_number", "chunk_method", "char_count", "word_count"]
        if column in filtered.columns
    ]
    st.dataframe(filtered[display_columns], use_container_width=True, hide_index=True)

    selectable_ids = filtered["chunk_id"].dropna().astype(int).tolist() if "chunk_id" in filtered.columns else []
    if not selectable_ids:
        st.warning("No chunk IDs are available for detail inspection.")
        return

    selected_chunk_id = st.selectbox("Select chunk", selectable_ids, index=0, format_func=lambda value: f"Chunk {value}")
    chunk_row = filtered.loc[filtered["chunk_id"] == selected_chunk_id].iloc[0]

    st.subheader("Chunk Detail")
    metadata_columns = [column for column in filtered.columns if column != "text"]
    st.json({column: chunk_row[column] for column in metadata_columns})
    st.text_area("Chunk content", value=str(chunk_row["text"]), height=320)


def render_raw_chunk_analyzer(
    dataframe: pd.DataFrame,
    *,
    pdf_path: Path,
    min_chunk_size: int,
    target_chunk_size: int,
    max_chunk_size: int,
    bin_size: int,
) -> None:
    st.subheader("Chunk Size Analyzer")
    st.caption("Raw chunking uses `fitz` block extraction only. No LangChain splitter is used on this page.")

    if dataframe.empty:
        st.warning("No chunks were generated from the selected PDF.")
        return

    render_summary(dataframe)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min configured", min_chunk_size)
    col2.metric("Target configured", target_chunk_size)
    col3.metric("Max configured", max_chunk_size)
    col4.metric("Below min chunks", int((dataframe["char_count"] < min_chunk_size).sum()))

    st.subheader("Chunk Length Distribution")
    histogram_df = build_histogram(dataframe, bin_size=bin_size)
    st.bar_chart(histogram_df.set_index("Length bin"))

    st.subheader("Generated Chunks")
    display_columns = [
        column
        for column in [
            "chunk_id",
            "page_start",
            "page_end",
            "page_span",
            "block_count",
            "char_count",
            "word_count",
        ]
        if column in dataframe.columns
    ]
    st.dataframe(dataframe[display_columns], use_container_width=True, hide_index=True)

    chunk_ids = dataframe["chunk_id"].dropna().astype(int).tolist()
    selected_chunk_id = st.selectbox("Select generated chunk", chunk_ids, format_func=lambda value: f"Chunk {value}")
    chunk_row = dataframe.loc[dataframe["chunk_id"] == selected_chunk_id].iloc[0]

    st.subheader("Generated Chunk Detail")
    metadata_columns = [column for column in dataframe.columns if column != "text"]
    st.json({column: chunk_row[column] for column in metadata_columns})
    st.text_area("Chunk content", value=str(chunk_row["text"]), height=320)

    st.caption(f"PDF source: `{pdf_path.resolve()}`")


def reduce_embeddings(
    embeddings: np.ndarray,
    *,
    reduction_method: str,
    dimensions: int,
    n_neighbors: int,
) -> np.ndarray:
    if reduction_method == "UMAP":
        reducer = UMAP(
            n_components=dimensions,
            n_neighbors=min(n_neighbors, max(2, len(embeddings) - 1)),
            random_state=42,
        )
        return reducer.fit_transform(embeddings)

    perplexity = min(max(5, n_neighbors), max(5, len(embeddings) - 1))
    reducer = TSNE(
        n_components=dimensions,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def render_embedding_space(
    filtered: pd.DataFrame,
    *,
    index_path: Path,
    vector_backend: str,
    reduction_method: str,
    dimensions: int,
    n_neighbors: int,
) -> None:
    st.subheader("Embedding Space")
    st.caption("Projection is computed from vectors reconstructed from the selected vector store.")

    if vector_backend != "FAISS":
        st.warning(f"Vector backend `{vector_backend}` is not supported in this project.")
        return

    if not index_path.exists():
        st.error(f"Index file not found: {index_path.resolve()}")
        return

    index = load_faiss_index(str(index_path))
    embeddings = reconstruct_embeddings(index)
    if len(embeddings) != len(filtered.index) and len(embeddings) != int(index.ntotal):
        st.error("The FAISS vector count does not match the metadata rows.")
        return

    if filtered.empty:
        st.warning("No filtered chunks are available for embedding visualization.")
        return

    if "chunk_id" not in filtered.columns:
        st.error("Chunk IDs are required to align metadata with embeddings.")
        return

    chunk_ids = filtered["chunk_id"].dropna().astype(int).tolist()
    valid_chunk_ids = [chunk_id for chunk_id in chunk_ids if 0 <= chunk_id < len(embeddings)]
    if len(valid_chunk_ids) < 3:
        st.warning("At least 3 chunks are required to visualize the embedding space.")
        return

    selected_embeddings = embeddings[valid_chunk_ids]
    selected_metadata = (
        filtered.loc[filtered["chunk_id"].astype(int).isin(valid_chunk_ids)]
        .copy()
        .sort_values("chunk_id")
        .reset_index(drop=True)
    )
    selected_embeddings = embeddings[selected_metadata["chunk_id"].astype(int).to_numpy()]

    with st.spinner(f"Running {reduction_method} projection..."):
        reduced = reduce_embeddings(
            selected_embeddings,
            reduction_method=reduction_method,
            dimensions=dimensions,
            n_neighbors=n_neighbors,
        )

    selected_metadata["component_1"] = reduced[:, 0]
    selected_metadata["component_2"] = reduced[:, 1]

    hover_fields = {
        "chunk_id": True,
        "source_name": True,
        "page_number": True,
        "char_count": True,
        "word_count": True,
    }

    color_column = "source_name" if selected_metadata["source_name"].nunique() > 1 else "page_number"

    if dimensions == 3:
        selected_metadata["component_3"] = reduced[:, 2]
        figure = px.scatter_3d(
            selected_metadata,
            x="component_1",
            y="component_2",
            z="component_3",
            color=color_column,
            hover_data=hover_fields,
        )
    else:
        figure = px.scatter(
            selected_metadata,
            x="component_1",
            y="component_2",
            color=color_column,
            hover_data=hover_fields,
        )

    figure.update_layout(height=720)
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        selected_metadata[
            [column for column in ["chunk_id", "source_name", "page_number", "component_1", "component_2"] if column in selected_metadata.columns]
            + (["component_3"] if "component_3" in selected_metadata.columns else [])
        ],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="Chunk Statistics", layout="wide")
    st.title("Chunk Statistics Explorer")
    st.caption("Inspect chunk statistics, embedding-space projections, and custom PyMuPDF chunking behavior.")

    metadata_path_input = st.sidebar.text_input("Metadata path", str(DEFAULT_METADATA_PATH))
    index_path_input = st.sidebar.text_input("FAISS index path", str(DEFAULT_INDEX_PATH))
    pdf_path_input = st.sidebar.text_input("PDF path", str(DEFAULT_PDF_PATH))
    metadata_path = Path(metadata_path_input)
    index_path = Path(index_path_input)
    pdf_path = Path(pdf_path_input)

    navigation = st.sidebar.radio("Navigation", ["Chunk statistics", "Embedding space", "Chunk size analyzer"])

    if navigation == "Chunk size analyzer":
        min_chunk_size = st.sidebar.slider("Min chunk size", min_value=100, max_value=800, value=400, step=50)
        target_chunk_size = st.sidebar.slider("Soft chunk size", min_value=min_chunk_size, max_value=1200, value=800, step=50)
        max_chunk_size = st.sidebar.slider("Max chunk size", min_value=target_chunk_size, max_value=2000, value=1200, step=50)
        bin_size = st.sidebar.slider("Histogram bin size", min_value=50, max_value=1000, value=100, step=50)

        if not pdf_path.exists():
            st.error(f"PDF file not found: {pdf_path.resolve()}")
            st.stop()

        with st.spinner("Generating raw chunks from PDF..."):
            raw_chunk_dataframe = load_raw_chunk_dataframe(
                str(pdf_path),
                min_chunk_size,
                target_chunk_size,
                max_chunk_size,
            )

        render_raw_chunk_analyzer(
            raw_chunk_dataframe,
            pdf_path=pdf_path,
            min_chunk_size=min_chunk_size,
            target_chunk_size=target_chunk_size,
            max_chunk_size=max_chunk_size,
            bin_size=bin_size,
        )
        return

    if not metadata_path.exists():
        st.error(f"Metadata file not found: {metadata_path.resolve()}")
        st.stop()

    try:
        dataframe = load_metadata_dataframe(str(metadata_path))
    except Exception as exc:  # pragma: no cover - defensive UI path
        st.error(f"Failed to load metadata: {exc}")
        st.stop()

    if dataframe.empty:
        st.warning("No chunk records were found in the metadata file.")
        st.stop()

    sources = sorted(dataframe["source_name"].dropna().unique().tolist())
    chunk_methods = sorted(dataframe["chunk_method"].dropna().unique().tolist())

    selected_sources = st.sidebar.multiselect("Source files", sources, default=sources)
    selected_methods = st.sidebar.multiselect("Chunk methods", chunk_methods, default=chunk_methods)
    min_chars, max_chars = int(dataframe["char_count"].min()), int(dataframe["char_count"].max())
    char_range = st.sidebar.slider("Character count range", min_chars, max_chars, (min_chars, max_chars))
    filtered = filter_dataframe(dataframe, selected_sources, selected_methods, char_range)

    if navigation == "Chunk statistics":
        bin_size = st.sidebar.slider("Histogram bin size", min_value=50, max_value=1000, value=100, step=50)
        render_chunk_statistics(filtered, bin_size)
        return

    vector_backend = st.sidebar.selectbox("Vector store backend", ["FAISS"])
    reduction_method = st.sidebar.selectbox("Reduction method", ["UMAP", "tSNE"])
    dimensions = st.sidebar.segmented_control("Dimension", options=[2, 3], default=2)
    n_neighbors = st.sidebar.slider("n_neighbors", min_value=2, max_value=100, value=15, step=1)

    render_embedding_space(
        filtered,
        index_path=index_path,
        vector_backend=vector_backend,
        reduction_method=reduction_method,
        dimensions=int(dimensions),
        n_neighbors=n_neighbors,
    )


if __name__ == "__main__":
    main()
