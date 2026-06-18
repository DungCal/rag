from __future__ import annotations

from typing import Any


DEFAULT_RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_RERANK_INSTRUCTION = "Given a document question, retrieve relevant passages that answer the question."


class QwenReranker:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        batch_size: int = 8,
        max_length: int = 4096,
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
        use_fp16: bool = False,
        apply_sigmoid: bool = False,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for Qwen reranking. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        model_kwargs: dict[str, Any] = {}
        if use_fp16:
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None:
                model_kwargs["torch_dtype"] = torch.float16

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction = instruction.strip() or DEFAULT_RERANK_INSTRUCTION
        self.apply_sigmoid = apply_sigmoid
        self._model = CrossEncoder(
            model_name,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            prompts={"rerank": self.instruction},
            default_prompt_name="rerank",
            max_length=max_length,
        )

    def score(self, query: str, documents: list[str]) -> list[float]:
        query_text = query.strip()
        if not query_text:
            raise ValueError("Query must not be empty")
        if not documents:
            return []

        pairs = [(query_text, document) for document in documents]
        activation_fn = None
        if self.apply_sigmoid:
            try:
                import torch
            except ImportError as exc:
                raise ImportError("torch is required when apply_sigmoid=True") from exc
            activation_fn = torch.nn.Sigmoid()

        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            activation_fn=activation_fn,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    def rerank(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not results:
            return []

        documents = [str(item.get("text", "")) for item in results]
        scores = self.score(query, documents)

        enriched: list[dict[str, Any]] = []
        for retrieval_rank, (item, rerank_score) in enumerate(zip(results, scores), start=1):
            payload = dict(item)
            payload["retrieval_rank"] = retrieval_rank
            payload["retrieval_score"] = float(item.get("score", 0.0))
            payload["rerank_score"] = rerank_score
            enriched.append(payload)

        reranked = sorted(enriched, key=lambda item: item["rerank_score"], reverse=True)
        for rerank_rank, item in enumerate(reranked, start=1):
            item["rerank_rank"] = rerank_rank
            item["rank_shift"] = item["retrieval_rank"] - rerank_rank

        return reranked
