"""Aspect-aware semantic retriever using NovaSearch/stella_en_1.5B_v5.

Pipeline:
    chunk_document (sentence-boundary aware) → encode aspect + chunks
    → retrieve top-K chunks → reconstruct D_pruned in original document order
    with a final global word cap.

Use prune_batch() when processing multiple (document, aspect) pairs.
"""

import re
from typing import Dict, List, Tuple

import torch
from sentence_transformers import SentenceTransformer

# stella_en_1.5B_v5 ships a custom modeling_qwen.py written for transformers ~4.38-4.45.
# The vLLM container's newer transformers removed several DynamicCache methods.
# Patch all of them before any model code runs so the forward pass doesn't crash.
# The real fix is disabling use_cache on the loaded model (see __init__), but these
# patches are a safety net in case anything slips through.
try:
    from transformers import DynamicCache

    if not hasattr(DynamicCache, "get_max_length"):
        def _get_max_length(self):
            return None  # DynamicCache is unbounded
        DynamicCache.get_max_length = _get_max_length

    if not hasattr(DynamicCache, "get_seq_length"):
        def _get_seq_length(self, layer_idx: int = 0) -> int:
            if not hasattr(self, "key_cache") or len(self.key_cache) <= layer_idx:
                return 0
            return self.key_cache[layer_idx].shape[-2]
        DynamicCache.get_seq_length = _get_seq_length

    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
            max_length = self.get_max_length()
            previous_seq_length = self.get_seq_length(layer_idx)
            if max_length is not None and previous_seq_length + new_seq_length > max_length:
                return max_length - new_seq_length
            return previous_seq_length
        DynamicCache.get_usable_length = _get_usable_length

    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx, kv in enumerate(past_key_values):
                    cache.update(kv[0], kv[1], layer_idx)
            return cache
        DynamicCache.from_legacy_cache = _from_legacy_cache

    if not hasattr(DynamicCache, "to_legacy_cache"):
        def _to_legacy_cache(self):
            legacy = []
            for layer_idx in range(len(self.key_cache)):
                legacy.append((self.key_cache[layer_idx], self.value_cache[layer_idx]))
            return tuple(legacy)
        DynamicCache.to_legacy_cache = _to_legacy_cache

except Exception:
    pass

try:
    from transformers import Qwen2Config
    if not hasattr(Qwen2Config, "rope_theta"):
        Qwen2Config.rope_theta = 1000000.0
except Exception:
    pass

STELLA_MODEL = "NovaSearch/stella_en_1.5B_v5"
DEFAULT_CHUNK_WORDS = 256
DEFAULT_TOP_K_CHUNKS = 15
DEFAULT_WORD_BUDGET = 512


class Retriever:
    def __init__(
        self,
        model_name: str = STELLA_MODEL,
        device: str = "cpu",
        chunk_size: int = DEFAULT_CHUNK_WORDS,
        top_k_chunks: int = DEFAULT_TOP_K_CHUNKS,
        word_budget: int = DEFAULT_WORD_BUDGET,
    ):
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, device=device
        )
        # Disable KV cache on the underlying model — we only do forward passes for
        # encoding, never generation, so the cache is unused. This avoids all
        # DynamicCache compatibility issues with the vLLM container's transformers.
        try:
            self.model._first_module().auto_model.config.use_cache = False
        except Exception:
            pass
        self.chunk_size = chunk_size
        self.top_k_chunks = top_k_chunks
        self.word_budget = word_budget

    def chunk_document(self, text: str) -> List[str]:
        """Split text into ~chunk_size-word chunks without splitting sentences."""
        sentences = self.split_sentences(text)
        if not sentences:
            text = text.strip()
            return [text] if text else []

        chunks: List[str] = []
        current: List[str] = []
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current and current_words + sentence_words > self.chunk_size:
                chunks.append(" ".join(current).strip())
                current = []
                current_words = 0

            current.append(sentence)
            current_words += sentence_words

        if current:
            chunks.append(" ".join(current).strip())

        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _encode_aspects(self, aspects: List[str]) -> torch.Tensor:
        return self.model.encode(
            aspects,
            prompt_name="s2p_query",
            normalize_embeddings=True,
            convert_to_tensor=True,
        )  # (N, D)

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )  # (N, D)

    # ------------------------------------------------------------------
    # Batch API (preferred for production)
    # ------------------------------------------------------------------
    def prune_batch(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, dict]]:
        """Retrieve and prune a batch of (document_text, aspect) pairs.

        Encodes the entire batch with 2 model.encode() calls total:
          1. All unique aspects (query-side)
          2. All chunks across all documents (document-side)

        Args:
            pairs: List of (document_text, aspect_name) tuples.

        Returns:
            List of (d_pruned, stats) aligned with the input pairs.
        """
        if not pairs:
            return []

        # --- Step 1: encode unique aspects ---
        unique_aspects = list(dict.fromkeys(a for _, a in pairs))  # dedup, preserve order
        aspect_embs_all = self._encode_aspects(unique_aspects)     # (U, D)
        aspect_emb_map: Dict[str, torch.Tensor] = {
            a: aspect_embs_all[i] for i, a in enumerate(unique_aspects)
        }

        # --- Step 2: chunk all documents and encode all chunks in one call ---
        all_doc_chunks: List[List[str]] = [self.chunk_document(doc) for doc, _ in pairs]

        flat_chunks: List[str] = []
        chunk_offsets: List[Tuple[int, int]] = []  # (start, end) index into flat_chunks
        for doc_chunks in all_doc_chunks:
            start = len(flat_chunks)
            flat_chunks.extend(doc_chunks)
            chunk_offsets.append((start, len(flat_chunks)))

        chunk_embs_flat = self._encode_texts(flat_chunks) if flat_chunks else torch.empty(0)

        # --- Step 3: retrieve top-K chunks per document ---
        retrieved_indices_per_doc: List[List[int]] = []

        for i, (doc_chunks, (c_start, c_end)) in enumerate(zip(all_doc_chunks, chunk_offsets)):
            if not doc_chunks:
                retrieved_indices_per_doc.append([])
                continue

            aspect_emb = aspect_emb_map[pairs[i][1]]  # (D,)
            doc_chunk_embs = chunk_embs_flat[c_start:c_end]  # (N_chunks, D)
            chunk_scores = doc_chunk_embs @ aspect_emb       # (N_chunks,)

            k = min(self.top_k_chunks, len(doc_chunks))
            top_k_indices = torch.topk(chunk_scores, k).indices.tolist()
            retrieved = sorted(top_k_indices)  # restore document order
            retrieved_indices_per_doc.append(retrieved)

        # --- Step 4: per-document reconstruction with global word cap ---
        results: List[Tuple[str, dict]] = []

        for i, (doc_chunks, (c_start, c_end)) in enumerate(zip(all_doc_chunks, chunk_offsets)):
            doc_text, aspect = pairs[i]
            aspect_emb = aspect_emb_map[aspect]
            doc_chunk_embs = chunk_embs_flat[c_start:c_end]
            chunk_scores = (doc_chunk_embs @ aspect_emb).tolist()
            retrieved = retrieved_indices_per_doc[i]
            k = len(retrieved)

            if not doc_chunks:
                results.append((doc_text, {"num_chunks": 0, "num_chunks_retrieved": 0,
                                            "num_sentences_retained": 0, "num_words_retained": 0}))
                continue

            selected_chunks = [doc_chunks[idx] for idx in retrieved]
            if not selected_chunks:
                results.append((doc_text, {
                    "num_chunks": len(doc_chunks),
                    "num_chunks_retrieved": k,
                    "retrieved_chunk_indices": retrieved,
                    "num_sentences_retained": 0,
                    "num_words_retained": 0,
                    "chunk_scores": chunk_scores,
                }))
                continue

            words_used = 0
            retained_chunks: List[str] = []
            for chunk_text in selected_chunks:
                chunk_words = chunk_text.split()
                remaining = self.word_budget - words_used
                if remaining <= 0:
                    break
                if len(chunk_words) <= remaining:
                    retained_chunks.append(chunk_text)
                    words_used += len(chunk_words)
                else:
                    retained_chunks.append(" ".join(chunk_words[:remaining]))
                    words_used += remaining
                    break

            d_pruned = " ".join(retained_chunks).strip()
            retained_sentences = self.split_sentences(d_pruned)

            if not d_pruned:
                results.append((doc_text, {
                    "num_chunks": len(doc_chunks),
                    "num_chunks_retrieved": k,
                    "retrieved_chunk_indices": retrieved,
                    "num_sentences_retained": 0,
                    "num_words_retained": 0,
                    "chunk_scores": chunk_scores,
                }))
                continue

            results.append((d_pruned, {
                "num_chunks": len(doc_chunks),
                "num_chunks_retrieved": k,
                "retrieved_chunk_indices": retrieved,
                "num_sentences_retained": len(retained_sentences),
                "num_words_retained": len(d_pruned.split()),
                "chunk_scores": chunk_scores,
            }))

        return results

    # ------------------------------------------------------------------
    # Single-document API (kept for debugging / standalone use)
    # ------------------------------------------------------------------
    def prune(self, document_text: str, aspect: str) -> Tuple[str, dict]:
        return self.prune_batch([(document_text, aspect)])[0]
