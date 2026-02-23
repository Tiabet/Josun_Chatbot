#!/usr/bin/env python3
"""
Hybrid Path Retriever
======================
Combines BM25 (sparse) and Dense Embedding (semantic) search for metadata paths.

Search Flow:
1. BM25 search on paths
2. Dense embedding search on paths
3. Combine scores (weighted fusion)
4. Return top-k paths with metadata
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import asyncio
import os
import zipfile
from dotenv import load_dotenv
from openai import AsyncOpenAI

import bm25s
import Stemmer

from llm_provider import create_async_embed_client, detect_provider


class HybridPathRetriever:
    """Hybrid retriever combining BM25 and dense embeddings."""
    
    def __init__(
        self,
        bm25_index_path: str = 'HotpotQA/bm25_index',
        embeddings_path: str = 'HotpotQA/path_embeddings.npz',
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        load_values: bool | None = None,
        enable_dense: bool | None = None,
    ):
        """
        Args:
            bm25_index_path: Path to BM25 index directory
            embeddings_path: Path to embeddings .npz file
            bm25_weight: Weight for BM25 scores (0-1)
            dense_weight: Weight for dense scores (0-1)
        """
        load_dotenv()
        
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
        # Load BM25 index
        print(f"Loading BM25 index from: {bm25_index_path}")
        self.bm25 = bm25s.BM25.load(bm25_index_path)
        
        # Load path metadata arrays from embeddings artifact.
        # IMPORTANT: for very large corpora (e.g., Sillok), loading the full dense embedding matrix
        # into RAM is not feasible. We therefore load metadata arrays unconditionally, but only load
        # the embedding matrix if dense search is explicitly enabled.
        print(f"Loading embeddings metadata from: {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True)
        self.titles = data['titles']
        self.key_paths = data['key_paths']
        self.doc_ids = data['doc_ids'] if 'doc_ids' in data.files else None
        self.source_titles = data['source_titles'] if 'source_titles' in data.files else None
        self.entity_titles = data['entity_titles'] if 'entity_titles' in data.files else None

        # Values: optional (can be huge). Default policy:
        # - If load_values is explicitly set, follow it.
        # - Otherwise, auto-disable if embedding_texts JSON is "too large".
        self.values: Optional[List[str]] = None
        embedding_texts_path = embeddings_path.replace('path_embeddings', 'embedding_texts').replace('.npz', '.json')
        try:
            et_path = Path(embedding_texts_path)
            et_size = et_path.stat().st_size if et_path.exists() else 0
        except Exception:
            et_size = 0
        if load_values is None:
            # 256MB threshold: above this, JSON array loading is likely to OOM.
            load_values = bool(et_size and et_size <= 256 * 1024 * 1024)
        if load_values:
            try:
                print(f"Loading values from: {embedding_texts_path}")
                with open(embedding_texts_path, 'r', encoding='utf-8') as f:
                    embedding_texts_data = json.load(f)
                self.values = [str(item.get('value', '')) for item in embedding_texts_data]
            except Exception as e:
                self.values = None
                print(f"Warning: Failed to load values (will omit value field). ({e})")
        else:
            if et_size:
                print(
                    f"Warning: Values file is large ({et_size/1024/1024:.1f}MB). "
                    "Skipping values load to avoid OOM (value field will be empty)."
                )
            else:
                print("Warning: Values file not found. (value field will be empty)")

        # Dense search enablement
        # - enable_dense explicitly controls whether we ever load the full embedding matrix.
        # - Default: disabled unless ENABLE_DENSE_SEARCH=1.
        self.embeddings = None
        self.embeddings_normalized = None

        # Candidate-only dense rerank support via memmap (preferred for huge corpora).
        # By default we try to use the checkpoint produced by step4:
        #   <embeddings_path>.with_suffix('.ckpt')/embeddings.npy
        self.embeddings_memmap = None
        self.embeddings_memmap_path: Optional[str] = None
        self._default_memmap_path: Optional[str] = None
        self._embeddings_npz_path: Optional[str] = None
        self._memmap_extract_attempted = False
        self.auto_extract_memmap_from_npz = (
            (os.getenv("AUTO_EXTRACT_MEMMAP_FROM_NPZ", "1") or "").strip().lower()
            not in ("0", "false", "no", "n")
        )
        try:
            self._embeddings_npz_path = str(Path(embeddings_path).resolve())
        except Exception:
            self._embeddings_npz_path = str(embeddings_path)
        memmap_env = (os.getenv('EMBEDDINGS_MEMMAP_PATH') or '').strip()
        try:
            # Resolve relative paths so the memmap can be found reliably even if CWD differs.
            default_ckpt = (Path(embeddings_path).resolve()).with_suffix('.ckpt') / 'embeddings.npy'
        except Exception:
            default_ckpt = None
        memmap_path = Path(memmap_env) if memmap_env else default_ckpt
        if memmap_path and isinstance(memmap_path, Path):
            try:
                self._default_memmap_path = str(memmap_path.resolve())
            except Exception:
                self._default_memmap_path = str(memmap_path)
        if memmap_path and isinstance(memmap_path, Path) and memmap_path.exists():
            try:
                print(f"[INFO] Using embeddings memmap for dense rerank: {memmap_path}")
                self.embeddings_memmap = np.lib.format.open_memmap(str(memmap_path), mode='r')
                self.embeddings_memmap_path = str(memmap_path)
            except Exception as e:
                self.embeddings_memmap = None
                self.embeddings_memmap_path = None
                print(f"Warning: Failed to open embeddings memmap. Dense rerank may be unavailable. ({e})")
        try:
            emb_size = Path(embeddings_path).stat().st_size
        except Exception:
            emb_size = 0
        if enable_dense is None:
            forced_dense = (os.getenv('ENABLE_DENSE_SEARCH') or '').strip().lower() in ('1', 'true', 'yes', 'y')
            enable_dense = bool(forced_dense and emb_size)
        if enable_dense:
            print("[INFO] Dense search enabled: loading full embedding matrix (may require large RAM)")
            self.embeddings = data['embeddings']
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings_normalized = self.embeddings / (norms + 1e-8)
        else:
            print(
                "[INFO] Dense search disabled (BM25-only / rank-fusion without dense). "
                "Set ENABLE_DENSE_SEARCH=1 to force dense (requires enough RAM)."
            )
        
        # Stemmer for BM25 preprocessing
        self.stemmer = Stemmer.Stemmer('english')
        
        # Stopwords
        self.stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose',
            'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also'
        }
        
        # OpenAI client for query embedding
        self.embed_client = None
        self.embed_model = "text-embedding-3-small"

        # Dense search/rerank is optional. If no API keys are set, we still allow BM25-only retrieval.
        has_any_key = bool(
            (os.getenv("GOOGLE_API_KEY") or "").strip()
            or (os.getenv("OPENAI_API_KEY") or "").strip()
            or (os.getenv("ALICE_OPENAI_KEY") or "").strip()
        )

        # We configure embed_client as long as we have a key.
        # Even if full-matrix dense search is disabled, we may still need query embeddings
        # for candidate-only dense reranking.
        if has_any_key:
            try:
                cfg = detect_provider()
                self.embed_model = cfg.embed_model
                self.embed_client = create_async_embed_client(cfg)
            except Exception as e:
                self.embed_client = None
                print(f"Warning: Embedding client not configured. Dense search disabled. ({e})")
        else:
            if not has_any_key:
                print("Warning: No API key found. Dense search disabled (BM25-only).")

        print(f"[OK] BM25 weight: {bm25_weight}, Dense weight: {dense_weight}")
        
        # Build title to indices map for fast lookup.
        # NOTE: For very large corpora (millions of paths), building this map can explode memory.
        # It's not required for the v12 pipeline, so we skip it by default beyond a threshold.
        self.title_to_indices: Optional[Dict[str, List[int]]] = None
        try:
            titles_n = int(len(self.titles))
        except Exception:
            titles_n = 0

        if titles_n and titles_n <= 500_000:
            self.title_to_indices = {}
            print("Building title index...")
            for idx, title in enumerate(self.titles):
                t = str(title)
                if t not in self.title_to_indices:
                    self.title_to_indices[t] = []
                self.title_to_indices[t].append(int(idx))
        else:
            print("[INFO] Skipping title index build (corpus too large)")

    def _value_at(self, idx: int) -> str:
        if not self.values:
            return ""
        try:
            return str(self.values[idx])
        except Exception:
            return ""

    def has_dense_rerank(self) -> bool:
        # Dense rerank needs a query embedder + either memmap or full normalized matrix.
        if self.embeddings_memmap is None and self.embeddings_normalized is None:
            self._ensure_embeddings_memmap_open()
        return (self.embed_client is not None) and (
            self.embeddings_memmap is not None or self.embeddings_normalized is not None
        )

    @staticmethod
    def _human_gb(n_bytes: int) -> str:
        return f"{(float(n_bytes) / (1024.0 ** 3)):.2f}GB"

    def _extract_memmap_from_npz(self, out_path: Path) -> bool:
        """Best-effort one-time extraction of embeddings.npy from path_embeddings.npz."""
        npz_path = Path(str(self._embeddings_npz_path or "")).resolve()
        if (not npz_path.exists()) or (npz_path.suffix.lower() != ".npz"):
            return False

        member_name = "embeddings.npy"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

            with zipfile.ZipFile(npz_path, "r") as zf:
                if member_name not in zf.namelist():
                    print(
                        f"Warning: '{member_name}' not found in {npz_path}. "
                        "Dense rerank from NPZ is unavailable."
                    )
                    return False
                info = zf.getinfo(member_name)
                print(
                    "[INFO] Extracting embeddings memmap from NPZ for dense rerank: "
                    f"{npz_path} -> {out_path} ({self._human_gb(int(info.file_size))})"
                )
                with zf.open(member_name, "r") as src, open(tmp_path, "wb") as dst:
                    while True:
                        chunk = src.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

            if out_path.exists():
                out_path.unlink(missing_ok=True)
            os.replace(str(tmp_path), str(out_path))
            print(f"[OK] Created dense memmap: {out_path}")
            return True
        except Exception as e:
            print(f"Warning: Failed to extract embeddings memmap from NPZ. ({e})")
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return False

    def _ensure_embeddings_memmap_open(self) -> None:
        """Try to open the embeddings memmap if it isn't already open."""
        if self.embeddings_memmap is not None:
            return
        memmap_env = (os.getenv('EMBEDDINGS_MEMMAP_PATH') or '').strip()
        candidates: List[str] = []
        if memmap_env:
            candidates.append(memmap_env)
        if self._default_memmap_path:
            candidates.append(self._default_memmap_path)
        for p in candidates:
            try:
                pp = Path(p)
                if not pp.is_absolute():
                    pp = pp.resolve()
                if not pp.exists():
                    continue
                print(f"[INFO] Opening embeddings memmap for dense rerank: {pp}")
                self.embeddings_memmap = np.lib.format.open_memmap(str(pp), mode='r')
                self.embeddings_memmap_path = str(pp)
                return
            except Exception:
                continue

        # If memmap doesn't exist yet, optionally bootstrap it from NPZ once.
        if self.auto_extract_memmap_from_npz and (not self._memmap_extract_attempted):
            self._memmap_extract_attempted = True
            target = Path(self._default_memmap_path) if self._default_memmap_path else None
            if target is not None and self._extract_memmap_from_npz(target):
                try:
                    print(f"[INFO] Opening embeddings memmap for dense rerank: {target}")
                    self.embeddings_memmap = np.lib.format.open_memmap(str(target), mode='r')
                    self.embeddings_memmap_path = str(target)
                    return
                except Exception:
                    pass

    def get_normalized_embeddings(self, indices: List[int]) -> np.ndarray:
        """Return L2-normalized embeddings for the given indices.

        This supports candidate-only reranking without loading the full matrix into RAM.
        """
        idx = [int(i) for i in indices]

        # Preferred: memmap-backed row reads (lazy-open if needed)
        if self.embeddings_memmap is None:
            self._ensure_embeddings_memmap_open()
        if self.embeddings_memmap is not None:
            rows = np.asarray(self.embeddings_memmap[idx], dtype=np.float32)
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            return rows / (norms + 1e-8)

        # Fallback: full-matrix loaded
        if self.embeddings_normalized is not None:
            return np.asarray(self.embeddings_normalized[idx], dtype=np.float32)

        raise RuntimeError(
            "Dense embeddings not available. Provide EMBEDDINGS_MEMMAP_PATH pointing to a .npy memmap "
            "(e.g., Sillok/path_embeddings_v5.ckpt/embeddings.npy), keep AUTO_EXTRACT_MEMMAP_FROM_NPZ=1 "
            "to auto-bootstrap from .npz, or enable full dense loading."
        )

    @staticmethod
    def _opt_field(arr, idx):
        if arr is None:
            return None
        v = arr[idx]
        return None if v is None else str(v)
            
    def get_indices_for_title(self, title: str) -> List[int]:
        """Get all path indices for a given title."""
        if self.title_to_indices is None:
            # Slow fallback; intended only for debugging.
            t = str(title)
            return [int(i) for i, v in enumerate(self.titles) if str(v) == t]
        return self.title_to_indices.get(str(title), [])

    async def score_candidates_rrf(self, query: str, candidate_indices: List[int], top_k: int = 10) -> List[Dict]:
        """
        Score specific candidate paths using RRF of BM25 and Dense scores.
        
        Args:
            query: The query string
            candidate_indices: List of indices to score
            top_k: Number of results to return
            
        Returns:
            List of dicts with scored results
        """
        if not candidate_indices:
            return []
            
        # 1. Dense Scores
        # Embed query
        query_embedding = await self.embed_query(query)
        
        # Get candidate embeddings
        # Prefer candidate-only row reads so we don't require loading the full matrix.
        candidate_embeddings = self.get_normalized_embeddings(candidate_indices)
        
        # Compute Cosine Similarity (Dot product of normalized vectors)
        dense_raw_scores = np.dot(candidate_embeddings, query_embedding)
        
        # 2. BM25 Scores
        query_tokens = self.preprocess_query(query)
        
        bm25_raw_scores = np.zeros(len(candidate_indices))
        
        if query_tokens:
            # We retrieve top-k where k is large enough to likely include our candidates
            # or we can try to retrieve all. 
            # bm25s.retrieve with k=len(self.titles) gets all non-zero scores?
            # Actually, bm25s returns top-k. 
            # Let's use k=50000 as a safe upper bound for now, but capped at corpus size.
            bm25_k = min(50000, len(self.titles))
            results, scores = self.bm25.retrieve([query_tokens], k=bm25_k)
            
            # Create a map for fast lookup
            # results[0] are indices, scores[0] are scores
            bm25_map = dict(zip(results[0], scores[0]))
            
            for i, idx in enumerate(candidate_indices):
                bm25_raw_scores[i] = bm25_map.get(idx, 0.0)
        
        # 3. RRF Calculation
        rrf_k = 60
        
        # Get ranks for Dense
        # argsort gives indices that would sort the array. [::-1] reverses to descending.
        # We need the rank of each item in the original array.
        dense_sort_indices = np.argsort(dense_raw_scores)[::-1]
        dense_ranks = np.zeros(len(candidate_indices), dtype=int)
        for rank, i in enumerate(dense_sort_indices):
            dense_ranks[i] = rank
            
        # Get ranks for BM25
        bm25_sort_indices = np.argsort(bm25_raw_scores)[::-1]
        bm25_ranks = np.zeros(len(candidate_indices), dtype=int)
        for rank, i in enumerate(bm25_sort_indices):
            bm25_ranks[i] = rank
            
        combined_scores = []
        scored_candidates = []
        
        for i, idx in enumerate(candidate_indices):
            # RRF Score
            dense_rrf = 1.0 / (rrf_k + dense_ranks[i])
            bm25_rrf = 1.0 / (rrf_k + bm25_ranks[i])
            
            combined = (self.dense_weight * dense_rrf) + (self.bm25_weight * bm25_rrf)
            
            scored_candidates.append({
                'index': idx,
                'title': str(self.titles[idx]),
                'doc_id': self._opt_field(self.doc_ids, idx),
                'source_title': self._opt_field(self.source_titles, idx),
                'entity_title': self._opt_field(self.entity_titles, idx),
                'key_path': str(self.key_paths[idx]),
                'value': self._value_at(idx),
                'score': float(combined),
                'dense_score': float(dense_raw_scores[i]),
                'bm25_score': float(bm25_raw_scores[i]),
                'dense_rank': int(dense_ranks[i]),
                'bm25_rank': int(bm25_ranks[i])
            })
            
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_candidates[:top_k]

    def preprocess_query(self, query: str) -> List[str]:
        """Preprocess query for BM25."""
        import re
        
        # Lowercase
        text = query.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]
        
        # Stemming
        tokens = self.stemmer.stemWords(tokens)
        
        return tokens
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Get embedding for query."""
        if not self.embed_client:
            raise ValueError("Embedding client not configured")
        
        response = await self.embed_client.embeddings.create(
            model=self.embed_model,
            input=[query]
        )
        
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def search_bm25(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        BM25 search.
        
        Returns:
            List of (index, score) tuples
        """
        query_tokens = self.preprocess_query(query)
        
        if not query_tokens:
            return []
        
        results, scores = self.bm25.retrieve([query_tokens], k=top_k)
        
        return list(zip(results[0], scores[0]))
    
    async def search_dense(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        Dense embedding search.
        
        Returns:
            List of (index, score) tuples
        """
        # Dense search requires BOTH a query embed client AND a full normalized matrix.
        # (Candidate-only memmap is for reranking, not whole-corpus ANN search.)
        if (not self.embed_client) or (self.embeddings_normalized is None):
            return []

        query_embedding = await self.embed_query(query)
        
        # Cosine similarity (dot product since normalized)
        similarities = np.dot(self.embeddings_normalized, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    async def search_hybrid(
        self,
        query: str,
        top_k: int = 3,
        bm25_candidates: int = 50,
        dense_candidates: int = 50,
        fusion_method: str = 'rrf',
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 and dense.
        
        Args:
            query: Search query
            top_k: Number of final results
            bm25_candidates: Number of BM25 candidates
            dense_candidates: Number of dense candidates
            
        Returns:
            List of result dicts with path metadata and scores
        """
        fusion_method = str(fusion_method or 'rrf').lower().strip()

        # Get component results.
        # NOTE: On huge corpora (e.g., Sillok), full-matrix dense search is disabled.
        # To keep dense+bm25 fusion behavior, we approximate dense search by scoring
        # the BM25 candidate pool using memmap-backed embeddings (candidate-only).
        bm25_results = self.search_bm25(query, bm25_candidates)
        dense_results = await self.search_dense(query, dense_candidates)

        if (not dense_results) and bm25_results and self.has_dense_rerank():
            try:
                cand_indices = [int(idx) for idx, _ in bm25_results]
                query_embedding = await self.embed_query(query)
                cand_emb = self.get_normalized_embeddings(cand_indices)
                sims = np.dot(cand_emb, query_embedding)
                k_take = min(int(dense_candidates), int(len(cand_indices)))
                if k_take > 0:
                    top_pos = np.argsort(sims)[::-1][:k_take]
                    dense_results = [(cand_indices[int(i)], float(sims[int(i)])) for i in top_pos]
            except Exception:
                # If anything goes wrong, fall back to BM25-only.
                dense_results = []

        # Pure modes (ablation): return a single signal without rank fusion.
        if fusion_method == 'bm25':
            if not bm25_results:
                return []
            bm25_sorted = sorted(bm25_results, key=lambda x: x[1], reverse=True)[:top_k]
            results: List[Dict] = []
            for idx, bm25_raw in bm25_sorted:
                idx_i = int(idx)
                results.append(
                    {
                        'index': idx_i,
                        'title': str(self.titles[idx_i]),
                        'doc_id': self._opt_field(self.doc_ids, idx_i),
                        'source_title': self._opt_field(self.source_titles, idx_i),
                        'entity_title': self._opt_field(self.entity_titles, idx_i),
                        'key_path': str(self.key_paths[idx_i]),
                        'value': self._value_at(idx_i),
                        'score': float(bm25_raw),
                        'bm25_score': float(bm25_raw),
                        'dense_score': 0.0,
                        'fusion_method': fusion_method,
                        'bm25_raw_score': float(bm25_raw),
                        'dense_raw_score': None,
                    }
                )
            return results

        if fusion_method == 'dense':
            if not dense_results:
                return []
            dense_sorted = sorted(dense_results, key=lambda x: x[1], reverse=True)[:top_k]
            results: List[Dict] = []
            for idx, dense_raw in dense_sorted:
                idx_i = int(idx)
                results.append(
                    {
                        'index': idx_i,
                        'title': str(self.titles[idx_i]),
                        'doc_id': self._opt_field(self.doc_ids, idx_i),
                        'source_title': self._opt_field(self.source_titles, idx_i),
                        'entity_title': self._opt_field(self.entity_titles, idx_i),
                        'key_path': str(self.key_paths[idx_i]),
                        'value': self._value_at(idx_i),
                        'score': float(dense_raw),
                        'bm25_score': 0.0,
                        'dense_score': float(dense_raw),
                        'fusion_method': fusion_method,
                        'bm25_raw_score': None,
                        'dense_raw_score': float(dense_raw),
                    }
                )
            return results

        # RRF (Reciprocal Rank Fusion)
        # Score = 1 / (k + rank)
        rrf_k = 60
        
        all_indices = set()
        bm25_ranks = {}
        if bm25_results:
            for rank, (idx, _) in enumerate(bm25_results):
                bm25_ranks[idx] = rank
                all_indices.add(idx)
        
        dense_ranks = {}
        if dense_results:
            for rank, (idx, _) in enumerate(dense_results):
                dense_ranks[idx] = rank
                all_indices.add(idx)
        
        def _minmax_scale(values: Dict[int, float]) -> Dict[int, float]:
            if not values:
                return {}
            vmin = min(values.values())
            vmax = max(values.values())
            denom = (vmax - vmin)
            if denom <= 0:
                # All equal (or only one value): no separation.
                return {k: 0.0 for k in values.keys()}
            return {k: (float(v) - float(vmin)) / float(denom) for k, v in values.items()}

        combined_scores = []
        if fusion_method == 'minmax':
            # Build raw-score maps over the *candidate union*.
            bm25_raw_map: Dict[int, float] = {int(idx): float(score) for idx, score in (bm25_results or [])}
            dense_raw_map: Dict[int, float] = {int(idx): float(score) for idx, score in (dense_results or [])}

            # Ensure every candidate exists in the maps (missing => 0.0)
            bm25_raw_map_all = {int(idx): float(bm25_raw_map.get(int(idx), 0.0)) for idx in all_indices}
            dense_raw_map_all = {int(idx): float(dense_raw_map.get(int(idx), 0.0)) for idx in all_indices}

            bm25_scaled = _minmax_scale(bm25_raw_map_all)
            dense_scaled = _minmax_scale(dense_raw_map_all)

            for idx in all_indices:
                idx_i = int(idx)
                bm25_s = float(bm25_scaled.get(idx_i, 0.0))
                dense_s = float(dense_scaled.get(idx_i, 0.0))
                combined = (self.bm25_weight * bm25_s) + (self.dense_weight * dense_s)
                combined_scores.append(
                    (
                        idx_i,
                        float(combined),
                        float(bm25_s),
                        float(dense_s),
                        float(bm25_raw_map_all.get(idx_i, 0.0)),
                        float(dense_raw_map_all.get(idx_i, 0.0)),
                    )
                )
        else:
            # Default: RRF fusion over ranks.
            for idx in all_indices:
                # Calculate RRF scores
                bm25_score = 0.0
                if idx in bm25_ranks:
                    bm25_score = 1.0 / (rrf_k + bm25_ranks[idx])

                dense_score = 0.0
                if idx in dense_ranks:
                    dense_score = 1.0 / (rrf_k + dense_ranks[idx])

                # Weighted RRF
                combined = (self.bm25_weight * bm25_score) + (self.dense_weight * dense_score)
                combined_scores.append((int(idx), float(combined), float(bm25_score), float(dense_score), None, None))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for idx, combined, bm25_s, dense_s, bm25_raw, dense_raw in combined_scores[:top_k]:
            results.append({
                'index': idx,
                'title': str(self.titles[idx]),
                'doc_id': self._opt_field(self.doc_ids, idx),
                'source_title': self._opt_field(self.source_titles, idx),
                'entity_title': self._opt_field(self.entity_titles, idx),
                'key_path': str(self.key_paths[idx]),
                'value': self._value_at(idx),
                'score': combined,
                'bm25_score': bm25_s,
                'dense_score': dense_s,
                'fusion_method': fusion_method,
                'bm25_raw_score': bm25_raw,
                'dense_raw_score': dense_raw,
            })
        
        return results
    
    def get_paths_by_title(self, title: str) -> List[Dict]:
        """Get all paths for a given title."""
        results = []
        for i, t in enumerate(self.titles):
            if str(t) == title:
                results.append({
                    'index': i,
                    'title': str(self.titles[i]),
                    'key_path': str(self.key_paths[i]),
                    'value': self._value_at(i)
                })
        return results


async def test_hybrid_search():
    """Test hybrid search functionality."""
    print("="*80)
    print("Testing Hybrid Path Retriever")
    print("="*80)
    
    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6
    )
    
    test_queries = [
        "Who directed The Wolf of Wall Street?",
        "What is the nationality of The Wolf of Wall Street film?",
        "Leonardo DiCaprio actor movies",
        "Argentina education system",
        "airport in Myanmar"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = await retriever.search_hybrid(query, top_k=3)
        
        for i, r in enumerate(results, 1):
            print(f"\n{i}. [Score: {r['score']:.3f}] (BM25: {r['bm25_score']:.3f}, Dense: {r['dense_score']:.3f})")
            print(f"   Title: {r['title']}")
            print(f"   Path: {r['key_path']}")
            print(f"   Value: {r['value'][:80]}..." if len(r['value']) > 80 else f"   Value: {r['value']}")


if __name__ == "__main__":
    asyncio.run(test_hybrid_search())
