"""
FastAPI server wired to the local no-QD RAG pipeline.
"""

from datetime import datetime
import logging
import os
import time
import json
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hybrid_path_retriever import HybridPathRetriever
from llm_logger import init_logger
from llm_provider import create_async_chat_client, detect_provider
from new_multihop_pipeline_paths_hint_expansion import (
    NewMultihopPipelineV12PathsHintExpansion,
    _default_artifact_paths,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="RAG_1 API Server",
    description="Retrieval-Augmented Generation API Server",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


rag_system: Optional[NewMultihopPipelineV12PathsHintExpansion] = None
rag_config: Optional[Dict[str, Any]] = None
indexed_chunks: List[str] = []
startup_time: Optional[datetime] = None


class QueryRequest(BaseModel):
    """RAG query request."""

    query: str = Field(..., description="User question")


class RetrieveRequest(BaseModel):
    """Retrieve-only request."""

    query: str = Field(..., description="Search query")
    top_n: Optional[int] = Field(5, description="Number of documents", ge=1, le=50)


class SourceDocument(BaseModel):
    """Source document."""

    title: str
    content: str
    score: float


class QueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    sources: List[SourceDocument]
    metadata: Dict[str, Any]


class RetrieveResponse(BaseModel):
    """Retrieve-only response."""

    documents: List[SourceDocument]
    retrieval_time: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    rag_initialized: bool
    method: Optional[str] = None
    dataset: Optional[str] = None
    indexed_docs: int
    uptime_seconds: Optional[float] = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return int(default)
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _resolve_artifact_paths(dataset_name: str) -> Dict[str, str]:
    defaults = _default_artifact_paths(dataset_name)
    paths = {
        "data_path": os.getenv("RAG_DATA_PATH", defaults["data_path"]),
        "db_path": os.getenv("RAG_DB_PATH", defaults["db_path"]),
        "bm25_index_path": os.getenv("RAG_BM25_INDEX_PATH", defaults["bm25_index_path"]),
        "embeddings_path": os.getenv("RAG_EMBEDDINGS_PATH", defaults["embeddings_path"]),
    }

    # Local fallback for this workspace: sillok QA file is often named with "_custom".
    if (
        dataset_name == "sillok"
        and not os.path.exists(paths["data_path"])
        and not os.getenv("RAG_DATA_PATH")
    ):
        candidates = [
            "Sillok/sillok_qa_compact_custom.json",
            "Sillok/sillok_qa_compact.json",
            "Sillok/sillok_corpus_for_pipeline.json",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                paths["data_path"] = cand
                break

    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{key} not found: {path}")
    return paths


def _build_source_documents_from_passages(passages: List[Dict[str, Any]], limit: int) -> List[SourceDocument]:
    docs: List[SourceDocument] = []
    for p in passages[: max(0, int(limit))]:
        title = str(p.get("title") or "")
        content = str(p.get("original_passage") or "")
        score = float(p.get("passage_score") or p.get("support_path_score") or 0.0)
        docs.append(
            SourceDocument(
                title=title,
                content=content[:500],
                score=score,
            )
        )
    return docs


def _parse_index_segment(seg: str) -> Optional[int]:
    s = str(seg).strip()
    if s.startswith("[") and s.endswith("]"):
        body = s[1:-1].strip()
        if body.isdigit():
            return int(body)
    return None


def _match_list_item_by_label(items: List[Any], label: str) -> Optional[Any]:
    target = str(label).strip()
    if not target:
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("name", "title", "id", "이름", "명칭"):
            value = item.get(key)
            if value is not None and str(value).strip() == target:
                return item
    return None


def _extract_value_by_key_path(metadata_obj: Any, key_path: str) -> Any:
    if metadata_obj is None:
        return None
    cur = metadata_obj
    if isinstance(cur, dict) and isinstance(cur.get("metadata"), dict):
        cur = cur.get("metadata")

    segments = [seg for seg in str(key_path or "").split(".") if str(seg).strip() != ""]
    for seg in segments:
        if isinstance(cur, dict):
            if seg in cur:
                cur = cur.get(seg)
                continue
            idx = _parse_index_segment(seg)
            if idx is not None and seg in cur:
                cur = cur.get(seg)
                continue
            return None

        if isinstance(cur, list):
            idx = _parse_index_segment(seg)
            if idx is not None:
                if 0 <= idx < len(cur):
                    cur = cur[idx]
                    continue
                return None
            matched = _match_list_item_by_label(cur, seg)
            if matched is None:
                return None
            cur = matched
            continue

        return None
    return cur


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _format_embedding_text_like(title: str, key_path: str, value_text: str) -> str:
    clean_title = str(title or "").strip()
    clean_key = str(key_path or "").strip()
    clean_val = str(value_text or "").strip()
    if clean_title and clean_key and clean_val:
        return f"The {clean_key} of {clean_title} is {clean_val}"
    if clean_title and clean_key:
        return f"The {clean_key} of {clean_title}"
    if clean_key and clean_val:
        return f"{clean_key}: {clean_val}"
    if clean_val:
        return clean_val
    return clean_key


@app.on_event("startup")
async def startup_event():
    """Initialize no-QD pipeline."""

    global rag_system, rag_config, indexed_chunks, startup_time

    startup_time = datetime.now()
    indexed_chunks = []
    rag_config = None
    rag_system = None

    logger.info("Starting API server with no-QD pipeline...")
    load_dotenv()
    init_logger()

    try:
        provider_cfg = detect_provider()
        client = create_async_chat_client(provider_cfg)
    except Exception as e:
        logger.error("Provider initialization failed: %s", str(e))
        return

    dataset_name = str(os.getenv("RAG_DATASET", "sillok")).strip().lower()
    try:
        artifact_paths = _resolve_artifact_paths(dataset_name)
    except Exception as e:
        logger.error("Artifact path resolution failed: %s", str(e))
        return

    try:
        top_k_passages = _env_int("RAG_TOP_K_PASSAGES", 20)
        top_k_paths = _env_int("RAG_TOP_K_PATHS", 50)
        answer_k_passages = _env_int("RAG_ANSWER_K_PASSAGES", top_k_passages)
        answer_k_paths = _env_int("RAG_ANSWER_K_PATHS", 30)
        path_fetch_k = _env_int("RAG_PATH_FETCH_K", 100)

        retriever = HybridPathRetriever(
            bm25_index_path=artifact_paths["bm25_index_path"],
            embeddings_path=artifact_paths["embeddings_path"],
            bm25_weight=_env_float("RAG_BM25_WEIGHT", 1.0),
            dense_weight=_env_float("RAG_DENSE_WEIGHT", 1.3),
        )

        rag_system = NewMultihopPipelineV12PathsHintExpansion(
            client=client,
            retriever=retriever,
            hotpotqa_path=artifact_paths["data_path"],
            db_path=artifact_paths["db_path"],
            top_k_passages=top_k_passages,
            top_k_paths=top_k_paths,
            answer_k_passages=answer_k_passages,
            answer_k_paths=answer_k_paths,
            path_fetch_k=path_fetch_k,
            verbose=bool(_env_int("RAG_VERBOSE", 0)),
            seed_k=_env_int("RAG_SEED_K", 20),
            expansion_k=_env_int("RAG_EXPANSION_K", 10),
            expansion_dense_candidates=_env_int("RAG_EXPANSION_DENSE_CANDIDATES", 500),
            seed_passages_in_final=_env_int("RAG_SEED_PASSAGES_IN_FINAL", 3),
            sq_fusion_method=str(os.getenv("RAG_SQ_FUSION_METHOD", "rrf")).strip().lower(),
        )

        indexed_docs = int(len(getattr(retriever, "titles", [])))
        rag_config = {
            "method_name": "meta",
            "dataset": dataset_name,
            "indexed_docs": indexed_docs,
            "top_k_passages": top_k_passages,
            "top_k_paths": top_k_paths,
            "path_fetch_k": path_fetch_k,
            "chat_model": provider_cfg.chat_model,
            "provider": provider_cfg.provider,
            "artifact_paths": artifact_paths,
        }
        logger.info(
            "Pipeline initialized: dataset=%s provider=%s indexed_docs=%d",
            dataset_name,
            provider_cfg.provider,
            indexed_docs,
        )
    except Exception as e:
        logger.error("Pipeline initialization failed: %s", str(e))
        logger.exception(e)
        rag_system = None
        rag_config = None


@app.on_event("shutdown")
async def shutdown_event():
    global rag_system
    try:
        if rag_system is not None:
            rag_system.close()
    except Exception:
        pass


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""

    return {
        "service": "RAG_1 API Server",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health endpoint."""

    uptime = None
    if startup_time:
        uptime = (datetime.now() - startup_time).total_seconds()

    method = None
    dataset = None
    indexed_docs = 0
    if rag_config:
        method = str(rag_config.get("method_name") or "")
        dataset = str(rag_config.get("dataset") or "")
        indexed_docs = int(rag_config.get("indexed_docs") or 0)

    return HealthResponse(
        status="healthy" if rag_system else "unhealthy",
        rag_initialized=rag_system is not None,
        method=method,
        dataset=dataset,
        indexed_docs=indexed_docs,
        uptime_seconds=uptime,
    )


@app.post("/api/rag/query", response_model=QueryResponse, tags=["RAG"])
async def rag_query(request: QueryRequest):
    """
    Full RAG query (retrieve + generation).

    Endpoint contract is intentionally kept unchanged.
    """

    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Check server logs.")

    try:
        start_time = time.time()
        logger.info("Query received: %s...", request.query[:100])

        result = await rag_system.process_question(request.query)
        if not bool(result.get("success")):
            raise RuntimeError(str(result.get("error") or "Pipeline query failed"))

        answer = str(result.get("final_answer") or result.get("predicted_answer") or "")

        retrieved_passages = result.get("final_retrieved_passages") or []
        retrieved_paths = result.get("final_retrieved_paths") or []
        sources: List[SourceDocument] = []

        # Passages used for answering.
        passage_rows: List[Dict[str, Any]] = []
        for p in retrieved_passages:
            doc_id = p.get("doc_id")
            title = str(p.get("title") or "")
            content = str(p.get("content") or "")
            if (not content) and doc_id is not None:
                try:
                    content = str(rag_system.get_original_passage_by_doc_id(str(doc_id)) or "")
                    if not title:
                        title = str(rag_system.get_title_by_doc_id(str(doc_id)) or "")
                except Exception:
                    content = ""
            score_raw = float(p.get("score") or p.get("passage_score") or p.get("support_path_score") or 0.0)
            passage_rows.append(
                {
                    "title": f"[PASSAGE] {title}" if title else "[PASSAGE]",
                    "content": content,
                    "score_raw": score_raw,
                }
            )

        # Paths used for answering.
        metadata_cache: Dict[str, Any] = {}
        path_rows: List[Dict[str, Any]] = []
        for p in retrieved_paths:
            title = str(p.get("title") or "")
            key_path = str(p.get("key_path") or "")
            value_text = str(p.get("value") or "")
            doc_id = p.get("doc_id")
            origin = str(p.get("origin") or "unknown")
            if (not value_text) and doc_id is not None:
                did = str(doc_id)
                if did not in metadata_cache:
                    try:
                        metadata_cache[did] = rag_system.get_full_metadata(title, did)
                    except Exception:
                        metadata_cache[did] = None
                extracted = _extract_value_by_key_path(metadata_cache.get(did), key_path)
                value_text = _stringify_value(extracted)
            if (not value_text) and title:
                # Fallback by title when doc_id path lookup fails.
                cache_key = f"title::{title}"
                if cache_key not in metadata_cache:
                    try:
                        metadata_cache[cache_key] = rag_system.get_full_metadata(title, None)
                    except Exception:
                        metadata_cache[cache_key] = None
                extracted = _extract_value_by_key_path(metadata_cache.get(cache_key), key_path)
                value_text = _stringify_value(extracted)

            embedding_text_like = _format_embedding_text_like(title, key_path, value_text)
            path_content = f"[origin={origin}] {embedding_text_like}" if origin else embedding_text_like
            score_raw = float(p.get("score") or p.get("bm25_score") or p.get("dense_score") or 0.0)
            path_rows.append(
                {
                    "title": f"[PATH] {title}" if title else "[PATH]",
                    "content": path_content,
                    "score_raw": score_raw,
                }
            )

        # Normalize scores so top item in each group is 1.0.
        max_passage_score = max((float(x["score_raw"]) for x in passage_rows), default=0.0)
        for row in passage_rows:
            s = float(row["score_raw"])
            row["score"] = (s / max_passage_score) if max_passage_score > 0 else s
            sources.append(SourceDocument(title=row["title"], content=row["content"], score=float(row["score"])))

        max_path_score = max((float(x["score_raw"]) for x in path_rows), default=0.0)
        for row in path_rows:
            s = float(row["score_raw"])
            row["score"] = (s / max_path_score) if max_path_score > 0 else s
            sources.append(SourceDocument(title=row["title"], content=row["content"], score=float(row["score"])))

        # Fallback: retrieve directly if answer payload does not include used items.
        if not sources:
            response_top_k = _env_int("RAG_RESPONSE_TOP_K", 5)
            passages, _ = await rag_system.retrieve_for_query_with_limits(
                query=request.query,
                top_k_passages=response_top_k,
                top_k_paths=int((rag_config or {}).get("top_k_paths") or 50),
                path_fetch_k=int((rag_config or {}).get("path_fetch_k") or 100),
            )
            sources = _build_source_documents_from_passages(passages, response_top_k)

        total_time = time.time() - start_time
        logger.info("Query completed in %.2fs", total_time)

        used_passages_count = int(len(retrieved_passages))
        used_paths_count = int(len(retrieved_paths))
        used_contents_count = int(len(sources))

        return QueryResponse(
            answer=answer,
            sources=sources,
            metadata={
                "total_time": total_time,
                "method": "meta",
                "top_k": used_contents_count,
                "used_passages_count": used_passages_count,
                "used_paths_count": used_paths_count,
                "used_contents_count": used_contents_count,
                "path_score_normalization": {
                    "enabled": True,
                    "max_raw_path_score": max_path_score,
                },
                "timestamp": datetime.now().isoformat(),
            },
        )
    except Exception as e:
        logger.error("Query failed: %s", str(e))
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/retrieve", response_model=RetrieveResponse, tags=["RAG"])
async def rag_retrieve(request: RetrieveRequest):
    """
    Retrieve-only endpoint.

    Endpoint contract is intentionally kept unchanged.
    """

    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Check server logs.")

    try:
        start_time = time.time()
        logger.info("Retrieve request: %s...", request.query[:100])

        top_n = int(request.top_n or 5)
        passages, _ = await rag_system.retrieve_for_query_with_limits(
            query=request.query,
            top_k_passages=top_n,
            top_k_paths=max(top_n, int((rag_config or {}).get("top_k_paths") or 50)),
            path_fetch_k=int((rag_config or {}).get("path_fetch_k") or 100),
        )
        documents = _build_source_documents_from_passages(passages, top_n)

        retrieval_time = time.time() - start_time
        logger.info("Retrieved %d documents in %.2fs", len(documents), retrieval_time)

        return RetrieveResponse(documents=documents, retrieval_time=retrieval_time)
    except Exception as e:
        logger.error("Retrieve failed: %s", str(e))
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("RAG_PORT", 8000))
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
