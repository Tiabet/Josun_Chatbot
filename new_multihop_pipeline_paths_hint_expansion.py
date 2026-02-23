#!/usr/bin/env python3
from __future__ import annotations

"""New Multi-hop Pipeline v12 (Paths-as-Hints, no query decomposition).

Runtime flow:
1) one-shot hybrid retrieval with the original question
2) final answer synthesis from retrieved passages + path hints
"""

import asyncio
import json
import os
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from openai import AsyncOpenAI

from hybrid_path_retriever import HybridPathRetriever
from llm_logger import log_llm_call



def _default_artifact_paths(dataset: str) -> Dict[str, str]:
	# Keep this helper local so the v12 runner/test scripts don’t depend on extra modules.
	if dataset == 'musique':
		return {
			'data_path': 'MuSiQue/musique.json',
			'db_path': 'MuSiQue/metadata_v5.db',
			'bm25_index_path': 'MuSiQue/bm25_index_v5',
			'embeddings_path': 'MuSiQue/path_embeddings_v5.npz',
		}
	if dataset == 'hotpot':
		return {
			'data_path': 'HotpotQA/hotpotqa.json',
			'db_path': 'HotpotQA/metadata_v5.db',
			'bm25_index_path': 'HotpotQA/bm25_index_v5',
			'embeddings_path': 'HotpotQA/path_embeddings_v5.npz',
		}
	if dataset == '2wiki':
		return {
			'data_path': '2WikiMultihopQA/2wikimultihopqa.json',
			'db_path': '2WikiMultihopQA/metadata_v5.db',
			'bm25_index_path': '2WikiMultihopQA/bm25_index_v5',
			'embeddings_path': '2WikiMultihopQA/path_embeddings_v5.npz',
		}
	if dataset == 'lveval':
		# Note: LVEVAL has two relevant inputs:
		# - lveval_qa_compact.json: questions (recommended for running the pipeline)
		# - lveval_corpus_for_pipeline.json: QA-like wrapper over corpus (used for indexing)
		return {
			'data_path': 'LVEVAL/lveval_qa_compact.json',
			'db_path': 'LVEVAL/metadata_v5.db',
			'bm25_index_path': 'LVEVAL/bm25_index_v5',
			'embeddings_path': 'LVEVAL/path_embeddings_v5.npz',
		}
	if dataset == 'sillok':
		# Note: SILLOK also has two relevant inputs:
		# - sillok_qa_compact.json: questions (to be added later)
		# - sillok_corpus.json: corpus mapping (idx -> title/text)
		return {
			'data_path': 'Sillok/sillok_qa_compact.json',
			'db_path': 'Sillok/metadata_v5.db',
			'bm25_index_path': 'Sillok/bm25_index_v5',
			'embeddings_path': 'Sillok/path_embeddings_v5.npz',
		}
	raise ValueError(f"Unknown dataset: {dataset}. Expected one of: musique, hotpot, 2wiki, lveval, sillok")


def _json_default(obj):
	"""Best-effort JSON serializer for numpy scalars/arrays and other non-JSON types."""
	# numpy (optional)
	try:
		import numpy as _np  # type: ignore

		if isinstance(obj, (_np.integer,)):
			return int(obj)
		if isinstance(obj, (_np.floating,)):
			return float(obj)
		if isinstance(obj, (_np.ndarray,)):
			return obj.tolist()
	except Exception:
		pass

	# Generic scalar types that expose .item() (covers numpy-like objects)
	if hasattr(obj, 'item'):
		try:
			return obj.item()
		except Exception:
			pass

	# Fallback: string representation
	return str(obj)


class NewMultihopPipelineV11PathsHint:
	"""Pipeline using hybrid retrieval + original passages + top path hints for SQ answering."""

	def __init__(
		self,
		client: AsyncOpenAI,
		retriever: HybridPathRetriever,
		hotpotqa_path: str = 'HotpotQA/hotpotqa.json',
		db_path: str = 'HotpotQA/metadata_v4aligned.db',
		top_k_passages: int = 5,
		top_k_paths: int = 30,
		answer_k_passages: int | None = None,
		answer_k_paths: int | None = None,
		path_fetch_k: int = 50,
		fusion_method: str = 'rrf',
		use_previous_context: bool = True,
		verbose: bool = True,
		chat_model: str | None = None,
	):
		self.client = client
		self.retriever = retriever
		self.db_path = db_path
		# Retrieval/storage counts (used for ranking, logging, and downstream final selection).
		self.top_k_passages = int(top_k_passages)
		self.top_k_paths = int(top_k_paths)

		# Answering counts (what is actually injected into LLM prompts).
		# If not specified, default to retrieval counts.
		self.answer_k_passages = int(self.top_k_passages if answer_k_passages is None else answer_k_passages)
		self.answer_k_paths = int(self.top_k_paths if answer_k_paths is None else answer_k_paths)
		self.fusion_method = str(fusion_method or 'rrf').lower().strip()
		self.use_previous_context = bool(use_previous_context)
		# For top-k UNIQUE paths, we often need to fetch much more than k due to duplicates.
		# Store the effective fetch-k so metadata/logging matches actual behavior.
		self.path_fetch_k_input = path_fetch_k
		self.path_fetch_k = max(path_fetch_k, top_k_paths * 10, top_k_passages * 10, 100)
		self.verbose = verbose

		self.chat_model = (str(chat_model).strip() if chat_model else "")
		if not self.chat_model and self.client is not None:
			from llm_provider import detect_provider
			self.chat_model = detect_provider().chat_model

		self.original_passages, self.doc_id_passages = self._load_passage_indices(hotpotqa_path)
		if self.verbose:
			print(f"[OK] Loaded {len(self.original_passages)} original passages")

		self.conn = sqlite3.connect(db_path)
		self.conn.row_factory = sqlite3.Row

	def close(self) -> None:
		try:
			self.conn.close()
		except Exception:
			pass

	def _load_passage_indices(self, hotpotqa_path: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
		with open(hotpotqa_path, 'r', encoding='utf-8') as f:
			data = json.load(f)

		passages_by_title: Dict[str, str] = {}
		passages_by_doc_id: Dict[str, Dict[str, str]] = {}

		for item in data:
			sample_id = item.get('_id') or item.get('id')

			# MuSiQue-style corpus_idx paragraphs
			if isinstance(item.get('paragraphs'), list):
				paragraphs = item.get('paragraphs') or []
				# Stable ordering if local_idx exists
				def _pkey(p):
					if isinstance(p, dict) and p.get('local_idx') is not None:
						return int(p.get('local_idx'))
					return 10**9

				for p in sorted([p for p in paragraphs if isinstance(p, dict)], key=_pkey):
					title = str(p.get('title') or '')
					text = str(p.get('paragraph_text') or '').strip()
					corpus_idx = p.get('corpus_idx')
					doc_id = str(corpus_idx) if corpus_idx is not None else None

					if title and title not in passages_by_title and text:
						passages_by_title[title] = text
					if doc_id and doc_id not in passages_by_doc_id and text:
						passages_by_doc_id[doc_id] = {"title": title, "text": text}
				continue

			# HotpotQA-style context
			context = item.get('context', []) or []
			for ctx_idx, c in enumerate(context):
				title = None
				sentences: List[str] = []
				doc_id = None

				if isinstance(c, list) and len(c) >= 2:
					title = str(c[0])
					s = c[1]
					if isinstance(s, list):
						sentences = [str(x) for x in s]
					else:
						sentences = [str(s)] if s else []
					if sample_id is not None:
						doc_id = f"{sample_id}::ctx{ctx_idx}"
				elif isinstance(c, dict):
					title = str(c.get('title') or '')
					s = c.get('sentences')
					if isinstance(s, list):
						sentences = [str(x) for x in s]
					else:
						sentences = [str(s)] if s else []

					corpus_idx = c.get('corpus_idx')
					if corpus_idx is not None:
						doc_id = str(corpus_idx)
					elif sample_id is not None:
						# Fallback for legacy dict contexts
						local_idx = c.get('local_idx', ctx_idx)
						doc_id = f"{sample_id}::ctx{int(local_idx)}"

				if not title:
					continue

				full_text = ''.join(sentences).strip() if sentences else ''
				if title not in passages_by_title and full_text:
					passages_by_title[title] = full_text
				if doc_id and doc_id not in passages_by_doc_id and full_text:
					passages_by_doc_id[str(doc_id)] = {"title": title, "text": full_text}

		# LVEVAL special-case: lveval_qa_compact.json contains questions only (no passages).
		# For passage rendering we need to load the global corpus mapping (idx -> title/text).
		if (not passages_by_doc_id) and isinstance(hotpotqa_path, str) and ('lveval_qa_compact' in hotpotqa_path.lower()):
			try:
				qa_path = Path(hotpotqa_path)
				corpus_path = qa_path.parent / 'lveval_corpus.json'
				if corpus_path.exists():
					with corpus_path.open('r', encoding='utf-8') as f:
						corpus = json.load(f)
					if isinstance(corpus, list):
						for doc in corpus:
							if not isinstance(doc, dict):
								continue
							idx = doc.get('idx')
							if idx is None:
								continue
							doc_id = str(int(idx)) if str(idx).isdigit() else str(idx)
							title = str(doc.get('title') or '')
							text = str(doc.get('text') or '').strip()
							if title and title not in passages_by_title and text:
								passages_by_title[title] = text
							if doc_id and doc_id not in passages_by_doc_id and text:
								passages_by_doc_id[doc_id] = {"title": title, "text": text}
			except Exception: pass  # Best-effort fallback; retrieval can still proceed using paths-only.

		# SILLOK special-case: sillok_qa_compact.json(향후)은 질문만 들어갈 수 있으므로
		# passage 렌더링은 sillok_corpus.json(idx->title/text)에 의존하도록 지원
		if (not passages_by_doc_id) and isinstance(hotpotqa_path, str) and ('sillok_qa_compact' in hotpotqa_path.lower()):
			try:
				qa_path = Path(hotpotqa_path)
				corpus_path = qa_path.parent / 'sillok_corpus.json'
				if corpus_path.exists():
					with corpus_path.open('r', encoding='utf-8') as f:
						corpus = json.load(f)
					if isinstance(corpus, list):
						for doc in corpus:
							if not isinstance(doc, dict):
								continue
							idx = doc.get('idx')
							if idx is None:
								continue
							doc_id = str(int(idx)) if str(idx).isdigit() else str(idx)
							title = str(doc.get('title') or '')
							text = str(doc.get('text') or '').strip()
							if title and title not in passages_by_title and text:
								passages_by_title[title] = text
							if doc_id and doc_id not in passages_by_doc_id and text:
								passages_by_doc_id[doc_id] = {"title": title, "text": text}
			except Exception: pass  # Best-effort fallback; retrieval can still proceed using paths-only.

		return passages_by_title, passages_by_doc_id

	def get_original_passage(self, title: str) -> Optional[str]:
		return self.original_passages.get(title)

	def get_original_passage_by_doc_id(self, doc_id: Optional[str]) -> Optional[str]:
		if not doc_id:
			return None
		entry = self.doc_id_passages.get(str(doc_id))
		if entry:
			return entry.get('text')
		return None

	def get_title_by_doc_id(self, doc_id: Optional[str]) -> Optional[str]:
		if not doc_id:
			return None
		entry = self.doc_id_passages.get(str(doc_id))
		if entry:
			return entry.get('title')
		return None

	def get_full_metadata(self, title: str, doc_id: Optional[str] = None) -> Optional[Dict]:
		cursor = self.conn.cursor()

		if doc_id:
			try:
				cursor.execute(
					"SELECT metadata_json FROM metadata WHERE doc_id = ?",
					(doc_id,),
				)
				row = cursor.fetchone()
				if row:
					return json.loads(row['metadata_json'])
			except Exception:
				pass

		try:
			cursor.execute(
				"SELECT metadata_json FROM metadata WHERE title = ?",
				(title,),
			)
			row = cursor.fetchone()
			if row:
				return json.loads(row['metadata_json'])
		except Exception:
			return None

		return None

	async def retrieve_for_query(self, query: str) -> Tuple[List[Dict], List[Dict]]:
		"""Return (top_unique_passages, top_unique_paths_as_hints)."""
		return await self.retrieve_for_query_with_limits(
			query=query,
			top_k_passages=self.top_k_passages,
			top_k_paths=self.top_k_paths,
			path_fetch_k=self.path_fetch_k,
		)

	async def retrieve_for_query_with_limits(
		self,
		query: str,
		top_k_passages: int,
		top_k_paths: int,
		path_fetch_k: int,
	) -> Tuple[List[Dict], List[Dict]]:
		"""Retrieve using explicit limits (used for final-answer override like top-30 paths)."""
		# For top-30 UNIQUE paths, we often need to fetch much more than 30 due to duplicates.
		fetch_k = max(path_fetch_k, top_k_paths * 10, top_k_passages * 10, 100)

		fetched_paths = await self.retriever.search_hybrid(
			query,
			top_k=fetch_k,
			bm25_candidates=max(50, fetch_k),
			dense_candidates=max(50, fetch_k),
			fusion_method=self.fusion_method,
		)

		# 1) Pick top-k UNIQUE paths in ranked order
		seen_path_keys = set()
		top_paths: List[Dict] = []
		value_recovered_count = 0
		value_recover_budget = max(50, int(top_k_paths) * 2, int(top_k_passages) * 2)
		for p in fetched_paths:
			if len(top_paths) >= top_k_paths:
				break
			source_title = p.get('source_title') or p.get('title') or ''
			entity_title = p.get('entity_title') or p.get('title') or ''
			key_path = p.get('key_path', '')
			value = p.get('value', '')
			# If values are not loaded (common for huge corpora), value may be empty for all items.
			# Use index as a disambiguator to avoid collapsing distinct paths.
			idx = p.get('index')
			path_key = (
				str(source_title),
				str(entity_title),
				str(key_path),
				str(value),
				str(idx) if (value == '' or value is None) else '',
			)
			if path_key in seen_path_keys:
				continue
			seen_path_keys.add(path_key)

			# Best-effort: if value is empty (values skipped to avoid OOM), recover it from metadata DB.
			if (
				(not value or not str(value).strip())
				and key_path
				and p.get('doc_id') is not None
				and value_recovered_count < value_recover_budget
			):
				try:
					meta = self.get_full_metadata(str(entity_title), doc_id=str(p.get('doc_id')))
					recovered = self._extract_value_from_metadata(meta, str(key_path)) if meta else ""
					if recovered:
						p['value'] = recovered
						value_recovered_count += 1
				except Exception:
					pass

			top_paths.append(p)

		# 2) Pick top-k unique passages derived from the top-scoring UNIQUE paths (doc_id-based).
		seen_doc_ids = set()
		passages: List[Dict] = []

		# Precompute doc_id -> best path (max score) from the fetched pool.
		best_path_by_doc_id: Dict[str, Dict] = {}
		for p in fetched_paths:
			doc_id = p.get('doc_id')
			if not doc_id:
				continue
			doc_id_str = str(doc_id)
			prev = best_path_by_doc_id.get(doc_id_str)
			if prev is None or (self._safe_score(p) > self._safe_score(prev)):
				best_path_by_doc_id[doc_id_str] = p

		sorted_paths_for_passages = sorted(fetched_paths, key=self._safe_score, reverse=True)
		for path in sorted_paths_for_passages:
			if len(passages) >= top_k_passages:
				break

			doc_id = path.get('doc_id')
			if not doc_id:
				continue
			doc_id_str = str(doc_id)
			if doc_id_str in seen_doc_ids:
				continue
			seen_doc_ids.add(doc_id_str)

			entity_title = path.get('entity_title') or path.get('title')
			source_title = path.get('source_title') or entity_title

			original_passage = self.get_original_passage_by_doc_id(doc_id_str)
			if not original_passage:
				if self.verbose:
					print(f"[WARN] No passage found for doc_id={doc_id_str} (during SQ passage selection)")
				continue
			title_from_doc = self.get_title_by_doc_id(doc_id_str)
			display_title = title_from_doc or str(source_title)
			metadata = self.get_full_metadata(str(entity_title), doc_id=doc_id_str)

			support_path = best_path_by_doc_id.get(doc_id_str) or path
			passage_score = self._safe_score(support_path)

			matched_value = path.get('value')
			if (
				(not matched_value or not str(matched_value).strip())
				and path.get('key_path')
				and value_recovered_count < value_recover_budget
			):
				try:
					recovered = self._extract_value_from_metadata(metadata, str(path.get('key_path'))) if metadata else ""
					if recovered:
						matched_value = recovered
						value_recovered_count += 1
				except Exception:
					pass

			support_path_value = support_path.get('value')
			if (
				(not support_path_value or not str(support_path_value).strip())
				and support_path.get('key_path')
				and value_recovered_count < value_recover_budget
			):
				try:
					recovered = self._extract_value_from_metadata(metadata, str(support_path.get('key_path'))) if metadata else ""
					if recovered:
						support_path_value = recovered
						value_recovered_count += 1
				except Exception:
					pass

			passages.append({
				'title': display_title,
				'source_title': str(source_title),
				'entity_title': str(entity_title),
				'doc_id': doc_id_str,
				'original_passage': original_passage,
				'metadata': metadata,
				'matched_path': path.get('key_path'),
				'matched_value': matched_value,
				'score': path.get('score'),
				'bm25_score': path.get('bm25_score', 0),
				'dense_score': path.get('dense_score', 0),
				'passage_score': float(passage_score),
				'support_path_index': support_path.get('index'),
				'support_path_title': support_path.get('title'),
				'support_path_doc_id': support_path.get('doc_id'),
				'support_path_source_title': support_path.get('source_title'),
				'support_path_entity_title': support_path.get('entity_title'),
				'support_path_key_path': support_path.get('key_path'),
				'support_path_value': support_path_value,
				'support_path_score': support_path.get('score'),
			})

		if self.verbose and value_recovered_count:
			print(f"[INFO] Recovered {value_recovered_count} empty path values from metadata_json")

		return passages, top_paths

	def _build_simple_previous_context(self, current_sq: 'SubQuestion', decomposition: 'QueryDecomposition') -> str:
		"""Build previous SQ context for answering."""
		if not current_sq.depends_on:
			return ""

		visited = set()
		ordered_dep_ids: List[str] = []

		def dfs(sq_id: str) -> None:
			if sq_id in visited:
				return
			visited.add(sq_id)
			sq = decomposition.get_subquestion(sq_id)
			if not sq:
				return
			for parent_id in (sq.depends_on or []):
				dfs(parent_id)
			ordered_dep_ids.append(sq_id)

		for dep_id in (current_sq.depends_on or []):
			dfs(dep_id)

		context_parts: List[str] = []
		for dep_id in ordered_dep_ids:
			dep_sq = decomposition.get_subquestion(dep_id)
			if dep_sq and getattr(dep_sq, 'answer', None):
				context_parts.append(f"{dep_id}: {dep_sq.question}")
				context_parts.append(f"Answer: {dep_sq.answer}")
				context_parts.append("")

		if context_parts:
			return "Previous Sub-Questions:\n" + "\n".join(context_parts)
		return ""

	@staticmethod
	def _path_dedupe_key(p: Dict) -> Tuple[str, str, str, str, str]:
		source_title = p.get('source_title') or p.get('title') or ''
		entity_title = p.get('entity_title') or p.get('title') or ''
		key_path = p.get('key_path', '')
		value = p.get('value', '')
		idx = p.get('index')
		return (
			str(source_title),
			str(entity_title),
			str(key_path),
			str(value),
			str(idx) if (value == '' or value is None) else '',
		)

	def _collect_all_unique_paths(self, decomposition: 'QueryDecomposition') -> List[Dict]:
		seen = set()
		unique_paths: List[Dict] = []

		for sq in decomposition.subquestions:
			paths = getattr(sq, 'retrieved_paths', None)
			if not paths:
				continue
			for p in paths:
				key = self._path_dedupe_key(p)
				if key in seen:
					continue
				seen.add(key)
				unique_paths.append(p)

		return unique_paths

	@staticmethod
	def _safe_score(p: Dict) -> float:
		try:
			s = p.get('score', None)
			return float(s) if s is not None else float('-inf')
		except Exception:
			return float('-inf')

	@staticmethod
	def _format_paths_as_hints(paths: List[Dict]) -> str:
		if not paths:
			return "(No paths.)"

		lines = []
		for i, p in enumerate(paths, 1):
			entity_title = p.get('entity_title') or p.get('title') or ''
			key_path = p.get('key_path', '')
			value = p.get('value', '')

			pretty_value = value
			if isinstance(pretty_value, str):
				s = pretty_value.strip()
				if s and ((s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']'))):
					try:
						obj = json.loads(s)
						if isinstance(obj, dict) and len(obj) == 1:
							if 'target' in obj:
								obj = obj.get('target')
							elif '대상' in obj:
								obj = obj.get('대상')
						pretty_value = json.dumps(obj, ensure_ascii=False)
					except Exception:
						pretty_value = pretty_value

			if isinstance(pretty_value, str) and len(pretty_value) > 10000:
				pretty_value = pretty_value[:10000] + "..."

			lines.append(
				f"[{i}] entity_title: {entity_title}\n"
				f"  key_path: {key_path}\n"
				f"  value: {pretty_value}"
			)
		return "\n".join(lines)

	@staticmethod
	def _extract_value_from_metadata(metadata: Dict, key_path: str) -> str:
		"""Best-effort value extraction from metadata_json using embedding key_path."""
		if not isinstance(metadata, dict):
			return ""
		kp = str(key_path or "").strip()
		if not kp:
			return ""

		inner = metadata.get('metadata', metadata)
		if not isinstance(inner, dict):
			return ""

		root = inner.get('attributes') if isinstance(inner.get('attributes'), dict) else inner
		segs = [s for s in kp.split('.') if s != ""]
		obj = root
		for seg in segs:
			if isinstance(obj, dict):
				if seg in obj:
					obj = obj.get(seg)
					continue
				# Fallback: some keys may exist at inner root even if attributes exist.
				if obj is not inner and seg in inner:
					obj = inner.get(seg)
					continue
				return ""
			if isinstance(obj, list):
				# Numeric list indices like "[0]"
				if seg.startswith('[') and seg.endswith(']'):
					try:
						j = int(seg[1:-1])
						obj = obj[j]
						continue
					except Exception:
						return ""
				# Labeled list-of-dict items (name/title/id)
				found = None
				for it in obj:
					if not isinstance(it, dict):
						continue
					for k in ('name', 'title', 'id'):
						v = it.get(k)
						if isinstance(v, str) and v.strip() == seg:
							found = it
							break
					if found is not None:
						break
				if found is None:
					return ""
				obj = found
				continue
			return ""

		if obj is None:
			return ""
		if isinstance(obj, (dict, list)):
			try:
				return json.dumps(obj, ensure_ascii=False)
			except Exception:
				return str(obj)
		return str(obj)

	@staticmethod
	def _format_passages_original(passages: List[Dict]) -> str:
		if not passages:
			return "No passages retrieved."

		passage_texts = []
		for i, p in enumerate(passages, 1):
			title = p.get('title', '')
			original_text = p.get('original_passage', '')

			if original_text:
				passage_texts.append(f"[{i}] {title}\n{original_text}")
			else:
				if p.get('metadata'):
					metadata = p['metadata']
					parts = [f"[{i}] {title}"]
					excluded_keys = {'title'}
					for key, value in metadata.items():
						if key in excluded_keys or not value:
							continue
						value_str = str(value) if not isinstance(value, (dict, list)) else json.dumps(value, ensure_ascii=False)
						parts.append(f"  {key}: {value_str}")
					passage_texts.append("\n".join(parts))
				else:
					passage_texts.append(f"[{i}] {title}\n(No content available)")

		return "\n\n".join(passage_texts)

	async def generate_answer(
		self,
		question: str,
		passages: List[Dict],
		top_paths: List[Dict],
		previous_context: str,
		main_query: str,
		is_final_sq: bool = False,
	) -> str:
		from Prompt.answer_prompt import SUBQUESTION_ANSWERING_PROMPT

		passages_text = self._format_passages_original(passages)
		paths_text = self._format_paths_as_hints(top_paths)

		combined_info = (
			"---Top Retrieved Metadata Paths (STRONG HINTS)---\n"
			"The paths below are strong hints for where the answer might be found. "
			"Use them to focus your reading of the passages, but do NOT treat them as guaranteed truth.\n\n"
			f"{paths_text}\n\n"
			"---Top Passages from High-Score Paths (TOP-5 by doc_id)---\n"
			f"{passages_text}"
		)

		prompt = SUBQUESTION_ANSWERING_PROMPT.replace("{{subquestion}}", question)
		prompt = prompt.replace("{{passages}}", combined_info)
		prompt = prompt.replace("{{previous_context}}", previous_context if previous_context else "None")

		response = await self.client.chat.completions.create(
			model=self.chat_model,
			messages=[
				{
					"role": "system",
					"content": (
						"You are a historian-style question answering system. "
						"Answer in Korean with a short, direct answer (1-2 sentences). "
						"Be historically grounded and specific. "
						"Do not output a fixed fallback phrase like 'Insufficient information.'"
					),
				},
				{"role": "user", "content": prompt},
			],
			temperature=0.0,
			max_tokens=400,
		)

		answer_raw = (response.choices[0].message.content or '').strip()

		log_llm_call(
			call_type="Subquestion Answering (V11-PathsHint)",
			input_text=prompt,
			output_text=answer_raw,
			context={
				"question": question,
				"main_query": main_query,
				"is_final_sq": is_final_sq,
				"num_passages": len(passages),
				"num_paths": len(top_paths),
			},
		)

		if answer_raw.startswith("Answer:"):
			return answer_raw[7:].strip()
		return answer_raw

	def _select_top_paths_and_passages_from_decomposition(
		self,
		decomposition: 'QueryDecomposition',
		top_paths_k: int = 30,
		top_passages_k: int = 5,
	) -> Tuple[List[Dict], List[Dict]]:
		all_paths = self._collect_all_unique_paths(decomposition)
		sorted_paths = sorted(all_paths, key=self._safe_score, reverse=True)
		top_paths = sorted_paths[:top_paths_k]

		top_path_passages: List[Dict] = []
		seen_doc_ids = set()
		for p in sorted_paths:
			if len(top_path_passages) >= top_passages_k:
				break
			doc_id = p.get('doc_id')
			if not doc_id:
				continue
			doc_id_str = str(doc_id)
			if doc_id_str in seen_doc_ids:
				continue
			passage_text = self.get_original_passage_by_doc_id(doc_id_str)
			if not passage_text:
				if self.verbose:
					print(f"[WARN] No passage found for doc_id={doc_id_str} (from high-score path)")
				continue
			seen_doc_ids.add(doc_id_str)
			title_from_doc = self.get_title_by_doc_id(doc_id_str) or (p.get('source_title') or p.get('title') or '')
			top_path_passages.append({
				'title': str(title_from_doc),
				'doc_id': doc_id_str,
				'original_passage': passage_text,
				'metadata': None,
			})

		return top_paths, top_path_passages

	async def answer_subquestion(
		self,
		sq: 'SubQuestion',
		decomposition: 'QueryDecomposition',
		is_final_sq: bool = False,
	) -> Dict:
		t0_total = time.perf_counter()
		retrieval_s = 0.0
		llm_s = 0.0
		try:
			# QD dependency removed: keep subquestion text as-is.
			actual_question = sq.question
			setattr(sq, 'actual_question', actual_question)
			previous_context = (
				self._build_simple_previous_context(sq, decomposition)
				if bool(getattr(self, 'use_previous_context', True))
				else ""
			)

			t0 = time.perf_counter()
			passages, top_paths = await self.retrieve_for_query(actual_question)
			retrieval_s = time.perf_counter() - t0

			if self.verbose:
				print(f"\n   Retrieved {len(passages)} passages + {len(top_paths)} paths")

			answer_passages = list(passages or [])[: max(0, int(getattr(self, 'answer_k_passages', len(passages or []))))]
			answer_paths = list(top_paths or [])[: max(0, int(getattr(self, 'answer_k_paths', len(top_paths or []))))]

			t1 = time.perf_counter()
			answer = await self.generate_answer(
				actual_question,
				answer_passages,
				answer_paths,
				previous_context,
				decomposition.main_query,
				is_final_sq=is_final_sq,
			)
			llm_s = time.perf_counter() - t1

			sq.answer = answer
			sq.retrieved_passages = passages
			sq.retrieved_paths = top_paths

			support_paths: List[Dict] = []
			try:
				for ps in (passages or []):
					if not isinstance(ps, dict):
						continue
					sp_doc_id = ps.get('support_path_doc_id')
					sp_key = ps.get('support_path_key_path')
					sp_val = ps.get('support_path_value')
					if sp_doc_id is None or sp_key is None or sp_val is None:
						continue
					support_paths.append(
						{
							'index': ps.get('support_path_index'),
							'title': ps.get('support_path_title') or ps.get('title'),
							'doc_id': str(sp_doc_id),
							'source_title': ps.get('support_path_source_title') or ps.get('source_title') or ps.get('title'),
							'entity_title': ps.get('support_path_entity_title') or ps.get('entity_title') or ps.get('title'),
							'key_path': str(sp_key),
							'value': str(sp_val),
							'score': ps.get('support_path_score', ps.get('passage_score', ps.get('score'))),
							'origin': 'passage_support',
						}
					)
			except Exception:
				support_paths = []

			setattr(sq, 'support_paths_for_passages', support_paths)

			elapsed = time.perf_counter() - t0_total
			return {
				'success': True,
				'answer': answer,
				'retrieved_passages': passages,
				'retrieved_paths': top_paths,
				'time': elapsed,
				'timing': {
					'total_s': elapsed,
					'retrieval_s': retrieval_s,
					'llm_s': llm_s,
				},
			}
		except Exception as e:
			elapsed = time.perf_counter() - t0_total
			return {
				'success': False,
				'error': str(e),
				'time': elapsed,
				'timing': {
					'total_s': elapsed,
					'retrieval_s': retrieval_s,
					'llm_s': llm_s,
				},
			}


class NewMultihopPipelineV12PathsHintExpansion(NewMultihopPipelineV11PathsHint):
	# Default seed/rerank split policy for ablations.
	# These are the EXACT ratios requested for the 2wiki grid experiments.
	_PASSAGE_SEED_RERANK = {
		0: (0, 0),
		3: (2, 1),
		5: (3, 2),
		10: (6, 4),
		20: (12, 8),
	}
	_PATH_SEED_RERANK = {
		0: (0, 0),
		10: (6, 4),
		30: (20, 10),
		50: (30, 20),
	}

	def _split_seed_rerank(self, total_k: int, mapping: Dict[int, Tuple[int, int]], fallback_seed: int) -> Tuple[int, int]:
		"""Return (seed_k, rerank_k) for a requested total.

		- Uses the fixed mapping when available.
		- Otherwise falls back to (min(fallback_seed, total_k), total_k - seed).
		"""
		k = max(0, int(total_k))
		if k in mapping:
			s, r = mapping[k]
			return max(0, int(s)), max(0, int(r))
		seed = max(0, min(int(fallback_seed), k))
		rerank = max(0, k - seed)
		return seed, rerank
	def __init__(
		self,
		*args,
		seed_k: int = 20,
		expansion_k: int = 10,
		expansion_dense_candidates: int = 500,
		expansion_rrf_k: int = 60,
		seed_reinforce_weight: float = 1.0,
		seed_passages_in_final: int = 3,
		final_rerank_mode: str = 'minmax',
		sq_fusion_method: str = 'rrf',
		final_selection_mode: str = 'rerank',
		**kwargs,
	):
		super().__init__(*args, **kwargs)
		# SQ retrieval fusion (ablation toggle): rrf/minmax/bm25/dense
		self.fusion_method = str(sq_fusion_method or 'rrf').lower().strip()
		self.seed_k = int(seed_k)
		self.expansion_k = int(expansion_k)
		self.expansion_dense_candidates = int(expansion_dense_candidates)
		self.expansion_rrf_k = int(expansion_rrf_k)
		self.seed_reinforce_weight = float(seed_reinforce_weight)
		self.seed_passages_in_final = int(seed_passages_in_final)
		# Final rerank is fixed (no CLI flag): hybrid bm25+cosine (SQ-max pooling) + min-max scaling.
		self.final_rerank_mode = 'minmax'
		# Final selection mode (ablation):
		# - rerank: seed + rerank (default)
		# - rrf_only: skip rerank; fill using SQ path scores only
		self.final_selection_mode = str(final_selection_mode or 'rerank').lower().strip()

	def _path_from_retriever_index(self, idx: int, score: float, origin: str) -> Dict:
		"""Build a path dict from the underlying retriever arrays."""
		r = self.retriever
		title = str(r.titles[idx])
		return {
			'index': int(idx),
			'title': title,
			'doc_id': r._opt_field(r.doc_ids, idx),
			'source_title': r._opt_field(r.source_titles, idx),
			'entity_title': r._opt_field(r.entity_titles, idx),
			'key_path': str(r.key_paths[idx]),
			'value': (r._value_at(idx) if hasattr(r, '_value_at') else ''),
			'score': float(score),
			'origin': origin,
		}

	def _second_stage_rrf_scores(self, seed_paths: List[Dict], candidate_indices: List[int]) -> Dict[int, float]:
		"""Compute RRF aggregation scores over dense-neighbor ranks for each seed path.

		IMPORTANT (v12 restricted expansion):
		- We only consider candidates that appeared during SQ answering (i.e., paths in decomposition).
		- This avoids introducing brand-new paths from the global index at final time.
		"""
		r = self.retriever
		if not getattr(r, 'get_normalized_embeddings', None):
			return {}
		try:
			n = int(len(r.titles))
		except Exception:
			n = 0
		if n <= 0:
			return {}

		scores: Dict[int, float] = {}

		seed_indices: List[int] = []
		for p in seed_paths:
			try:
				seed_indices.append(int(p.get('index')))
			except Exception:
				continue

		seed_indices = [i for i in seed_indices if 0 <= i < n]
		if not seed_indices:
			return {}

		cand = []
		seen = set()
		for i in candidate_indices:
			try:
				j = int(i)
			except Exception:
				continue
			if (j < 0) or (j >= n) or (j in seen):
				continue
			seen.add(j)
			cand.append(j)
		if not cand:
			return {}

		rrf_k = float(self.expansion_rrf_k)
		top_m = max(1, int(self.expansion_dense_candidates))

		# Candidate-only dense similarity using memmap-backed embeddings.
		cand_emb = r.get_normalized_embeddings(cand)
		seed_emb = r.get_normalized_embeddings(seed_indices)
		# Each seed path votes for similar paths by rank (dense cosine similarity),
		# restricted to candidate indices derived from SQ retrieval.
		for seed_pos, seed_idx in enumerate(seed_indices):
			seed_vec = seed_emb[int(seed_pos)]
			sims = np.dot(cand_emb, seed_vec)

			# Get top neighbors among candidates (include more than needed; self-match removed below)
			k_take = min(int(len(cand)), int(top_m) + 1)
			top_pos = np.argsort(sims)[::-1][:k_take]

			rank = 0
			for pos in top_pos:
				cand_idx = int(cand[int(pos)])
				if cand_idx == seed_idx:
					continue  # exclude self-match
				# RRF uses 1/(k + rank). rank is 1-based.
				rank += 1
				scores[cand_idx] = scores.get(cand_idx, 0.0) + (1.0 / (rrf_k + float(rank)))

		return scores

	def _select_final_passages_from_paths(
		self,
		seed_paths: List[Dict],
		expansion_paths: List[Dict],
		top_passages_k: int,
	) -> List[Dict]:
		"""Pick final original passages (doc_id unique) from seed/expansion path sets."""
		seed_docs: Dict[str, Dict] = {}
		exp_docs: Dict[str, Dict] = {}

		def _add(bucket: Dict[str, Dict], p: Dict):
			doc_id = p.get('doc_id')
			if not doc_id:
				return
			doc_id = str(doc_id)
			s = self._safe_score(p)
			entry = bucket.get(doc_id)
			if entry is None:
				bucket[doc_id] = {
					'doc_id': doc_id,
					'max_score': s,
					'count': 1,
				}
			else:
				entry['count'] = int(entry.get('count', 0)) + 1
				if s > float(entry.get('max_score', float('-inf'))):
					entry['max_score'] = s

		for p in seed_paths:
			_add(seed_docs, p)
		for p in expansion_paths:
			_add(exp_docs, p)

		def _doc_rank(items: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
			def _score(kv):
				d = kv[1]
				return float(d.get('max_score', float('-inf'))) + 0.2 * float(d.get('count', 0))

			return sorted(items.items(), key=_score, reverse=True)

		seed_ranked = _doc_rank(seed_docs)
		exp_ranked = _doc_rank(exp_docs)

		wanted_seed = max(0, min(int(self.seed_passages_in_final), int(top_passages_k)))
		wanted_exp = max(0, int(top_passages_k) - wanted_seed)

		chosen_doc_ids: List[str] = []

		seed_chosen = 0
		for doc_id, _ in seed_ranked:
			if seed_chosen >= wanted_seed:
				break
			if doc_id in chosen_doc_ids:
				continue
			chosen_doc_ids.append(doc_id)
			seed_chosen += 1

		exp_chosen = 0
		for doc_id, _ in exp_ranked:
			if exp_chosen >= wanted_exp:
				break
			if doc_id in chosen_doc_ids:
				continue
			chosen_doc_ids.append(doc_id)
			exp_chosen += 1

		if len(chosen_doc_ids) < int(top_passages_k):
			for doc_id, _ in seed_ranked + exp_ranked:
				if len(chosen_doc_ids) >= int(top_passages_k):
					break
				if doc_id in chosen_doc_ids:
					continue
				chosen_doc_ids.append(doc_id)

		passages: List[Dict] = []
		for doc_id in chosen_doc_ids[: int(top_passages_k)]:
			passage_text = self.get_original_passage_by_doc_id(doc_id)
			if not passage_text:
				continue
			title = self.get_title_by_doc_id(doc_id) or ''
			passages.append(
				{
					'title': str(title),
					'doc_id': str(doc_id),
					'original_passage': passage_text,
					'metadata': None,
				}
			)
			if len(passages) >= int(top_passages_k):
				break

		return passages

	def _score_candidate_indices_hybrid(
		self,
		query: str,
		candidate_indices: List[int],
		fusion_method: str = 'rrf',
	) -> Dict[int, float]:
		"""Score a restricted set of path indices against a query.

		We reuse the existing HybridPathRetriever logic.
		- rrf: uses score_candidates_rrf (rank fusion within candidates)
		- minmax: uses the same scoring core as search_hybrid, but restricted to candidates
		"""
		# Dedupe + sanitize
		seen = set()
		cands: List[int] = []
		for i in candidate_indices:
			try:
				idx = int(i)
			except Exception:
				continue
			if idx in seen:
				continue
			seen.add(idx)
			cands.append(idx)

		if not cands:
			return {}

		fusion_method = (fusion_method or 'rrf').strip().lower()
		if fusion_method == 'minmax':
			# search_hybrid supports minmax but returns only top_k; we need a full map.
			# We approximate by calling score_candidates_rrf over candidates (still rank-based)
			# if embed client is missing or candidates are small.
			# If you want true minmax over raw scores, we can extend the retriever later.
			fusion_method = 'rrf'

		# This synchronous wrapper is intentionally conservative.
		# In typical pipeline execution we are already in an event loop; use the async variant.
		raise RuntimeError("Use _score_candidate_indices_hybrid_async inside the pipeline")

	async def _score_candidate_indices_hybrid_async(
		self,
		query: str,
		candidate_indices: List[int],
		fusion_method: str = 'rrf',
	) -> Dict[int, float]:
		"""Async variant used inside pipeline execution."""
		seen = set()
		cands: List[int] = []
		for i in candidate_indices:
			try:
				idx = int(i)
			except Exception:
				continue
			if idx in seen:
				continue
			seen.add(idx)
			cands.append(idx)
		if not cands:
			return {}
		fusion_method = (fusion_method or 'rrf').strip().lower()
		if fusion_method == 'minmax':
			fusion_method = 'rrf'
		scored = await self.retriever.score_candidates_rrf(query, cands, top_k=len(cands))
		return {
			int(p['index']): float(p.get('score', 0.0))
			for p in scored
			if isinstance(p, dict) and p.get('index') is not None
		}

	def _passage_from_doc_id(self, doc_id: str) -> Optional[Dict]:
		passage_text = self.get_original_passage_by_doc_id(doc_id)
		if not passage_text:
			return None
		title = self.get_title_by_doc_id(doc_id) or ''
		return {
			'title': str(title),
			'doc_id': str(doc_id),
			'original_passage': passage_text,
			'metadata': None,
		}

	def _select_seed_passage_doc_ids_from_paths(
		self,
		all_paths: List[Dict],
		k: int,
	) -> List[str]:
		"""Seed passages: top-k doc_ids by max(path.score) over all SQ-observed paths."""
		best: Dict[str, float] = {}
		for p in all_paths:
			if not isinstance(p, dict):
				continue
			doc_id = p.get('doc_id')
			if doc_id is None:
				continue
			doc_id_str = str(doc_id)
			s = self._safe_score(p)
			prev = best.get(doc_id_str)
			if (prev is None) or (s > float(prev)):
				best[doc_id_str] = float(s)

		ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
		return [doc_id for doc_id, _ in ranked[: max(0, int(k))]]

	def _select_seed_paths(
		self,
		all_paths: List[Dict],
		k: int,
	) -> List[Dict]:
		"""Seed paths: top-k unique paths by path.score."""
		sorted_paths = sorted(all_paths, key=self._safe_score, reverse=True)
		seen = set()
		out: List[Dict] = []
		for p in sorted_paths:
			if not isinstance(p, dict):
				continue
			key = self._path_dedupe_key(p)
			if key in seen:
				continue
			seen.add(key)
			p2 = dict(p)
			p2['origin'] = 'seed'
			out.append(p2)
			if len(out) >= int(k):
				break
			
		return out

	def _collect_all_unique_paths_with_support(self, decomposition) -> List[Dict]:
		"""Collect unique paths from SQs, including passage-support paths (additive).

		This keeps the final path pool aligned with the doc_ids that actually formed
		SQ passages (which can be sourced from the larger fetch_k pool).
		"""
		seen = set()
		unique_paths: List[Dict] = []

		for sq in decomposition.subquestions:
			paths = []
			try:
				paths.extend(getattr(sq, 'retrieved_paths', None) or [])
			except Exception:
				pass
			try:
				paths.extend(getattr(sq, 'support_paths_for_passages', None) or [])
			except Exception:
				pass

			for p in paths:
				if not isinstance(p, dict):
					continue
				key = self._path_dedupe_key(p)
				if key in seen:
					continue
				seen.add(key)
				unique_paths.append(p)

		return unique_paths

	def _select_final_passages_from_scored_sq_passages(self, decomposition, top_passages_k: int) -> List[Dict]:
		"""Select final passages from SQ-collected passages, ranked by passage_score.

		A-proxy: passage_score is derived from best supporting path score per doc_id.
		"""
		all_passages = self._collect_all_unique_passages(decomposition)
		by_doc: Dict[str, Dict] = {}

		for p in all_passages:
			if not isinstance(p, dict):
				continue
			doc_id = p.get('doc_id')
			if not doc_id:
				continue
			doc_id_str = str(doc_id)
			try:
				score = float(p.get('passage_score') if p.get('passage_score') is not None else p.get('score'))
			except Exception:
				score = float('-inf')

			entry = by_doc.get(doc_id_str)
			if entry is None:
				by_doc[doc_id_str] = {
					'doc_id': doc_id_str,
					'max_score': score,
					'count': 1,
				}
			else:
				entry['count'] = int(entry.get('count', 0)) + 1
				if score > float(entry.get('max_score', float('-inf'))):
					entry['max_score'] = score

		def _rank(kv):
			d = kv[1]
			return float(d.get('max_score', float('-inf'))) + 0.2 * float(d.get('count', 0))

		ranked = sorted(by_doc.items(), key=_rank, reverse=True)
		chosen_doc_ids = [doc_id for doc_id, _ in ranked[: int(top_passages_k)]]

		passages: List[Dict] = []
		for doc_id in chosen_doc_ids:
			passage_text = self.get_original_passage_by_doc_id(doc_id)
			if not passage_text:
				continue
			title = self.get_title_by_doc_id(doc_id) or ''
			passages.append(
				{
					'title': str(title),
					'doc_id': str(doc_id),
					'original_passage': passage_text,
					'metadata': None,
				}
			)
			if len(passages) >= int(top_passages_k):
				break

		return passages

	async def _select_top_paths_and_passages_from_decomposition(
		self,
		decomposition,
		top_paths_k: int = 30,
		top_passages_k: int = 5,
	) -> Tuple[List[Dict], List[Dict]]:
		"""Final selection = seed + rerank (simplified).

		Policy (current implementation):
		- Split the requested totals (top_paths_k / top_passages_k) into (seed_k, rerank_k)
		  using fixed mapping tables for ablations (see _PASSAGE_SEED_RERANK / _PATH_SEED_RERANK).
		- Seed passages: top doc_ids by max(SQ path.score)
		- Seed paths: top unique paths by SQ path.score
		- Build rerank pools from SQ-observed paths/doc_ids, then rerank with hybrid scores.
		- Final outputs contain up to top_passages_k unique passages and up to top_paths_k unique paths.
		"""
		main_query = getattr(decomposition, 'main_query', None) or ''
		all_paths = self._collect_all_unique_paths_with_support(decomposition)

		top_paths_k = max(0, int(top_paths_k))
		top_passages_k = max(0, int(top_passages_k))

		# Ablation: no reranking. Use SQ-derived scores only (typically RRF scores).
		if str(getattr(self, 'final_selection_mode', 'rerank')).lower().strip() in ('rrf_only', 'rrf', 'no_rerank'):
			sorted_paths = sorted(all_paths, key=self._safe_score, reverse=True)
			final_paths: List[Dict] = []
			seen_path_keys = set()
			for p in sorted_paths:
				if len(final_paths) >= int(top_paths_k):
					break
				if not isinstance(p, dict):
					continue
				k = self._path_dedupe_key(p)
				if k in seen_path_keys:
					continue
				seen_path_keys.add(k)
				p2 = dict(p)
				p2['origin'] = p2.get('origin') or 'rrf_only'
				final_paths.append(p2)

			final_passages: List[Dict] = []
			seen_doc = set()
			# Fill passages by unique doc_id using best supporting path score.
			for p in sorted_paths:
				if len(final_passages) >= int(top_passages_k):
					break
				if not isinstance(p, dict):
					continue
				doc_id = p.get('doc_id')
				if doc_id is None:
					continue
				doc_id_str = str(doc_id)
				if doc_id_str in seen_doc:
					continue
				original_passage = self.get_original_passage_by_doc_id(doc_id_str)
				if not original_passage:
					continue
				seen_doc.add(doc_id_str)
				title_from_doc = self.get_title_by_doc_id(doc_id_str) or (p.get('source_title') or p.get('title') or '')
				final_passages.append(
					{
						'title': str(title_from_doc),
						'doc_id': doc_id_str,
						'original_passage': original_passage,
						'metadata': None,
						'passage_score': float(self._safe_score(p)),
						'support_path_score': float(self._safe_score(p)),
						'support_path_origin': 'rrf_only',
					}
				)

			return final_paths[: int(top_paths_k)], final_passages

		# Seed/rerank splits are defined over the *answering* counts.
		# Retrieval/collection (RRF etc.) remains unchanged; we only control how many
		# passages/paths are selected into the final answering context.
		seed_passage_k, rerank_passage_k = self._split_seed_rerank(
			total_k=top_passages_k,
			mapping=self._PASSAGE_SEED_RERANK,
			fallback_seed=int(getattr(self, 'seed_passages_in_final', 3) or 3),
		)
		seed_path_k, rerank_path_k = self._split_seed_rerank(
			total_k=top_paths_k,
			mapping=self._PATH_SEED_RERANK,
			fallback_seed=int(getattr(self, 'seed_k', 20) or 20),
		)

		# 1) Seed selection
		seed_passage_doc_ids = self._select_seed_passage_doc_ids_from_paths(all_paths, k=seed_passage_k)
		seed_paths = self._select_seed_paths(all_paths, k=seed_path_k)
		seed_path_keys = {self._path_dedupe_key(p) for p in seed_paths}

		# 2) Remaining passage pool (paths-based, not passages-based)
		# MuSiQue note: gold may be absent from SQ-level retrieved_passages(top-5),
		# while retrieved_paths often contains the gold doc_id. Restricting rerank
		# candidates to "passages observed in SQ" can reduce recall, so we expand
		# the rerank candidate doc_ids to all doc_ids observed in SQ paths.
		doc_sq_max: Dict[str, float] = {}
		for p in all_paths:
			if not isinstance(p, dict):
				continue
			doc_id = p.get('doc_id')
			if doc_id is None:
				continue
			doc_id_str = str(doc_id)
			sc = self._safe_score(p)
			prev = doc_sq_max.get(doc_id_str)
			if (prev is None) or (float(sc) > float(prev)):
				doc_sq_max[doc_id_str] = float(sc)

		seen_doc = set(seed_passage_doc_ids)
		ranked_remaining_docs = sorted(doc_sq_max.items(), key=lambda kv: kv[1], reverse=True)
		remaining_doc_ids: List[str] = []
		for doc_id_str, _ in ranked_remaining_docs:
			if doc_id_str in seen_doc:
				continue
			seen_doc.add(doc_id_str)
			remaining_doc_ids.append(doc_id_str)

		# 3) Collect candidate paths restricted to remaining passages
		remaining_doc_set = set(remaining_doc_ids)
		paths_by_doc: Dict[str, List[Dict]] = {}
		candidate_indices: List[int] = []
		for p in all_paths:
			if not isinstance(p, dict):
				continue
			doc_id = p.get('doc_id')
			if doc_id is None:
				continue
			doc_id_str = str(doc_id)
			if doc_id_str not in remaining_doc_set:
				continue
			idx = p.get('index')
			if idx is None:
				continue
			try:
				idx_i = int(idx)
			except Exception:
				continue
			paths_by_doc.setdefault(doc_id_str, []).append(p)
			candidate_indices.append(idx_i)

		# 4) Rerank scoring: bm25 + cosine, both SQ-max pooled, then min-max scaled and combined
		# score(path) = 1.0 * bm25_scaled + 1.3 * cosine_scaled
		idx_score_map: Dict[int, float] = {}
		# If the requested rerank sizes are 0, skip dense/bm25 scoring entirely.
		if candidate_indices and (int(rerank_path_k) > 0 or int(rerank_passage_k) > 0):
			r = self.retriever
			has_dense = bool(getattr(r, 'has_dense_rerank', None)) and bool(r.has_dense_rerank())
			if not has_dense:
				raise RuntimeError(
					'Dense reranking is required but not available. '
					'Check that an embedding API key is set and embeddings memmap exists. '
					'Example: set EMBEDDINGS_MEMMAP_PATH=Sillok/path_embeddings_v5.ckpt/embeddings.npy'
				)

			# Prefer SQ-level queries (NaiveRAG-QD style). Fall back to main_query if SQs are missing.
			sq_queries: List[str] = []
			for sq in getattr(decomposition, 'subquestions', []) or []:
				q = getattr(sq, 'actual_question', None) or getattr(sq, 'question', None) or ''
				q = str(q).strip()
				if q:
					sq_queries.append(q)
			if not sq_queries and str(main_query).strip():
				sq_queries = [str(main_query).strip()]
			if not sq_queries:
				sq_queries = []

			# Dedupe + sanitize candidate indices
			cand: List[int] = []
			seen = set()
			try:
				n = int(len(r.titles))
			except Exception:
				n = int(len(candidate_indices))
			for i in candidate_indices:
				try:
					j = int(i)
				except Exception:
					continue
				if (j < 0) or (j >= n) or (j in seen):
					continue
				seen.add(j)
				cand.append(j)

			def _minmax_scale_over_candidates(raw_map: Dict[int, float], candidates: List[int]) -> Dict[int, float]:
				if not candidates:
					return {}
				vals = [float(raw_map.get(int(i), 0.0)) for i in candidates]
				vmin = min(vals)
				vmax = max(vals)
				den = (vmax - vmin)
				if den <= 0:
					return {int(i): 0.0 for i in candidates}
				return {int(i): (float(raw_map.get(int(i), 0.0)) - float(vmin)) / float(den) for i in candidates}

			if cand and sq_queries:
				# 4a) Cosine max over SQ embeddings (candidate-only; memmap-backed)
				embs = await asyncio.gather(*[r.embed_query(q) for q in sq_queries])
				qmat = np.stack(embs, axis=1)  # (dim, Q)
				cand_emb = r.get_normalized_embeddings(cand)  # (N, dim)
				sims = np.dot(cand_emb, qmat)  # (N, Q)
				cos_best = np.max(sims, axis=1)
				cos_raw_map: Dict[int, float] = {int(idx_i): float(sc) for idx_i, sc in zip(cand, cos_best)}

				bm25_raw_map: Dict[int, float] = {int(i): 0.0 for i in cand}
				bm25_k = min(50000, len(r.titles))
				for q in sq_queries:
					qtoks = r.preprocess_query(q)
					if not qtoks:
						continue
					results, scores = r.bm25.retrieve([qtoks], k=bm25_k)
					bm25_map = dict(zip(results[0], scores[0]))
					for idx_i in cand:
						sc = float(bm25_map.get(int(idx_i), 0.0))
						prev = float(bm25_raw_map.get(int(idx_i), 0.0))
						if sc > prev:
							bm25_raw_map[int(idx_i)] = sc

				# 4c) Min-max scale each feature over candidates
				bm25_scaled = _minmax_scale_over_candidates(bm25_raw_map, cand)
				cos_scaled = _minmax_scale_over_candidates(cos_raw_map, cand)

				# 4d) Weighted sum (requested)
				w_bm25 = 0.4
				w_cos = 0.6
				for idx_i in cand:
					idx_i = int(idx_i)
					sc = (w_bm25 * float(bm25_scaled.get(idx_i, 0.0))) + (w_cos * float(cos_scaled.get(idx_i, 0.0)))
					idx_score_map[idx_i] = float(sc)

		# 5) Select reranked paths (top-10 unique), excluding seed paths
		reranked_paths: List[Dict] = []
		seen_path = set(seed_path_keys)
		scored_indices_sorted = sorted(idx_score_map.items(), key=lambda kv: kv[1], reverse=True)
		for idx_i, sc in scored_indices_sorted:
			if len(reranked_paths) >= int(rerank_path_k):
				break
			cand = self._path_from_retriever_index(int(idx_i), score=float(sc), origin='rerank')
			k = self._path_dedupe_key(cand)
			if k in seen_path:
				continue
			seen_path.add(k)
			reranked_paths.append(cand)

		# 6) Select reranked passages (top-2) using cosine-only max path similarity
		passage_main_best: Dict[str, float] = {}
		for doc_id, plist in paths_by_doc.items():
			best_sc = float('-inf')
			for p in plist:
				idx = p.get('index')
				if idx is None:
					continue
				try:
					idx_i = int(idx)
				except Exception:
					continue
				sc = float(idx_score_map.get(idx_i, float('-inf')))
				if sc > best_sc:
					best_sc = sc
			if best_sc != float('-inf'):
				passage_main_best[doc_id] = best_sc

		passage_rerank_scores = sorted(passage_main_best.items(), key=lambda kv: kv[1], reverse=True)
		reranked_passage_doc_ids = [doc for doc, _ in passage_rerank_scores[: int(rerank_passage_k)]]

		# 7) Build final passages (5 unique)
		final_passage_doc_ids: List[str] = []
		seen_pass_doc = set()
		for doc_id in seed_passage_doc_ids:
			if doc_id in seen_pass_doc:
				continue
			seen_pass_doc.add(doc_id)
			final_passage_doc_ids.append(doc_id)
			if len(final_passage_doc_ids) >= int(top_passages_k):
				break
		for doc_id in reranked_passage_doc_ids:
			if len(final_passage_doc_ids) >= int(top_passages_k):
				break
			if doc_id in seen_pass_doc:
				continue
			seen_pass_doc.add(doc_id)
			final_passage_doc_ids.append(doc_id)

		# Fill if short: continue down rerank list then remaining_doc_ids
		if len(final_passage_doc_ids) < int(top_passages_k):
			for doc_id, _ in passage_rerank_scores:
				if len(final_passage_doc_ids) >= int(top_passages_k):
					break
				if doc_id in seen_pass_doc:
					continue
				seen_pass_doc.add(doc_id)
				final_passage_doc_ids.append(doc_id)
		if len(final_passage_doc_ids) < int(top_passages_k):
			for doc_id in remaining_doc_ids:
				if len(final_passage_doc_ids) >= int(top_passages_k):
					break
				if doc_id in seen_pass_doc:
					continue
				seen_pass_doc.add(doc_id)
				final_passage_doc_ids.append(doc_id)

		final_passages: List[Dict] = []
		for doc_id in final_passage_doc_ids[: int(top_passages_k)]:
			pobj = self._passage_from_doc_id(doc_id)
			if pobj is None:
				continue
			final_passages.append(pobj)
			if len(final_passages) >= int(top_passages_k):
				break

		# 8) Build final paths (unique)
		final_paths: List[Dict] = []
		for p in seed_paths:
			k = self._path_dedupe_key(p)
			if k in seed_path_keys:
				# seed_path_keys built from seed_paths; keep as-is
				pass
			final_paths.append(p)
			if len(final_paths) >= int(top_paths_k):
				break
		for p in reranked_paths:
			if len(final_paths) >= int(top_paths_k):
				break
			k = self._path_dedupe_key(p)
			if k in {self._path_dedupe_key(x) for x in final_paths}:
				continue
			final_paths.append(p)

		# Fill if short: continue down scored indices, excluding duplicates/seed
		if len(final_paths) < int(top_paths_k):
			seen_final = {self._path_dedupe_key(x) for x in final_paths}
			for idx_i, sc in scored_indices_sorted:
				if len(final_paths) >= int(top_paths_k):
					break
				cand = self._path_from_retriever_index(int(idx_i), score=float(sc), origin='rerank_fill')
				k = self._path_dedupe_key(cand)
				if k in seen_final:
					continue
				if k in seed_path_keys:
					continue
				seen_final.add(k)
				final_paths.append(cand)

		# Ultimate fallback: append best remaining SQ paths by score
		if len(final_paths) < int(top_paths_k):
			seen_final = {self._path_dedupe_key(x) for x in final_paths}
			for p in sorted(all_paths, key=self._safe_score, reverse=True):
				if len(final_paths) >= int(top_paths_k):
					break
				k = self._path_dedupe_key(p)
				if k in seen_final:
					continue
				p2 = dict(p)
				p2['origin'] = p2.get('origin') or 'fallback'
				seen_final.add(k)
				final_paths.append(p2)

		return final_paths[: int(top_paths_k)], final_passages

	async def generate_final_answer(
		self,
		main_query: str,
		decomposition: 'QueryDecomposition',
		all_passages: List[Dict],
	) -> str:
		from Prompt.answer_prompt import FINAL_ANSWER_SYNTHESIS_PROMPT

		if bool(getattr(self, 'use_previous_context', True)):
			chain_parts = []
			for sq in decomposition.subquestions:
				chain_parts.append(f"{sq.id}: {sq.question}")
				chain_parts.append(f"Answer: {sq.answer if sq.answer else '(Not answered)'}")
				chain_parts.append("")
			subquestion_chain = '\n'.join(chain_parts)
		else:
			subquestion_chain = "None"

		# Use precomputed selection if process_question already ran it.
		final_paths = getattr(decomposition, '_final_selected_paths', None)
		top_path_passages = getattr(decomposition, '_final_selected_passages', None)
		if not isinstance(final_paths, list) or not isinstance(top_path_passages, list):
			# IMPORTANT: Allow 0 as a valid ablation value.
			# Do NOT use `or <default>` on these because `0` is falsy.
			_answer_k_paths = getattr(self, 'answer_k_paths', None)
			_answer_k_passages = getattr(self, 'answer_k_passages', None)
			top_paths_k = int(getattr(self, 'top_k_paths', 30) if _answer_k_paths is None else _answer_k_paths)
			top_passages_k = int(getattr(self, 'top_k_passages', 5) if _answer_k_passages is None else _answer_k_passages)
			final_paths, top_path_passages = await self._select_top_paths_and_passages_from_decomposition(
				decomposition,
				top_paths_k=top_paths_k,
				top_passages_k=top_passages_k,
			)

		paths_text = self._format_paths_as_hints(final_paths)
		top_path_passages_text = self._format_passages_original(top_path_passages)
		combined_info = (
			f"---Top Retrieved Metadata Paths (TOP-{len(final_paths)} by score, UNIQUE; reused from SQs)---\n"
			"The paths below are strong hints for where the answer might be found. "
			"Use them to focus your reading of the passages, but do NOT treat them as guaranteed truth.\n\n"
			f"{paths_text}\n\n"
			f"---Top Passages from High-Score Paths (TOP-{len(top_path_passages)} by doc_id)---\n"
			f"{top_path_passages_text}"
		)

		prompt = FINAL_ANSWER_SYNTHESIS_PROMPT.replace("{{main_question}}", main_query)
		prompt = prompt.replace("{{subquestion_chain}}", subquestion_chain)
		prompt = prompt.replace("{{passages}}", combined_info)

		response = await self.client.chat.completions.create(
			model=self.chat_model,
			messages=[
				{
					"role": "system",
					"content": (
						"You are a historian-style question answering system. "
						"Answer in Korean with long, specific, historically grounded detail. "
						"Do not output a fixed fallback phrase like 'Insufficient information.'"
					),
				},
				{"role": "user", "content": prompt},
			],
			temperature=0.0,
			max_tokens=900,
		)

		answer_raw = (response.choices[0].message.content or '').strip()

		log_llm_call(
			call_type="Final Answer Synthesis (V12-PathsHint-Rerank)",
			input_text=prompt,
			output_text=answer_raw,
			context={
				"main_query": main_query,
				"num_passages": len(top_path_passages),
				"num_paths": len(final_paths),
				"num_top_path_passages": len(top_path_passages),
			},
		)

		if answer_raw.startswith("Answer:"):
			return answer_raw[7:].strip()
		return answer_raw

	async def process_question(self, question: str) -> Dict:
		"""Run non-QD pipeline: one-shot retrieval + final answer synthesis."""
		t0_total = time.perf_counter()
		retrieval_s = 0.0
		final_llm_s = 0.0

		try:
			if self.verbose:
				print(f"\n{'='*60}")
				print(f"Question: {question}")
				print(f"{'='*60}")
				print("\n[1] Retrieving passages/paths (no QD)...")

			t_retrieval = time.perf_counter()
			final_passages, final_paths = await self.retrieve_for_query(question)
			retrieval_s = time.perf_counter() - t_retrieval

			if self.verbose:
				print(
					f"   Retrieved: passages={len(final_passages or [])}, "
					f"paths={len(final_paths or [])}"
				)
				print("\n[2] Generating final answer...")

			from Prompt.answer_prompt import FINAL_ANSWER_SYNTHESIS_PROMPT

			paths_text = self._format_paths_as_hints(final_paths or [])
			top_path_passages_text = self._format_passages_original(final_passages or [])
			combined_info = (
				f"---Top Retrieved Metadata Paths (TOP-{len(final_paths or [])}, UNIQUE)---\n"
				"The paths below are hints for where the answer might be found. "
				"Use them to focus your reading of the passages, but do NOT treat them as guaranteed truth.\n\n"
				f"{paths_text}\n\n"
				f"---Top Passages from Retrieved Paths (TOP-{len(final_passages or [])} by score)---\n"
				f"{top_path_passages_text}"
			)

			prompt = FINAL_ANSWER_SYNTHESIS_PROMPT.replace("{{main_question}}", question)
			prompt = prompt.replace("{{subquestion_chain}}", "None")
			prompt = prompt.replace("{{passages}}", combined_info)

			t_final = time.perf_counter()
			response = await self.client.chat.completions.create(
				model=self.chat_model,
				messages=[
					{
						"role": "system",
						"content": (
							"You are a historian-style question answering system. "
							"Answer in Korean with long, specific, historically grounded detail. "
							"Do not output a fixed fallback phrase like 'Insufficient information.'"
						),
					},
					{"role": "user", "content": prompt},
				],
				temperature=0.0,
				max_tokens=900,
			)
			answer_raw = (response.choices[0].message.content or '').strip()
			final_llm_s = time.perf_counter() - t_final

			log_llm_call(
				call_type="Final Answer Synthesis (V12-NoQD)",
				input_text=prompt,
				output_text=answer_raw,
				context={
					"main_query": question,
					"num_passages": len(final_passages or []),
					"num_paths": len(final_paths or []),
					"pipeline_mode": "no_qd",
				},
			)

			if answer_raw.startswith("Answer:"):
				final_answer = answer_raw[7:].strip()
			else:
				final_answer = answer_raw

			elapsed = time.perf_counter() - t0_total

			try:
				origin_counts = {}
				doc_id_counts = {}
				for p in (final_paths or []):
					origin = p.get('origin') or 'unknown'
					origin = str(origin)
					origin_counts[origin] = int(origin_counts.get(origin, 0)) + 1
					doc_id = p.get('doc_id')
					if doc_id is not None:
						k = str(doc_id)
						doc_id_counts[k] = int(doc_id_counts.get(k, 0)) + 1

				top_doc_ids = sorted(doc_id_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
				final_selection_stats = {
					'paths_total': int(len(final_paths or [])),
					'passages_total': int(len(final_passages or [])),
					'path_origin_counts': origin_counts,
					'path_doc_id_unique': int(len(doc_id_counts)),
					'path_doc_id_top10': [{'doc_id': k, 'count': v} for k, v in top_doc_ids],
					'passage_doc_id_unique': int(
						len({str(p.get('doc_id')) for p in (final_passages or []) if p.get('doc_id') is not None})
					),
				}
			except Exception:
				final_selection_stats = None

			return {
				'success': True,
				'final_answer': final_answer,
				'predicted_answer': final_answer,
				'final_retrieved_passages': [
					{
						'doc_id': str(p.get('doc_id')) if p.get('doc_id') is not None else None,
						'title': p.get('title'),
						'content': p.get('original_passage'),
						'score': float(p.get('passage_score') or p.get('support_path_score') or p.get('score') or 0.0),
						'passage_score': p.get('passage_score'),
						'support_path_score': p.get('support_path_score'),
						'support_path_origin': p.get('support_path_origin'),
					}
					for p in (final_passages or [])
				],
				'final_retrieved_paths': [
					{
						'doc_id': str(p.get('doc_id')) if p.get('doc_id') is not None else None,
						'title': p.get('title'),
						'key_path': p.get('key_path'),
						'value': p.get('value'),
						'score': float(p.get('score') or 0.0),
						'bm25_score': p.get('bm25_score'),
						'dense_score': p.get('dense_score'),
						'origin': p.get('origin'),
					}
					for p in (final_paths or [])
				],
				'final_selection_stats': final_selection_stats,
				'decomposition': None,
				'num_passages': len(final_passages),
				'num_paths': len(final_paths),
				'time': elapsed,
				'timing': {
					'total_s': elapsed,
					'retrieval_s': retrieval_s,
					'final_llm_s': final_llm_s,
					'qd_s': 0.0,
					'subquestions_total_s': 0.0,
					'subquestions_retrieval_s': 0.0,
					'subquestions_llm_s': 0.0,
					'final_select_s': retrieval_s,
				},
			}
		except Exception as e:
			elapsed = time.perf_counter() - t0_total
			return {
				'success': False,
				'error': str(e),
				'time': elapsed,
				'timing': {
					'total_s': elapsed,
					'retrieval_s': retrieval_s,
					'final_llm_s': final_llm_s,
					'qd_s': 0.0,
					'subquestions_total_s': 0.0,
					'subquestions_retrieval_s': 0.0,
					'subquestions_llm_s': 0.0,
					'final_select_s': retrieval_s,
				},
			}


async def run_small_batch(
	dataset: str,
	limit: int | None,
	output_path: str,
	data_path: str,
	db_path: str,
	bm25_index_path: str,
	embeddings_path: str,
	top_k_passages: int,
	top_k_paths: int,
	path_fetch_k: int,
	verbose: bool,
	seed_k: int,
	expansion_k: int,
	expansion_dense_candidates: int,
	seed_passages_in_final: int,
	no_llm: bool = False,
) -> None:
	from dotenv import load_dotenv
	from llm_logger import finalize_log, init_logger
	from llm_provider import create_async_chat_client, detect_provider

	load_dotenv()
	init_logger()

	client = None
	if not no_llm:
		cfg = detect_provider()
		client = create_async_chat_client(cfg)

	retriever = HybridPathRetriever(
		bm25_weight=0.4,
		dense_weight=0.6,
		bm25_index_path=bm25_index_path,
		embeddings_path=embeddings_path,
	)

	pipeline = NewMultihopPipelineV12PathsHintExpansion(
		client=client,  # type: ignore[arg-type]
		retriever=retriever,
		hotpotqa_path=data_path,
		db_path=db_path,
		top_k_passages=top_k_passages,
		top_k_paths=top_k_paths,
		path_fetch_k=path_fetch_k,
		verbose=verbose,
		seed_k=seed_k,
		expansion_k=expansion_k,
		expansion_dense_candidates=expansion_dense_candidates,
		seed_passages_in_final=seed_passages_in_final,
	)

	with open(data_path, 'r', encoding='utf-8') as f:
		gold = json.load(f)

	results: List[Dict] = []
	n = len(gold) if limit is None else min(int(limit), len(gold))
	print(f'[run_small_batch:v12] dataset={dataset} examples={n} output={output_path}')
	print(f'  data_path={data_path}')
	print(f'  db_path={db_path}')
	print(f'  bm25_index_path={bm25_index_path}')
	print(f'  embeddings_path={embeddings_path}')

	for i, item in enumerate(gold[:n], 1):
		q = item.get('question', '')
		item_id = item.get('_id') or item.get('id')
		print(f'[{i}/{n}] id={item_id}')
		if verbose:
			print(f'Q: {q}')

		if no_llm:
			passages, paths = await pipeline.retrieve_for_query(q)
			out = {
				'success': True,
				'final_answer': item.get('answer', ''),
				'predicted_answer': item.get('answer', ''),
				'final_retrieved_passages': [
					{'doc_id': p.get('doc_id'), 'title': p.get('title')} for p in (passages or [])
				],
				'final_retrieved_paths': [{'doc_id': p.get('doc_id')} for p in (paths or [])],
				'decomposition': None,
				'num_passages': len(passages or []),
				'num_paths': len(paths or []),
				'time': 0.0,
			}
		else:
			out = await pipeline.process_question(q)

		status = 'OK' if out.get('success') else 'FAIL'
		t = out.get('time')
		if isinstance(t, (int, float)):
			print(f'  -> {status} ({t:.1f}s)')
		else:
			print(f'  -> {status}')
		if (not out.get('success')) and out.get('error'):
			print(f"  error: {out.get('error')}")

		merged: Dict = {
			'question': q,
			'answer': item.get('answer'),
			'answer_aliases': item.get('answer_aliases'),
			'_id': item_id,
			'id': item.get('id') or item_id,
		}
		merged.update(out)
		if merged.get('predicted_answer') is None:
			merged['predicted_answer'] = merged.get('final_answer', '')
		results.append(merged)

	pipeline.close()
	finalize_log()

	os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
	payload = {
		'meta': {
			'dataset': dataset,
			'limit': n,
			'artifact_paths': {
				'data_path': data_path,
				'db_path': db_path,
				'bm25_index_path': bm25_index_path,
				'embeddings_path': embeddings_path,
			},
			'top_k_passages': top_k_passages,
			'top_k_paths': top_k_paths,
			'path_fetch_k_input': path_fetch_k,
			'seed_k': seed_k,
			'expansion_k': expansion_k,
			'expansion_dense_candidates': expansion_dense_candidates,
			'seed_passages_in_final': seed_passages_in_final,
		},
		'results': results,
	}
	with open(output_path, 'w', encoding='utf-8') as f:
		json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
	print(f'[OK] Wrote: {output_path}')


def main():
	import argparse

	parser = argparse.ArgumentParser(
		description='Run v12 paths-as-hints + fact expansion pipeline on a small batch (v5 artifacts).'
	)
	parser.add_argument('--dataset', choices=['musique', 'hotpot', '2wiki', 'lveval', 'sillok'], required=True)
	parser.add_argument('--output', type=str, default='')
	parser.add_argument('--data_path', type=str, default='')
	parser.add_argument('--db_path', type=str, default='')
	parser.add_argument('--bm25_index_path', type=str, default='')
	parser.add_argument('--embeddings_path', type=str, default='')
	parser.add_argument('--top_k_passages', type=int, default=5)
	parser.add_argument('--top_k_paths', type=int, default=30)
	parser.add_argument('--path_fetch_k', type=int, default=50)
	parser.add_argument('--seed_k', type=int, default=20)
	parser.add_argument('--expansion_k', type=int, default=10)
	parser.add_argument('--expansion_dense_candidates', type=int, default=500)
	parser.add_argument('--seed_passages_in_final', type=int, default=3)
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument(
		'--no_llm',
		action='store_true',
		help='Retrieval-only smoke run: skip LLM calls and use gold answers as predictions',
	)
	args = parser.parse_args()

	defaults = _default_artifact_paths(args.dataset)
	data_path = args.data_path or defaults['data_path']
	db_path = args.db_path or defaults['db_path']
	bm25_index_path = args.bm25_index_path or defaults['bm25_index_path']
	embeddings_path = args.embeddings_path or defaults['embeddings_path']
	output_path = args.output or f"Results/smoke_v12_{args.dataset}_v5_all.json"

	asyncio.run(
		run_small_batch(
			dataset=str(args.dataset),
			limit=None,
			output_path=str(output_path),
			data_path=str(data_path),
			db_path=str(db_path),
			bm25_index_path=str(bm25_index_path),
			embeddings_path=str(embeddings_path),
			top_k_passages=int(args.top_k_passages),
			top_k_paths=int(args.top_k_paths),
			path_fetch_k=int(args.path_fetch_k),
			verbose=bool(args.verbose),
			seed_k=int(args.seed_k),
			expansion_k=int(args.expansion_k),
			expansion_dense_candidates=int(args.expansion_dense_candidates),
			seed_passages_in_final=int(args.seed_passages_in_final),
			no_llm=bool(args.no_llm),
		)
	)


if __name__ == '__main__':
	main()
