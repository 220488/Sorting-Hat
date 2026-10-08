"""Local prompt-similarity router. No hosted inference or generative RAG.

Version 1 uses lossless non-overlapping tokenizer-bounded chunks. For each
query chunk, find its best cosine match within a candidate task; average those
scores weighted by query content-token count. Rank tasks, not individual chunks.
This is an explicit prototype policy, not a learned or validated optimum.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np

from .embedding_data import HARNESS_NAMES, LABEL_CASES, PromptCorpus

SCHEMA = 1
POLICY = 'token-boundary-no-overlap__query-token-weighted-mean-of-task-max-cosine-v1'
MODEL_ID = 'BAAI/bge-small-en-v1.5'
MODEL_REVISION = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
MODEL_FILES = (
    'config.json', 'config_sentence_transformers.json', 'modules.json',
    'sentence_bert_config.json', '1_Pooling/config.json', 'tokenizer.json',
    'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'model.safetensors',
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class Chunk:
    text: str
    content_tokens: int


def split_instruction(text: str, tokenizer, max_tokens: int) -> tuple[Chunk, ...]:
    """Preserve every character; recheck each slice with special tokens added."""
    if not isinstance(text, str):
        raise TypeError('instruction must be a string')
    if not text.strip():
        return ()
    special = tokenizer.num_special_tokens_to_add(pair=False)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= special:
        raise ValueError('Token budget must leave room for content and special tokens')
    chunks = []
    remaining = text
    while remaining:
        raw = tokenizer(remaining, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True)
        ids, offsets = raw['input_ids'], raw['offset_mapping']
        if len(ids) + special <= max_tokens:
            piece = remaining
        else:
            # A substring can re-tokenize differently at a WordPiece boundary.
            # Shrink the boundary until the actual substring fits; never truncate.
            ends = sorted({end for _start, end in offsets[:max_tokens-special] if end > 0}, reverse=True)
            piece = None
            for end in ends:
                candidate = remaining[:end]
                if len(tokenizer(candidate, add_special_tokens=True, truncation=False)['input_ids']) <= max_tokens:
                    piece = candidate
                    break
            if piece is None:
                raise ValueError('Tokenizer cannot produce a nonempty safe chunk')
        count = len(tokenizer(piece, add_special_tokens=False, truncation=False)['input_ids'])
        actual = len(tokenizer(piece, add_special_tokens=True, truncation=False)['input_ids'])
        if actual > max_tokens:
            raise ValueError('Chunk exceeds encoder limit')
        if count == 0:
            # Preserve a trailing whitespace-only tail without embedding it alone.
            if not chunks or piece.strip():
                raise ValueError('Tokenizer produced no content tokens')
            previous = chunks.pop()
            joined = previous.text + piece
            if len(tokenizer(joined, add_special_tokens=True, truncation=False)['input_ids']) > max_tokens:
                raise ValueError('Trailing text cannot be retained within token budget')
            chunks.append(Chunk(joined, previous.content_tokens))
        else:
            chunks.append(Chunk(piece, count))
        remaining = remaining[len(piece):]
    if ''.join(c.text for c in chunks) != text:
        raise AssertionError('Chunking lost input text')
    return tuple(chunks)


def unit_vectors(values, dimension: int, expected_rows: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != dimension or array.shape[0] == 0:
        raise ValueError('Invalid embedding matrix shape')
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise ValueError('Encoder returned the wrong number of vectors')
    norms = np.linalg.norm(array, axis=1)
    if not np.isfinite(array).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('Embeddings must be finite and nonzero')
    return array / norms[:, None]


class LocalBGE:
    """LlamaIndex adapter for the inspected local safetensors BGE model only."""
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir).absolute()
        if not model_dir.is_dir():
            raise ValueError('A downloaded local model directory is required')
        hashes = {}
        for name in MODEL_FILES:
            with (model_dir / name).open('rb') as stream:
                hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        config = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
        modules = json.loads((model_dir / 'modules.json').read_text(encoding='utf-8'))
        if config.get('auto_map') or config.get('model_type') != 'bert' or config.get('hidden_size') != 384:
            raise ValueError('Unexpected model config; inspect before use')
        expected_modules = [
            {'idx': 0, 'name': '0', 'path': '', 'type': 'sentence_transformers.models.Transformer'},
            {'idx': 1, 'name': '1', 'path': '1_Pooling', 'type': 'sentence_transformers.models.Pooling'},
            {'idx': 2, 'name': '2', 'path': '2_Normalize', 'type': 'sentence_transformers.models.Normalize'},
        ]
        if modules != expected_modules:
            raise ValueError('Unexpected sentence-transformers modules; inspect before use')
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        self.encoder = HuggingFaceEmbedding(
            model_name=str(model_dir), device='cpu', trust_remote_code=False,
            local_files_only=True, model_kwargs={'use_safetensors': True},
            query_instruction='', text_instruction='', normalize=True,
            cache_folder=str(model_dir / '.cache'), embed_batch_size=8,
        )
        self.tokenizer = self.encoder._model.tokenizer
        if not self.tokenizer.is_fast:
            raise ValueError('Fast tokenizer offsets required for lossless chunk boundaries')
        self.max_tokens = min(self.encoder.max_length, self.tokenizer.model_max_length,
                              config['max_position_embeddings'])
        self.dimension = 384
        self.signature = {
            'model_id': MODEL_ID, 'reference_revision': MODEL_REVISION,
            'files_sha256': hashes, 'dimension': self.dimension, 'max_tokens': self.max_tokens,
            'device': 'cpu', 'normalize': True, 'text_prefix': '', 'query_prefix': '',
            'libraries': {name: importlib.metadata.version(name) for name in (
                'llama-index-core', 'llama-index-embeddings-huggingface',
                'sentence-transformers', 'transformers', 'torch', 'tokenizers',
            )},
        }

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts or any(not t.strip() for t in texts):
            raise ValueError('Cannot embed empty input')
        for text in texts:
            if len(self.tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']) > self.max_tokens:
                raise ValueError('Refusing silent model truncation')
        # Use the same encoding path for stored prompts and incoming prompts.
        # IDs, harnesses and outcomes are never passed to the embedder.
        return unit_vectors(self.encoder.get_text_embedding_batch(texts), self.dimension, len(texts))


def build_index(corpus: PromptCorpus, backend) -> dict:
    entries = []
    for record in sorted(corpus.records, key=lambda r: r.task_id):
        chunks = split_instruction(record.instruction, backend.tokenizer, backend.max_tokens)
        vectors = unit_vectors(backend.embed([c.text for c in chunks]), backend.dimension, len(chunks))
        entries.append({
            'task_id': record.task_id, 'instruction_sha256': record.instruction_sha256,
            'harness': record.harness, 'label_case': record.label_case,
            'chunk_tokens': [c.content_tokens for c in chunks],
            'chunk_sha256': [hashlib.sha256(c.text.encode('utf-8')).hexdigest() for c in chunks],
            'vectors': vectors.tolist(),
        })
    payload = {
        'schema': SCHEMA, 'policy': POLICY, 'backend': backend.signature,
        'sources': {'csv_sha256': corpus.csv_sha256, 'rules_sha256': corpus.rules_sha256},
        'entries': entries,
    }
    return {'index_id': digest(payload), 'payload': payload}


def save_index(index: dict, path: str | Path) -> None:
    """Write a new artifact only; never silently replace an existing index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(index, stream, allow_nan=False, separators=(',', ':'))
        stream.write('\n')


@dataclass(frozen=True)
class Decision:
    harness: str
    matched_rule: str
    diagnostics: dict

    def as_tuple(self) -> tuple[str, str]:
        return self.harness, self.matched_rule


class EmbeddingIndex:
    def __init__(self, index: dict, backend):
        try:
            payload = index['payload']
            if index['index_id'] != digest(payload):
                raise ValueError('Index checksum mismatch')
            if payload['schema'] != SCHEMA or payload['policy'] != POLICY:
                raise ValueError('Unsupported index policy/schema')
            if payload['backend'] != backend.signature:
                raise ValueError('Index/model/runtime mismatch: rebuild with the current environment')
            entries = payload['entries']
            if not isinstance(entries, list) or not entries:
                raise ValueError('Index must contain at least one task')
            seen = set()
            parsed = []
            for entry in entries:
                task_id = entry['task_id']
                if not isinstance(task_id, str) or not task_id.strip() or task_id in seen:
                    raise ValueError('Invalid or duplicate index task_id')
                if entry['harness'] not in HARNESS_NAMES or entry['label_case'] not in LABEL_CASES:
                    raise ValueError('Invalid index label')
                sha = entry['instruction_sha256']
                if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                    raise ValueError('Invalid prompt checksum')
                counts = entry['chunk_tokens']
                if not counts or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in counts):
                    raise ValueError('Invalid chunk token counts')
                if len(entry['chunk_sha256']) != len(counts):
                    raise ValueError('Chunk metadata mismatch')
                vectors = unit_vectors(entry['vectors'], backend.dimension, len(counts))
                parsed.append((dict(entry), vectors))
                seen.add(task_id)
        except (KeyError, TypeError) as exc:
            raise ValueError('Malformed embedding index') from exc
        self.backend = backend
        self.index_id = index['index_id']
        self.sources = dict(payload['sources'])
        self.entries = tuple(sorted(parsed, key=lambda pair: pair[0]['task_id']))

    @classmethod
    def load(cls, path, backend):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')), backend)

    def check_sources(self, corpus: PromptCorpus):
        if self.sources != {'csv_sha256': corpus.csv_sha256, 'rules_sha256': corpus.rules_sha256}:
            raise ValueError('Index source files changed: rebuild before evaluation')

    def rank(self, instruction: str, *, exclude_task_ids=(), exclude_prompt_hashes=()) -> tuple[list[dict], int]:
        chunks = split_instruction(instruction, self.backend.tokenizer, self.backend.max_tokens)
        if not chunks:
            return [], 0
        excluded_ids, excluded_hashes = set(exclude_task_ids), set(exclude_prompt_hashes)
        # Excluding an ID also excludes duplicates of that task's entire prompt.
        excluded_hashes.update(e['instruction_sha256'] for e, _ in self.entries if e['task_id'] in excluded_ids)
        candidates = [(e, v) for e, v in self.entries
                      if e['task_id'] not in excluded_ids and e['instruction_sha256'] not in excluded_hashes]
        if not candidates:
            return [], len(chunks)
        query = unit_vectors(self.backend.embed([c.text for c in chunks]), self.backend.dimension, len(chunks))
        weights = np.asarray([c.content_tokens for c in chunks], dtype=np.float64)
        ranked = []
        for entry, vectors in candidates:
            best_per_query_chunk = np.clip(query @ vectors.T, -1, 1).max(axis=1)
            score = float(np.average(best_per_query_chunk, weights=weights))
            ranked.append({
                'task_id': entry['task_id'], 'harness': entry['harness'],
                'label_case': entry['label_case'], 'score': score,
            })
        # Exact ties choose lexicographically by ID, independent of source row order.
        ranked.sort(key=lambda r: (-r['score'], r['task_id']))
        return ranked, len(chunks)


class EmbeddingRouter:
    def __init__(self, index: EmbeddingIndex, *, min_similarity: float):
        if (isinstance(min_similarity, bool) or not isinstance(min_similarity, (float, int))
                or not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1):
            raise ValueError('min_similarity must be an explicit finite cosine cutoff in [-1, 1]')
        self.index = index
        self.min_similarity = float(min_similarity)

    def decide(self, instruction: str) -> Decision:
        started = time.perf_counter()
        ranked, chunks = self.index.rank(instruction)
        top = ranked[0] if ranked else None
        accepted = top is not None and top['score'] >= self.min_similarity
        diagnostics = {
            'reason': 'matched' if accepted else ('low_similarity' if top else 'empty_input'),
            'score': top['score'] if top else None, 'min_similarity': self.min_similarity,
            'nearest_task': top['task_id'] if top else None,
            'inherited_label_case': top['label_case'] if top else None,
            'query_chunks': chunks, 'index_id': self.index.index_id, 'policy': POLICY,
            'model_id': self.index.backend.signature.get('model_id'),
            'embedding_api_cost_usd': 0.0, 'cost_scope': 'local_embedding_only_excludes_cpu_and_solver',
            'latency_ms': (time.perf_counter()-started)*1000,
        }
        return Decision(top['harness'] if accepted else 'NA', top['task_id'] if accepted else 'NA', diagnostics)

    def route(self, instruction: str) -> tuple[str, str]:
        return self.decide(instruction).as_tuple()


def create_router(*, model_dir, index_path, min_similarity) -> EmbeddingRouter:
    """Per-agent configured instance; no global mutable routing/diagnostic state."""
    backend = LocalBGE(model_dir)
    return EmbeddingRouter(EmbeddingIndex.load(index_path, backend), min_similarity=min_similarity)
