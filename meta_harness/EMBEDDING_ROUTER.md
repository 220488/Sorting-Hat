# Local embedding router prototype

This adds an optional prompt-similarity strategy. The existing keyword strategy
remains the default. No benchmark solve-rate improvement has been established.
The first version has been checked locally on Windows CPU, not on AWS/Linux.

## How it works

The input is the `instruction` column of `data/tb2.1_tasks.csv`. Harness labels
and their `only-pass`, `multi-pass` or `all-fail` provenance come from the literal
`RULES` table in `routing_strategy/rules_based.py`. The loader does not execute
that source. All 89 legacy assignments are retained, including 53 all-fail
assignments. Those 53 are not labels of successful harnesses.

LlamaIndex's `HuggingFaceEmbedding` loads a local BGE model on CPU. Only instruction
text is embedded. Task IDs, categories, harness names and outcomes are excluded
from the embedding input. There is no LLM generation, hosted vector database,
remote inference or API key requirement. Exact cosine search uses NumPy because
this small index needs task-level aggregation rather than ordinary chunk top-k.

Stored and incoming prompts use the same encoding path, with no added prefix.
Text is sliced at tokenizer offsets into non-overlapping chunks that fit the
model limit including special tokens. Every character is retained and every
slice is re-tokenized before inference. Overlong text is never silently truncated.
These are token boundaries, not necessarily sentence boundaries.

For each query chunk, take its highest cosine similarity to any chunk of a
candidate task. Average these scores, weighted by the query chunks' content-token
counts. Choose the task with the highest aggregate score and inherit its harness.
Exact score ties use lexicographic task ID order. This is a simple, explicit
prototype policy, not an empirically established best choice. Tasks with more
stored chunks have more opportunities to match; this remains a limitation to test.

The model author documents both no-prefix use and the need to assess similarity
cutoffs on relevant data: [BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5).
The local adapter follows the [LlamaIndex Hugging Face interface](https://developers.llamaindex.ai/python/framework-api-reference/embeddings/huggingface/).

## Isolated setup

Use Python 3.12 and a dedicated virtual environment. Do not replace the team's
base `uv.lock`: these optional prototype requirements are separate. The versions
below were tested together locally. Linux installation has not been verified.
Commands assume the working directory is `meta_harness`; `PYTHON` below means
the chosen environment's Python executable, not a literal executable name.

```text
uv venv .venv-embedding --python 3.12
uv pip install --python PYTHON --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu
uv pip install --python PYTHON harbor==0.22.0 -r requirements-embedding.txt
uv pip check --python PYTHON
```

Download the public model once using that environment's `hf` command:

```text
hf download BAAI/bge-small-en-v1.5 config.json config_sentence_transformers.json modules.json sentence_bert_config.json 1_Pooling/config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt model.safetensors --revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a --local-dir MODEL_DIR
```

`MODEL_DIR` is a local directory. Set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and
`HF_HUB_DISABLE_TELEMETRY=1` before downloading if using a machine with a saved
Hugging Face login. Only safetensors weights are loaded, with
`trust_remote_code=False` and `local_files_only=True`. The CLI also blocks all
network connections. Do not disable networking globally for a Harbor job: its
solver needs its normal connection, independently of the local router.

## Build and inspect

Replace `PYTHON`, `MODEL_DIR` and output paths with local values. Choose a new
output filename for each build/analysis; commands will not overwrite an artifact.
Keep generated indexes, model weights and result logs out of Git.

```text
PYTHON scripts/audit_embedding_inputs.py --csv data/tb2.1_tasks.csv --rules routing_strategy/rules_based.py --model-dir MODEL_DIR --output TOKEN_AUDIT.json
PYTHON scripts/embedding_router.py --model-dir MODEL_DIR build --csv data/tb2.1_tasks.csv --rules routing_strategy/rules_based.py --output INDEX.json
PYTHON scripts/embedding_router.py --model-dir MODEL_DIR analyze --csv data/tb2.1_tasks.csv --rules routing_strategy/rules_based.py --index INDEX.json --output ANALYSIS.json
```

Indexes contain vectors, task/label provenance, prompt/chunk hashes and source
hashes, not plaintext prompts. A payload checksum catches corruption. Exact
model-file hashes, package versions and preprocessing policy must match at load
time; rebuild on the target environment when they differ. This strict check is
intentional. It does not prove an artifact came from a trusted publisher.

`analyze` separates self-match wiring checks from task-excluded retrieval.
Task exclusion removes every chunk of the task and any identical full prompts.
It reports similarity distributions and illustrative cutoff/coverage pairs.
It does not choose a cutoff, optimize one against final test outcomes, execute a
solver, or claim that inherited-label agreement measures successful task solving.

## Enable in either meta-harness wrapper

In a **copy** of the job configuration, set the agent's kwargs as follows:

```json
{
  "routing_strategy": "embedding",
  "routing_kwargs": {
    "model_dir": "ABSOLUTE_LOCAL_MODEL_DIRECTORY",
    "index_path": "ABSOLUTE_LOCAL_INDEX_PATH",
    "min_similarity": "REPLACE_WITH_AGREED_NUMERIC_CUTOFF"
  }
}
```

This example is deliberately not runnable until an explicit numeric cutoff is
chosen. There is no default cutoff and none has been calibrated yet. A cutoff
must be a finite number in [-1, 1]; a score exactly equal to it is accepted.
Below it, the router returns `NA`, preserving the wrapper's existing default
fallback (`mini-swe-agent`). Empty input also returns `NA`. Setting a cutoff to
accept every neighbour should be labelled explicitly as an always-nearest
exploration, not presented as calibrated confidence.

Both `SortingHat` and `SortingHatNoRetry` support the option. Their execution
retry behaviour is unchanged. Configuration/index/model failures stop agent
construction; runtime routing failures use the existing NA fallback and log
`routing_error`, distinct from `low_similarity`. Routing-only kwargs are not
forwarded to the selected harness. `EmbeddingRouter.route(text)` retains the
two-string tuple interface; wrappers use `decide(text)` for per-call diagnostics.

Logs record aggregate score, cutoff, nearest task, inherited label case, model,
index/policy version, chunk count, latency and fallback. The local embedding API
spend is zero; that excludes CPU time and solver spending. Model-load time and
memory are additional costs, and each agent currently owns its own model instance.
Concurrent AWS trials may need resource checks before a full run.

To switch back, use `routing_strategy=rules_based` and remove `routing_kwargs`.
The existing keyword source and frozen assignments have not been edited.

## Tests

```text
PYTHON -m pytest tests -q -p no:cacheprovider
```

Tests use fake harnesses and embeddings, except for reading the frozen CSV/rule
data. External socket connections are blocked; Windows asyncio requires local
loopback self-pipes. No solver, Docker or paid API is invoked. Coverage includes
token boundaries, full-text retention, aggregation, ties, exclusions, malformed
indexes, threshold boundaries, per-call diagnostics, both wrapper fallbacks and
the original retry map. Real-model index build/reload/analysis are separate
offline checks, not a completed Harbor benchmark.

The local suite currently passes 87 tests. A second offline index build is used
to check repeatability separately from those unit/integration tests.

## First local observations

The original 89 prompts contain 8 inputs over 512 tokens (maximum 1,545 including
special tokens, no prefix). This policy yields 99 chunks. Both the new embedding
self-match and existing keyword self-match checks identify all 89 source tasks.
This only establishes wiring against known prompts.

After removing each query task from the index, 27 of 89 nearest-task harnesses
agree with the inherited source assignments. Scores range from about 0.612 to
0.938 (median 0.723). These are neither pass rates nor proof of improvement.
The tasks were already exposed during development, and many labels are all-fail
fallback choices. Semantic similarity need not imply the same best harness.

No rejection cutoff has been selected. Fresh evaluation design, Linux/AWS
resource compatibility and a full meta-harness solver run remain open. Do not
compare these retrieval figures with the team's 35/36-task oracle figures;
that earlier numerical discrepancy also remains unresolved.
