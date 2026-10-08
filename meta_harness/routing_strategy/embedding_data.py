"""Read frozen prompt/label sources without executing their Python or task text.

This is input validation for the embedding prototype, not a routing strategy.
The legacy labels are inherited assignments, not new measured outcomes.
"""

import ast
import csv
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path


HARNESS_NAMES = frozenset({'mini-swe-agent', 'terminus-2', 'pi'})
LABEL_CASES = frozenset({'only-pass', 'multi-pass', 'all-fail'})


@dataclass(frozen=True)
class PromptRecord:
    task_id: str
    instruction: str
    harness: str
    label_case: str

    @property
    def instruction_sha256(self) -> str:
        return hashlib.sha256(self.instruction.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class PromptCorpus:
    records: tuple[PromptRecord, ...]
    csv_sha256: str
    rules_sha256: str


def _literal_labels(source: str) -> dict[str, tuple[str, str]]:
    """Extract the sole literal RULES assignment; never import/exec source."""
    tree = ast.parse(source)
    values = [
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'RULES' for t in node.targets)
    ]
    if len(values) != 1:
        raise ValueError('Expected exactly one literal RULES assignment')
    try:
        rules = ast.literal_eval(values[0])
    except (ValueError, TypeError) as exc:
        raise ValueError('RULES must be literal data, not executable expressions') from exc
    if not isinstance(rules, (list, tuple)) or not rules:
        raise ValueError('RULES must be a nonempty sequence')
    labels = {}
    for rule in rules:
        if not isinstance(rule, (list, tuple)) or len(rule) != 4:
            raise ValueError('Each rule must contain task_id, keywords, harness and case')
        task_id, _keywords, harness, label_case = rule
        if not isinstance(task_id, str) or not task_id.strip() or task_id != task_id.strip():
            raise ValueError('Invalid rule task_id')
        if task_id in labels:
            raise ValueError(f'Duplicate rule task_id: {task_id}')
        if not isinstance(harness, str) or harness not in HARNESS_NAMES:
            raise ValueError(f'Unknown harness for {task_id}')
        if not isinstance(label_case, str) or label_case not in LABEL_CASES:
            raise ValueError(f'Unknown label case for {task_id}')
        labels[task_id] = (harness, label_case)
    return labels


def load_corpus(csv_path: str | Path, rules_path: str | Path) -> PromptCorpus:
    """Join on task_id, retaining exact instruction text and every label case.

    Reject missing/extra IDs, malformed rows, empty prompts and duplicate prompts
    rather than silently dropping records, relabelling them or guessing a join.
    Other CSV columns are deliberately not used as embedding inputs.
    """
    csv_bytes = Path(csv_path).read_bytes()
    rules_bytes = Path(rules_path).read_bytes()
    labels = _literal_labels(rules_bytes.decode('utf-8-sig'))
    reader = csv.DictReader(io.StringIO(csv_bytes.decode('utf-8-sig'), newline=''), strict=True)
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)) or not {'task_id', 'instruction'} <= set(fields):
        raise ValueError('CSV requires unique headers including task_id and instruction')
    records = []
    seen_ids = set()
    seen_instructions = set()
    for row_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f'CSV record {row_number} has the wrong field count')
        task_id, instruction = row['task_id'], row['instruction']
        if not task_id.strip() or task_id != task_id.strip() or task_id in seen_ids:
            raise ValueError(f'Empty, padded or duplicate task_id at CSV record {row_number}')
        if not instruction.strip():
            raise ValueError(f'Empty instruction for {task_id}')
        if instruction in seen_instructions:
            raise ValueError(f'Duplicate instruction for {task_id}; resolve task-level exclusion first')
        if task_id not in labels:
            raise ValueError(f'Missing rule label for {task_id}')
        harness, label_case = labels[task_id]
        records.append(PromptRecord(task_id, instruction, harness, label_case))
        seen_ids.add(task_id)
        seen_instructions.add(instruction)
    extra = labels.keys() - seen_ids
    if extra:
        raise ValueError(f'Rules without CSV prompts: {sorted(extra)}')
    if not records:
        raise ValueError('Prompt corpus is empty')
    return PromptCorpus(
        records=tuple(records),
        csv_sha256=hashlib.sha256(csv_bytes).hexdigest(),
        rules_sha256=hashlib.sha256(rules_bytes).hexdigest(),
    )


def audit_token_lengths(corpus: PromptCorpus, tokenizer, max_tokens: int) -> dict:
    """Count exact, untruncated tokens including special tokens; do not embed.

    No query/document prefix is applied here. Any later prefix policy must be
    checked separately before model inference.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError('max_tokens must be a positive integer')
    rows = []
    for record in corpus.records:
        token_ids = tokenizer(
            record.instruction, add_special_tokens=True, truncation=False,
        )['input_ids']
        rows.append({
            'task_id': record.task_id, 'instruction_sha256': record.instruction_sha256,
            'harness': record.harness, 'label_case': record.label_case,
            'tokens_with_special_tokens': len(token_ids), 'exceeds_limit': len(token_ids) > max_tokens,
        })
    return {
        'csv_sha256': corpus.csv_sha256, 'rules_sha256': corpus.rules_sha256,
        'task_count': len(rows), 'max_tokens': max_tokens, 'prefix': '',
        'truncation': False, 'embeddings_computed': False,
        'over_limit_count': sum(r['exceeds_limit'] for r in rows),
        'minimum_tokens': min(r['tokens_with_special_tokens'] for r in rows),
        'maximum_tokens': max(r['tokens_with_special_tokens'] for r in rows),
        'records': rows,
    }
