import csv
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routing_strategy.embedding_data import load_corpus, audit_token_lengths


def source_files(tmp_path, rows=None, rules=None, header=None):
    csv_path = tmp_path / 'tasks.csv'
    rules_path = tmp_path / 'rules.py'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header or ['task_id', 'instruction'])
        writer.writerows(rows if rows is not None else [('a', 'Repair code\nwithout deleting tests')])
    rules_path.write_text(
        rules if rules is not None else "RULES = [('a', ('repair',), 'pi', 'all-fail')]", encoding='utf-8'
    )
    return csv_path, rules_path


def test_preserves_text_labels_and_stable_hashes(tmp_path):
    files = source_files(tmp_path)
    corpus = load_corpus(*files)
    record = corpus.records[0]
    assert record.instruction == 'Repair code\nwithout deleting tests'
    assert (record.harness, record.label_case) == ('pi', 'all-fail')
    assert len(record.instruction_sha256) == 64
    assert load_corpus(*files) == corpus


def test_never_executes_rules_module(tmp_path):
    files = source_files(tmp_path, rules="raise RuntimeError('must not run')\nRULES=[('a', (), 'pi', 'only-pass')]")
    assert load_corpus(*files).records[0].harness == 'pi'


@pytest.mark.parametrize('rules', [
    "RULES = make_rules()", 'RULES = []', 'OTHER = []',
    "RULES = [('a', (), 'other', 'all-fail')]",
    "RULES = [('a', (), 'pi', 'unknown')]",
    "RULES = [('a', (), 'pi', 'all-fail'), ('a', (), 'pi', 'all-fail')]",
    "RULES = [('b', (), 'pi', 'all-fail')]",
    "RULES = [('a', (), 'pi', 'all-fail'), ('b', (), 'pi', 'all-fail')]",
    "RULES = [('a', (), 'pi', 'all-fail')]\nRULES = []",
])
def test_bad_labels_fail_explicitly(tmp_path, rules):
    with pytest.raises(ValueError):
        load_corpus(*source_files(tmp_path, rules=rules))


@pytest.mark.parametrize('rows', [[], [('a', '')], [('a', '   ')], [(' a', 'x')],
    [('a', 'one'), ('a', 'two')], [('a', 'one', 'extra')], [('a',)],
])
def test_bad_csv_fails_explicitly(tmp_path, rows):
    with pytest.raises(ValueError):
        load_corpus(*source_files(tmp_path, rows=rows))


def test_duplicate_prompts_not_silently_kept(tmp_path):
    files = source_files(tmp_path, rows=[('a', 'same'), ('b', 'same')],
        rules="RULES=[('a', (), 'pi', 'all-fail'), ('b', (), 'pi', 'only-pass')]")
    with pytest.raises(ValueError, match='Duplicate instruction'):
        load_corpus(*files)


@pytest.mark.parametrize('headers', [['task_id', 'task_id'], ['task_id', 'description']])
def test_bad_headers(tmp_path, headers):
    with pytest.raises(ValueError):
        load_corpus(*source_files(tmp_path, header=headers))


def test_length_audit_explicit_boundary_and_no_truncation(tmp_path):
    corpus = load_corpus(*source_files(tmp_path))
    calls = []
    def tokenizer(text, **kwargs):
        calls.append((text, kwargs))
        return {'input_ids': [1, 2, 3, 4]}
    equal = audit_token_lengths(corpus, tokenizer, 4)
    over = audit_token_lengths(corpus, tokenizer, 3)
    assert equal['over_limit_count'] == 0
    assert over['over_limit_count'] == 1
    assert over['embeddings_computed'] is False
    assert calls[0] == (corpus.records[0].instruction, {'add_special_tokens': True, 'truncation': False})


@pytest.mark.parametrize('limit', [0, -1, True, 3.5])
def test_invalid_token_limit(tmp_path, limit):
    corpus = load_corpus(*source_files(tmp_path))
    with pytest.raises(ValueError):
        audit_token_lengths(corpus, lambda *a, **kw: {}, limit)


def test_real_sources_match_and_have_no_side_effects():
    root = Path(__file__).resolve().parents[1]
    corpus = load_corpus(root / 'data/tb2.1_tasks.csv', root / 'routing_strategy/rules_based.py')
    assert len(corpus.records) == 89
    assert sum(r.label_case == 'all-fail' for r in corpus.records) == 53
