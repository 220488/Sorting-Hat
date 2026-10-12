import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routing_strategy.embedding import (
    EmbeddingIndex, EmbeddingRouter, build_index, digest, save_index,
    split_instruction, unit_vectors,
)
from routing_strategy.embedding_data import PromptCorpus, PromptRecord


class CharTokenizer:
    """Deterministic fake: one token per character, two special tokens."""
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, *, add_special_tokens=True, truncation=False, return_offsets_mapping=False):
        assert truncation is False
        result = {'input_ids': list(range(len(text) + (2 if add_special_tokens else 0)))}
        if return_offsets_mapping:
            result['offset_mapping'] = [(i, i+1) for i in range(len(text))]
        return result


class FakeBackend:
    tokenizer = CharTokenizer()
    max_tokens = 4
    dimension = 2
    signature = {'model_id': 'fake', 'dimension': 2}

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.extend(texts)
        return [[1., 0.] if 'a' in text else [0., 1.] for text in texts]


def corpus(*spec):
    return PromptCorpus(tuple(PromptRecord(*r) for r in spec), 'csv', 'rules')


def fixture_index():
    backend = FakeBackend()
    source = corpus(('z', 'aa', 'pi', 'all-fail'), ('b', 'bb', 'terminus-2', 'only-pass'))
    return backend, build_index(source, backend)


@pytest.mark.parametrize('text', ['a', 'aa', 'aabb', 'aabbc', ' a\nbb  ', '中文🙂abc'])
def test_chunk_full_coverage_and_limit(text):
    chunks = split_instruction(text, CharTokenizer(), 4)
    assert ''.join(c.text for c in chunks) == text
    assert all(len(c.text) <= 2 and c.content_tokens == len(c.text) for c in chunks)


@pytest.mark.parametrize('text', ['', '  ', '\n\t'])
def test_blank_chunks(text):
    assert split_instruction(text, CharTokenizer(), 4) == ()


def test_invalid_input_and_budget():
    with pytest.raises(TypeError):
        split_instruction(None, CharTokenizer(), 4)
    with pytest.raises(ValueError):
        split_instruction('a', CharTokenizer(), 2)


@pytest.mark.parametrize('vectors', [[[0., 0.]], [[float('nan'), 1.]], [[float('inf'), 1.]], [[1.]], [], [1., 0.]])
def test_bad_vectors(vectors):
    with pytest.raises(ValueError):
        unit_vectors(vectors, 2)


def test_wrong_batch_size():
    with pytest.raises(ValueError):
        unit_vectors([[1., 0.]], 2, 2)


def test_whole_task_aggregation_not_single_best_chunk():
    backend = FakeBackend()
    source = corpus(('a-only', 'aa', 'pi', 'only-pass'), ('both', 'aabb', 'terminus-2', 'all-fail'))
    index = EmbeddingIndex(build_index(source, backend), backend)
    ranked, count = index.rank('aabb')
    assert count == 2
    assert ranked[0]['task_id'] == 'both'
    assert ranked[0]['score'] == pytest.approx(1)
    assert ranked[1]['score'] == pytest.approx(.5)


def test_task_score_weights_query_content_tokens():
    backend = FakeBackend()
    source = corpus(('a', 'aa', 'pi', 'only-pass'), ('b', 'bb', 'terminus-2', 'all-fail'))
    index = EmbeddingIndex(build_index(source, backend), backend)
    ranked, _ = index.rank('aab')
    assert ranked[0]['task_id'] == 'a'
    assert ranked[0]['score'] == pytest.approx(2/3)


def test_deterministic_ties_and_exclusion():
    backend = FakeBackend()
    source = corpus(('z', 'aa', 'pi', 'only-pass'), ('a', 'ab', 'terminus-2', 'all-fail'))
    index = EmbeddingIndex(build_index(source, backend), backend)
    assert index.rank('aa')[0][0]['task_id'] == 'a'
    assert index.rank('aa', exclude_task_ids={'a'})[0][0]['task_id'] == 'z'


def test_exclusion_removes_all_task_chunks_and_identical_prompts():
    backend = FakeBackend()
    source = corpus(('original', 'aabb', 'pi', 'only-pass'),
                    ('duplicate', 'aabb', 'pi', 'only-pass'), ('other', 'bb', 'terminus-2', 'all-fail'))
    index = EmbeddingIndex(build_index(source, backend), backend)
    ranks, _ = index.rank('aabb', exclude_task_ids={'original'})
    assert [r['task_id'] for r in ranks] == ['other']
    assert index.rank('aabb', exclude_task_ids={'original', 'other'})[0] == []


def test_no_metadata_sent_to_embedding_and_no_plaintext_saved():
    backend, artifact = fixture_index()
    assert set(backend.calls) == {'aa', 'bb'}
    assert all('instruction' not in e and 'text' not in e for e in artifact['payload']['entries'])


def test_threshold_boundary_na_and_per_call_diagnostics():
    backend, artifact = fixture_index()
    router = EmbeddingRouter(EmbeddingIndex(artifact, backend), min_similarity=.6)
    first = router.decide('aa')
    assert first.as_tuple() == ('pi', 'z')
    assert router.route('aabb') == ('NA', 'NA')
    assert router.decide('aabb').diagnostics['reason'] == 'low_similarity'
    assert first.diagnostics['reason'] == 'matched'  # not overwritten by next call
    assert router.decide('').diagnostics['reason'] == 'empty_input'
    boundary = EmbeddingRouter(router.index, min_similarity=.5)
    assert boundary.route('aabb') == ('terminus-2', 'b')


@pytest.mark.parametrize('threshold', [None, True, '0.8', float('nan'), float('inf'), -1.1, 1.1])
def test_no_implicit_or_invalid_threshold(threshold):
    backend, artifact = fixture_index()
    with pytest.raises(ValueError):
        EmbeddingRouter(EmbeddingIndex(artifact, backend), min_similarity=threshold)


def test_threshold_required():
    backend, artifact = fixture_index()
    with pytest.raises(TypeError):
        EmbeddingRouter(EmbeddingIndex(artifact, backend))


def test_roundtrip_and_no_overwrite(tmp_path):
    backend, artifact = fixture_index()
    path = tmp_path / 'index.json'
    save_index(artifact, path)
    assert EmbeddingIndex.load(path, backend).index_id == artifact['index_id']
    with pytest.raises(FileExistsError):
        save_index(artifact, path)


def test_corrupt_index(tmp_path):
    backend, artifact = fixture_index()
    artifact['payload']['entries'][0]['harness'] = 'pi'
    with pytest.raises(ValueError, match='checksum'):
        EmbeddingIndex(artifact, backend)


@pytest.mark.parametrize('mutation', ['schema', 'policy', 'backend', 'empty', 'duplicate', 'label', 'shape', 'tokens'])
def test_invalid_index_with_valid_envelope(mutation):
    backend, artifact = fixture_index()
    p = artifact['payload']
    if mutation in {'schema', 'policy', 'backend'}:
        p[mutation] = 'invalid'
    elif mutation == 'empty':
        p['entries'] = []
    elif mutation == 'duplicate':
        p['entries'].append(copy.deepcopy(p['entries'][0]))
    elif mutation == 'label':
        p['entries'][0]['harness'] = 'invalid'
    elif mutation == 'shape':
        p['entries'][0]['vectors'] = [[1]]
    elif mutation == 'tokens':
        p['entries'][0]['chunk_tokens'] = [-1]
    artifact['index_id'] = digest(p)
    with pytest.raises(ValueError):
        EmbeddingIndex(artifact, backend)


def test_stale_sources():
    backend, artifact = fixture_index()
    index = EmbeddingIndex(artifact, backend)
    index.check_sources(corpus(('z', 'aa', 'pi', 'all-fail')))
    with pytest.raises(ValueError, match='source'):
        index.check_sources(PromptCorpus((), 'changed', 'rules'))


def test_embedding_errors_not_mistaken_for_low_score():
    backend, artifact = fixture_index()
    index = EmbeddingIndex(artifact, backend)
    def fail(_):
        raise RuntimeError('encoder failure')
    backend.embed = fail
    with pytest.raises(RuntimeError):
        EmbeddingRouter(index, min_similarity=.5).route('aa')
