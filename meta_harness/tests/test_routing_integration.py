import asyncio
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sorting_hat.routing import call_route, load_route
from sorting_hat.sorting_hat import SortingHat
from sorting_hat.sorting_hat_no_retry import SortingHatNoRetry
from harbor.models.agent.context import AgentContext
from routing_strategy.embedding import Decision


def test_sync_async_and_failure_route_contract():
    async def async_route(_):
        return 'pi', 'some-task'
    assert asyncio.run(call_route(lambda _: ('pi', 'some-task'), 'x'))[:2] == ('pi', 'some-task')
    assert asyncio.run(call_route(async_route, 'x'))[:2] == ('pi', 'some-task')
    def failure(_):
        raise RuntimeError('do not log secrets')
    a, b, detail = asyncio.run(call_route(failure, 'x'))
    assert (a, b) == ('NA', 'NA')
    assert detail['reason'] == 'routing_error'
    assert 'do not log secrets' not in json.dumps(detail)
    assert asyncio.run(call_route(lambda _: (None, None), 'x'))[2]['reason'] == 'routing_error'


def test_keyword_loader_unchanged():
    route = load_route('rules_based', {'rules_based': 'routing_strategy.rules_based:route'})
    assert route('unrelated empty example') == ('NA', 'NA')
    with pytest.raises(ValueError):
        load_route('rules_based', {'rules_based': 'routing_strategy.rules_based:route'}, {'unused': True})


@pytest.mark.parametrize('cls', [SortingHat, SortingHatNoRetry])
def test_config_not_forwarded_to_harness(monkeypatch, tmp_path, cls):
    from routing_strategy import embedding
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(decide=lambda _: Decision('pi', 'x', {'reason': 'matched'}))
    monkeypatch.setattr(embedding, 'create_router', factory)
    kwargs = {'model_dir': 'local', 'index_path': 'index', 'min_similarity': .5}
    agent = cls(tmp_path, routing_strategy='embedding', routing_kwargs=kwargs)
    assert calls == [kwargs]
    assert 'routing_kwargs' not in agent.harbor_kwargs


class FakeHarness:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    async def setup(self, _):
        pass

    async def run(self, *_):
        if self.should_fail:
            raise RuntimeError('synthetic harness error')

    def version(self):
        return 'fake'

    def populate_context_post_run(self, context):
        pass


class FakeEnvironment:
    async def prepare_logs_for_host(self):
        pass


@pytest.mark.parametrize('cls', [SortingHat, SortingHatNoRetry])
@pytest.mark.parametrize('mode', ['matched', 'low_similarity', 'routing_error', 'forced'])
def test_wrapper_routing_logs_and_fallback(monkeypatch, tmp_path, cls, mode):
    module = sys.modules[cls.__module__]
    installed = []
    def create(**kwargs):
        installed.append(kwargs['name'])
        assert 'routing_kwargs' not in kwargs
        return FakeHarness()
    monkeypatch.setattr(module.AgentFactory, 'create_agent_from_name', create)
    agent = cls(tmp_path, force_harness='terminus-2' if mode == 'forced' else None)
    def decide(_):
        if mode == 'routing_error':
            raise ValueError('synthetic exception')
        return Decision('pi' if mode == 'matched' else 'NA', 'task' if mode == 'matched' else 'NA', {'reason': mode})
    agent.route_func = decide
    asyncio.run(agent.run('synthetic input', FakeEnvironment(), AgentContext()))
    log = json.loads((tmp_path / 'sorting_hat_log.json').read_text())
    expected = 'terminus-2' if mode == 'forced' else ('pi' if mode == 'matched' else 'mini-swe-agent')
    assert installed == [expected]
    assert log['assigned_harness'] == expected
    assert log['fallback_used'] == (mode in {'low_similarity', 'routing_error'})
    assert log['routing_details']['reason'] == mode


@pytest.mark.parametrize('first,second', [('pi','mini-swe-agent'), ('mini-swe-agent','terminus-2'), ('terminus-2','mini-swe-agent')])
def test_retry_map_preserved(monkeypatch, tmp_path, first, second):
    module = sys.modules[SortingHat.__module__]
    installed = []
    def create(**kwargs):
        installed.append(kwargs['name'])
        return FakeHarness(should_fail=len(installed) == 1)
    monkeypatch.setattr(module.AgentFactory, 'create_agent_from_name', create)
    agent = SortingHat(tmp_path, force_harness=first)
    asyncio.run(agent.run('synthetic input', FakeEnvironment(), AgentContext()))
    assert installed == [first, second]
    log = json.loads((tmp_path / 'sorting_hat_log.json').read_text())
    assert log['retry']['assigned_harness'] == second


def test_no_retry_wrapper_still_propagates_harness_error(monkeypatch, tmp_path):
    module = sys.modules[SortingHatNoRetry.__module__]
    installed = []
    def create(**kwargs):
        installed.append(kwargs['name'])
        return FakeHarness(should_fail=True)
    monkeypatch.setattr(module.AgentFactory, 'create_agent_from_name', create)
    agent = SortingHatNoRetry(tmp_path, force_harness='pi')
    with pytest.raises(RuntimeError, match='synthetic harness error'):
        asyncio.run(agent.run('synthetic input', FakeEnvironment(), AgentContext()))
    assert installed == ['pi']


def test_per_call_async_diagnostics_do_not_leak():
    async def decide(instruction):
        await asyncio.sleep(0)
        return Decision('pi', instruction, {'reason': 'matched', 'nearest_task': instruction})
    async def together():
        return await asyncio.gather(call_route(decide, 'a'), call_route(decide, 'b'))
    first, second = asyncio.run(together())
    assert first[1] == first[2]['nearest_task'] == 'a'
    assert second[1] == second[2]['nearest_task'] == 'b'
