"""Shared strategy loading and per-call diagnostics for both Harbor wrappers."""
import importlib
import inspect
import time


def load_route(strategy, registry, routing_kwargs=None):
    module_name, attribute = registry[strategy].split(':')
    module = importlib.import_module(module_name)
    if strategy == 'embedding':
        if not isinstance(routing_kwargs, dict):
            raise ValueError('embedding requires routing_kwargs: model_dir, index_path, min_similarity')
        # Configuration errors fail before running a paid solver, not silently NA.
        return getattr(module, attribute)(**routing_kwargs).decide
    if routing_kwargs:
        raise ValueError('routing_kwargs are supported only for the embedding strategy')
    return getattr(module, attribute)


async def call_route(route_func, instruction):
    started = time.perf_counter()
    try:
        result = route_func(instruction)
        if inspect.isawaitable(result):
            result = await result
        if hasattr(result, 'as_tuple') and hasattr(result, 'diagnostics'):
            assigned, matched = result.as_tuple()
            details = dict(result.diagnostics)
        else:
            assigned, matched = result
            details = {'reason': 'strategy_returned_na' if assigned == 'NA' else 'matched'}
        if not isinstance(assigned, str) or not isinstance(matched, str):
            raise TypeError('Router must return two strings')
        details['call_latency_ms'] = (time.perf_counter()-started)*1000
        return assigned, matched, details
    except Exception as exc:
        # Preserve existing runtime NA fallback, while separating errors from weak matches.
        # Do not log exception text: it can contain credentials or instruction text.
        return 'NA', 'NA', {'reason': 'routing_error', 'error_type': type(exc).__name__,
                            'call_latency_ms': (time.perf_counter()-started)*1000}
