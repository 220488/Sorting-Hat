"""Offline build, exploratory analysis and routing for the embedding prototype."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    subparsers = parser.add_subparsers(dest='command', required=True)
    build = subparsers.add_parser('build')
    analyze = subparsers.add_parser('analyze')
    route = subparsers.add_parser('route')
    for command in (build, analyze):
        command.add_argument('--csv', type=Path, required=True)
        command.add_argument('--rules', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    analyze.add_argument('--index', type=Path, required=True)
    analyze.add_argument('--output', type=Path, required=True)
    route.add_argument('--index', type=Path, required=True)
    route.add_argument('--instruction-file', type=Path, required=True)
    route.add_argument('--min-similarity', type=float, required=True)
    args = parser.parse_args()
    if args.command in {'build', 'analyze'} and args.output.exists():
        parser.error('Output exists; choose a new versioned path')
    os.environ.update({
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'TOKENIZERS_PARALLELISM': 'false',
    })
    def block_network(event, _args):
        if event in {'socket.connect', 'socket.getaddrinfo'}:
            raise RuntimeError('Network disabled for embedding CLI')
    sys.addaudithook(block_network)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from routing_strategy.embedding import LocalBGE, EmbeddingIndex, EmbeddingRouter, build_index, save_index
    from routing_strategy.embedding_data import load_corpus

    started = time.perf_counter()
    backend = LocalBGE(args.model_dir)
    print(json.dumps({'model_loaded_seconds': round(time.perf_counter()-started, 3)}), flush=True)
    if args.command == 'build':
        source = load_corpus(args.csv, args.rules)
        index = build_index(source, backend)
        EmbeddingIndex(index, backend)  # Validate before writing an artifact.
        save_index(index, args.output)
        print(json.dumps({'index_id': index['index_id'], 'tasks': len(index['payload']['entries']),
                          'chunks': sum(len(e['vectors']) for e in index['payload']['entries']),
                          'elapsed_seconds': round(time.perf_counter()-started, 3)}), flush=True)
    elif args.command == 'route':
        index = EmbeddingIndex.load(args.index, backend)
        decision = EmbeddingRouter(index, min_similarity=args.min_similarity).decide(
            args.instruction_file.read_text(encoding='utf-8'))
        print(json.dumps({'assigned_harness': decision.harness, 'matched_rule': decision.matched_rule,
                          'diagnostics': decision.diagnostics}, indent=2))
    else:
        from routing_strategy.rules_based import route as keyword_route
        source = load_corpus(args.csv, args.rules)
        index = EmbeddingIndex.load(args.index, backend)
        index.check_sources(source)
        rows = []
        for number, record in enumerate(source.records, start=1):
            self_rank, query_chunks = index.rank(record.instruction)
            t0 = time.perf_counter()
            other_rank, _ = index.rank(record.instruction,
                exclude_task_ids={record.task_id}, exclude_prompt_hashes={record.instruction_sha256})
            latency_ms = (time.perf_counter()-t0)*1000
            if not other_rank:
                raise ValueError('Task-excluded analysis requires at least two distinct prompts')
            keyword_harness, keyword_match = keyword_route(record.instruction)
            rows.append({
                'task_id': record.task_id, 'source_label_case': record.label_case,
                'source_harness': record.harness, 'query_chunks': query_chunks,
                'self_match': self_rank[0]['task_id'] == record.task_id,
                'self_score': self_rank[0]['score'],
                'keyword_harness': keyword_harness, 'keyword_match': keyword_match,
                'excluded_top': other_rank[0],
                'excluded_label_agreement': other_rank[0]['harness'] == record.harness,
                'excluded_latency_ms': latency_ms,
            })
            if number % 15 == 0:
                print(json.dumps({'analyzed_tasks': number, 'total': len(source.records)}), flush=True)
        scores = sorted(row['excluded_top']['score'] for row in rows)
        cutoffs = sorted(set(scores[int((len(scores)-1)*q)] for q in (0, .25, .5, .75, 1)))
        summary = {
            'tasks': len(rows), 'chunks': sum(len(v) for _, v in index.entries),
            'self_match_tasks': sum(row['self_match'] for row in rows),
            'keyword_self_match_tasks': sum(row['keyword_match'] == row['task_id'] for row in rows),
            'task_excluded_label_agreement': sum(row['excluded_label_agreement'] for row in rows),
            'task_excluded_scores': {'min': scores[0], 'median': statistics.median(scores), 'max': scores[-1]},
            'task_excluded_selected_harnesses': dict(Counter(row['excluded_top']['harness'] for row in rows)),
            'task_excluded_median_latency_ms': statistics.median(row['excluded_latency_ms'] for row in rows),
            'score_quantile_cutoff_examples_not_recommendations': [
                {'cutoff': cutoff, 'accepted': sum(score >= cutoff for score in scores),
                 'would_fallback': sum(score < cutoff for score in scores)} for cutoff in cutoffs
            ],
            'threshold_selected': None,
        }
        report = {
            'index_id': index.index_id, 'summary': summary, 'rows': rows,
            'limitations': [
                'Self-match checks wiring only; not success rate or unseen-task accuracy.',
                'All chunks and exact duplicate prompts of the excluded task are removed.',
                'The 89 tasks were already exposed during team development; this is not a clean held-out test.',
                'Label agreement measures inherited labels, including all-fail choices, not task-solving success.',
                'Keyword self-match uses all its original rules and is not a fair unseen-task comparator.',
                'Scores do not measure harness suitability probabilities; no cutoff selected or tuned here.',
                'No solver, Docker, AWS or paid API ran; timing is warm local CPU routing only.',
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
