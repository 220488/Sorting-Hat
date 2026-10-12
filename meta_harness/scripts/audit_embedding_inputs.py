"""Offline tokenizer-only input audit. Does not run task prompts or a solver.

Run from meta_harness: python scripts/audit_embedding_inputs.py --help
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--rules', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir.absolute()
    if not model_dir.is_dir():
        parser.error('--model-dir must be a local directory')
    if args.output.exists():
        parser.error('--output already exists; choose a new result path')
    os.environ.update({
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
    })
    def block_network(event, _args):
        if event in {'socket.connect', 'socket.getaddrinfo'}:
            raise RuntimeError('Network disabled during local tokenizer audit')
    sys.addaudithook(block_network)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from routing_strategy.embedding_data import load_corpus, audit_token_lengths
    from transformers import AutoTokenizer

    # Inspect local config limits, never allow trust_remote_code or model downloads.
    config = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    sentence_config = json.loads((model_dir / 'sentence_bert_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    limit = min(config['max_position_embeddings'], sentence_config['max_seq_length'], tokenizer.model_max_length)
    corpus = load_corpus(args.csv, args.rules)
    result = audit_token_lengths(corpus, tokenizer, limit)
    result['model_config_sha256'] = hashlib.sha256((model_dir / 'config.json').read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'records'}, indent=2))
    print(json.dumps([row for row in result['records'] if row['exceeds_limit']], indent=2))


if __name__ == '__main__':
    main()
