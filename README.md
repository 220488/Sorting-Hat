# Sorting Hat — baseline run logs

Execution logs from the four-harness Terminal-Bench 2.1 baseline for capstone
project #16 (meta-harness router).

All four harnesses ran the same 89 tasks against one constant model,
`qwen.qwen3-coder-480b-a35b-v1:0` on Amazon Bedrock (`us-east-2`), under
Harbor 0.22.0 on an EC2 `m7i.xlarge`. **356 trials.**

| Harness | Solved | Unique solves | Cost | Errors |
|---|---|---|---|---|
| mini-swe-agent | 28/89 | 5 | $21.97 | 5 |
| terminus-2 | 22/89 | 1 | $30.89 | 2 |
| opencode | 18/89 | 0 | $34.78 | 15 |
| pi | 16/89 | 2 | $16.27 | 5 |

Best single harness 28/89; oracle (any harness solves) 35/89. The 7-task gap is
the routing headroom the project competes for.

See [`logs/README.md`](logs/README.md) for layout, the error taxonomy, and
reproduction notes.

## Contents

```
logs/
├── README.md              full documentation
├── baseline89*.log        Harbor summary table per run
└── jobs/<job>/<task>__<id>/
    ├── trial.log          setup, agent invocation, verifier
    ├── exception.txt      traceback (errored trials only)
    └── agent/*.txt.gz     agent output, gzipped above 1 MB
```

Agent logs are gzipped — read with `gzip -dc <file> | head -50`.

One file is omitted: `pi`'s `caffe-cifar-10` log is 1.8 GB raw / 177 MB gzipped,
above GitHub's 100 MB limit. Head and tail samples are committed in its place —
see `OMITTED.md` in that directory.
