# Run configurations

The four Harbor job configs behind the baseline. Each ran all 89 Terminal-Bench 2.1
tasks against the same constant model, `qwen.qwen3-coder-480b-a35b-v1:0` on Amazon
Bedrock (`us-east-2`), so the harness is the only variable.

**356 trials.** Parsed results are in `aws/baseline_4harness.csv`; execution logs in
`aws/logs/`.

## The four configs

| Config | Harness | Version | Provider prefix | Solved | Errors | Cost |
|---|---|---|---|---|---|---|
| `baseline89-terminus2.json` | terminus-2 | 2.0.0 | `bedrock/` | 22/89 | 2 | $30.89 |
| `baseline89-miniswe.json` | mini-swe-agent | 2.4.6 | `bedrock/` | 28/89 | 5 | $21.97 |
| `baseline89-pi.json` | pi | 0.85.1 | `bedrock/` | 16/89 | 5 | $16.27 |
| `baseline89-opencode.json` | opencode | 1.18.31 | `amazon-bedrock/` | 18/89 | 15 | $34.78 |

When each ran (UTC):

```
terminus-2       2026-09-15 03:34 -> 07:54
pi               2026-09-15 12:13 -> 16:05
mini-swe-agent   2026-09-15 23:19 -> 2026-09-16 02:45
opencode         2026-09-17 00:04 -> 05:20
```

## Job-level run records

Alongside each config sits the `result.json` Harbor wrote for that run — a summary
of the job, not of any task. These are outputs, kept here so a config sits next to
the record of what it produced.

| File | Trials | Errored | Retries | Wall clock |
|---|---|---|---|---|
| `baseline89-terminus2.result.json` | 89 | 2 | 0 | 4.3 h |
| `baseline89-miniswe.result.json` | 89 | 5 | 0 | 3.4 h |
| `baseline89-pi.result.json` | 89 | 5 | 0 | 3.9 h |
| `baseline89-opencode.result.json` | 89 | 15 | 0 | 5.3 h |

Each holds `id`, `started_at`, `finished_at`, `n_total_trials`, and a `stats` block
(completed / errored / running / pending / cancelled / retries).

**All four ran with zero retries.** Every trial executed exactly once, so no result
is a second attempt. That also means the two trials Bedrock throttled
(`ApiOverloadedError`, `ApiRateLimitError`) are recorded as failures rather than
being re-attempted — Harbor's `max_retries` default is 0.

Wall clock here is job-level elapsed time, which includes scheduling gaps that
summing per-trial durations misses. Total across the four: **16.9 hours**.

Per-task outcomes are not in these files. They live in
`aws/results/<job>/<task>__<id>/result.json` — 356 of them — and are what
`aws/script/parse_full.py` reads.

## Shape of a config

```json
{
  "job_name": "baseline89-terminus2-qwen480b",
  "n_concurrent_trials": 2,
  "agents": [
    { "name": "terminus-2",
      "model_name": "bedrock/qwen.qwen3-coder-480b-a35b-v1:0" }
  ],
  "datasets": [
    { "name": "terminal-bench/terminal-bench-2-1",
      "ref": "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a",
      "task_names": ["terminal-bench/adaptive-rejection-sampler", "... 88 more"] }
  ]
}
```

`job_name` names the output directory: `baseline89-terminus2.json` writes results to
`aws/results/baseline89-terminus2-qwen480b/`. The `-qwen480b` suffix records the
constant model, so a second model can be run without collision.

`ref` pins the dataset by content hash. All four runs share it — results are
comparable because the tasks are provably identical.

No timeout or retry overrides were set; every run used Harbor 0.22.0 defaults.
Harbor copies the config it is given into the job directory verbatim, so
`aws/results/<job>/config.json` is byte-identical to the file here.

## The opencode difference

Three harnesses use `bedrock/`, which routes through LiteLLM and authenticates with
SigV4 from the EC2 instance role. opencode cannot: it is a TypeScript application
with its own HTTP client and no AWS request signing.

It uses `amazon-bedrock/` instead — its own native provider id, via the Vercel AI
SDK's `@ai-sdk/amazon-bedrock`. Two things about it are easy to get wrong:

- The prefix must be `amazon-bedrock/`, not Harbor's `bedrock/`. With the wrong id
  opencode resolves an unknown provider and fails with
  `"undefined/chat/completions" cannot be parsed as a URL`.
- Credentials must be the standard `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
  `AWS_SESSION_TOKEN` trio. **Harbor does not forward `AWS_BEARER_TOKEN_BEDROCK`**
  into the agent container, so a Bedrock API key alone produces
  `AWS SigV4 authentication requires AWS credentials`.

Routing opencode through the `openai/` provider against Bedrock's OpenAI-compatible
endpoint does not work either — opencode calls `/responses`, and Bedrock implements
only `/chat/completions`.

## Running them

From the project root, on a host with Docker and Bedrock access:

```bash
# the three SigV4 harnesses — instance role supplies credentials
harbor run --config aws/config/baseline89-terminus2.json
harbor run --config aws/config/baseline89-miniswe.json
harbor run --config aws/config/baseline89-pi.json

# opencode — export role credentials into the environment first
harbor run --config aws/config/baseline89-opencode.json
```

Then parse:

```bash
python3 aws/script/parse_full.py aws/results/baseline89-* -o aws/baseline_4harness.csv
```

## Environment these were run on

EC2 `m7i.xlarge` (4 vCPU / 16 GB), `us-east-2`, Harbor 0.22.0, concurrency 2.
Wall clock was 4–9 hours per harness; total inference cost $103.90.

Host choice affects results. Four tasks cannot pass in this environment regardless
of harness — three `qemu` tasks need `/dev/kvm`, which this instance type does not
expose, and `code-from-image` needs vision, which Qwen3-Coder-480B does not have.
Two opencode trials were also killed by the OOM reaper at 16 GB. A larger or
metal instance would change all of these.
