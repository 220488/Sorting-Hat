# Baseline run logs

Raw execution logs for the four-harness Terminal-Bench 2.1 baseline. Every harness
ran all 89 tasks against the same constant model, `qwen.qwen3-coder-480b-a35b-v1:0`
on Amazon Bedrock (`us-east-2`), under Harbor 0.22.0 on an EC2 `m7i.xlarge`
(4 vCPU / 16 GB), concurrency 2.

**356 trials total.** Parsed results live one level up in `aws/baseline_4harness.csv`;
this folder holds the unparsed record behind them.

## Contents

| Harness | Job directory | Run stdout | Errors | Size |
|---|---|---|---|---|
| terminus-2 | `jobs/baseline89-terminus2-qwen480b/` | `baseline89.log` | 2 | 2.3 M |
| mini-swe-agent | `jobs/baseline89-miniswe-qwen480b/` | `baseline89ms.log` | 5 | 6.3 M |
| pi | `jobs/baseline89-pi-qwen480b/` | `baseline89pi.log` | 5 | 242 M |
| opencode | `jobs/baseline89-opencode-qwen480b/` | `baseline89oc.log` | 15 | 14 M |

When each ran (UTC):

```
terminus-2       2026-09-15 03:34 -> 07:54
pi               2026-09-15 12:13 -> 16:05
mini-swe-agent   2026-09-15 23:19 -> 2026-09-16 02:45
opencode         2026-09-17 00:04 -> 05:20
```

## Layout

```
aws/logs/
├── baseline89*.log                  Harbor's summary table per run (small)
└── jobs/<job-name>/
    ├── job.log                      job-level log: setup, scheduling, failures
    └── <task>__<id>/
        ├── trial.log                setup commands, agent invocation, verifier
        ├── exception.txt            Python traceback — errored trials only
        └── agent/<harness>.txt(.gz) the agent's own output, where it writes one
```

`<id>` is Harbor's random per-trial suffix, e.g. `fix-git__kbMTiUn`. It matches the
`trial_name` column in the CSVs, so a row can always be traced back to its logs.

terminus-2 writes no per-trial agent log; the other three do (87 of 89 each — the
two missing are trials that died before the agent started).

## Compression

Agent logs over 1 MB are gzipped. This folder is 265 MB compressed, 2.3 GB raw.

```bash
zcat  jobs/baseline89-pi-qwen480b/<task>__<id>/agent/pi.txt.gz | head -50
zgrep -i error jobs/baseline89-opencode-qwen480b/*/agent/*.gz
```

Everything else — `trial.log`, `job.log`, `exception.txt` — is plain text.

`pi` is 242 MB of the 265 MB, almost all of it from runaway trials. Its
`caffe-cifar-10` log alone is 1.8 GB raw: 22,147 agent turns averaging ~85 KB each,
full build output embedded in every turn. That file is the clearest single piece of
evidence for the per-run token ceiling discussed in the cost analysis.

## The 27 errored trials

```
opencode         NonZeroAgentExitCodeError   12
opencode         AgentTimeoutError            2
opencode         ApiRateLimitError            1
mini-swe-agent   AgentTimeoutError            2
mini-swe-agent   NonZeroAgentExitCodeError    2
mini-swe-agent   ApiOverloadedError           1
pi               NonZeroAgentExitCodeError    3
pi               AgentTimeoutError            2
terminus-2       VerifierTimeoutError         1
terminus-2       AgentTimeoutError            1
```

Not all errors mean the same thing, and the distinction matters when reporting
pass rates:

- **`VerifierTimeoutError`** — the agent finished; the *verifier* hung and was killed,
  so no reward was ever produced. Says nothing about harness capability. Exclude.
- **`AgentTimeoutError`** — the agent ran to its wall-clock limit without converging.
  A genuine capability failure; count it as one.
- **`NonZeroAgentExitCodeError`** — the harness process exited non-zero. Cause varies;
  read `agent/*.txt` for the underlying message.
- **`ApiOverloadedError` / `ApiRateLimitError`** — Bedrock throttled the request.
  Infrastructure, not capability. Candidates for a re-run.

Most of opencode's 12 non-zero exits are Bedrock **"Mantle streaming"** errors, each
carrying an AWS request id. opencode streams model responses where the other three do
not, so it alone meets this failure mode. Two more are exit 137 (SIGKILL — the OOM
reaper) on memory-heavy tasks.

## Environment-blocked tasks

Four tasks cannot pass in this environment regardless of harness, and their logs show
why rather than an ordinary failure:

| Task | Reason |
|---|---|
| `qemu-alpine-ssh`, `qemu-startup`, `install-windows-3.11` | no `/dev/kvm` on `m7i.xlarge` |
| `code-from-image` | Qwen3-Coder-480B has no vision capability |

`code-from-image` is only visible as such in opencode's log, which reports
*"this model doesn't support the image content block"*. The other three harnesses
simply scored zero.

## Reproducing

Configs are in `aws/baseline89*.json`. Re-running requires an EC2 host with Docker
and Bedrock access:

```bash
harbor run --config aws/baseline89.json          # terminus-2
harbor run --config aws/baseline89-opencode.json # opencode
```

terminus-2, mini-swe-agent and pi use `bedrock/<model>` with SigV4 via an IAM
instance role. opencode uses `amazon-bedrock/<model>` and needs the standard
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` trio forwarded
into the container — Harbor does not forward `AWS_BEARER_TOKEN_BEDROCK`.
