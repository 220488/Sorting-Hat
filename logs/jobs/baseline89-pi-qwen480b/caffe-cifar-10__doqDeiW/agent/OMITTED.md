# pi.txt.gz omitted from this repository

The full agent log for this trial is **1.8 GB raw / 177 MB gzipped** — above
GitHub's 100 MB per-file hard limit, so it cannot be committed.

What it is: `pi` running away on `caffe-cifar-10`. 22,147 agent turns averaging
~85 KB each, full build output embedded in every turn. The trial consumed
4,179,089 tokens and $1.9027, and failed. It is the clearest single artifact
behind the per-run token-ceiling argument in the cost analysis.

Kept here instead:

- `pi.head-300-lines.txt.gz` — the opening 300 lines (setup, first turns)
- `pi.tail-300-lines.txt.gz` — the closing 300 lines (how it ended)

The complete file lives at:

- `aws/logs/jobs/baseline89-pi-qwen480b/caffe-cifar-10__doqDeiW/agent/pi.txt.gz`
  in the local working copy
- `~/jobs/baseline89-pi-qwen480b/caffe-cifar-10__doqDeiW/agent/pi.txt`
  on the EC2 host, until that instance is terminated

If it is needed in version control, Git LFS is the right mechanism.
