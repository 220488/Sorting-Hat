"""Per-task routing rules: instruction text in, harness out.

Usage:
    python routing_strategy/rules_based.py "<full text of an instruction.md>"
    -> prints the harness and the task whose rule matched, or "NA  NA"

    from routing_strategy.rules_based import route
    harness, task_id = route(instruction_text)

How a rule works:
    The router only ever sees the instruction text.

    One rule per task (89 tasks). IF the rule's keyword(s) appear in the
    instruction, THEN the task goes to that rule's harness. A rule has one
    keyword or several; with several, all of them must appear. Matching is
    case-insensitive and whole-word. A keyword can also be a phrase, and a
    space in a phrase also matches a line break in the instruction.

    Rules are checked top to bottom and the first match wins.


How the keywords were chosen:
    Keywords are taken only from the task's own instruction, the same text the
    router sees.

    Each instruction was read and its most distinctive words or phrases were
    chosen, usually the tool, method or subject it asks for (e.g. "build the
    CompCert C verified compiler" -> "CompCert"). Each rule matches its own
    instruction.


How each task's harness was chosen:
    From the Qwen baseline comparison (one trial per task per harness):
    harnesses that passed first (reward == 1); among those, lowest total_tokens,
    then lowest agent_exec_s. If none passed, the same tokens-then-time order
    over the harnesses that ran without an error.
    The "case" field records which applied:
      only-pass   exactly one harness passed
      multi-pass  several passed; the one with the fewest tokens was chosen
      all-fail    none passed; the one with the fewest tokens that ran without an error was chosen

Instructions from outside these 89 tasks:
    Usually NA. But an instruction that happens to contain a rule's keywords
    gets that rule's harness.

Editing:
    You can change any rule's keywords by hand. A single keyword is written
    ("word",), with the trailing comma; ("word") also works.
    Don't rebuild with make_keyword_rules.py: it picks keywords from the task
    name, task.toml and description, and would overwrite these rules.
    Afterwards, run test_router.py and check that every row's matched_rule
    equals its task_id. Rows whose rule is still TO_BE_FILLED show NA.

Built by using:
    Qwen_Router_Data_2026-09-23.xlsx, tb2.1_tasks.csv 
"""
import re
import sys


# (task_id, keywords that must ALL appear, harness, case)
RULES = [
    # ---- keywords chosen from the instruction ----
    ("adaptive-rejection-sampler", ('rejection sampler',), "mini-swe-agent", "all-fail"),
    ("bn-fit-modify", ('Bayesian Network',), "mini-swe-agent", "all-fail"),
    ("build-cython-ext", ('Cython',), "terminus-2", "all-fail"),
    ("build-pmars", ('pMARS', 'X11'), "mini-swe-agent", "multi-pass"),
    ("build-pov-ray", ('POV-Ray',), "terminus-2", "all-fail"),
    ("caffe-cifar-10", ('Caffe',), "pi", "all-fail"),
    ("cancel-async-tasks", ('concurrently', 'cancel'), "terminus-2", "only-pass"),
    ("chess-best-move", ('chess board', 'best move'), "pi", "all-fail"),
    ("circuit-fibsqrt", ('logic-gate simulator',), "mini-swe-agent", "all-fail"),
    ("cobol-modernization", ('COBOL',), "mini-swe-agent", "multi-pass"),
    ("code-from-image", ('pseudocode',), "mini-swe-agent", "only-pass"),
    ("compile-compcert", ('CompCert',), "pi", "only-pass"),
    ("configure-git-webserver", ('git server', 'webserver'), "mini-swe-agent", "multi-pass"),
    ("constraints-scheduling", ('meeting slot',), "mini-swe-agent", "multi-pass"),
    ("count-dataset-tokens", ('tokenizer', 'dataset'), "terminus-2", "only-pass"),
    ("custom-memory-heap-crash", ('RELEASE mode',), "mini-swe-agent", "only-pass"),
    ("db-wal-recovery", ('WAL file',), "terminus-2", "all-fail"),
    ("distribution-search", ('KL divergence',), "pi", "all-fail"),
    ("dna-assembly", ('Golden Gate',), "pi", "all-fail"),
    ("dna-insert", ('site-directed mutagenesis',), "mini-swe-agent", "all-fail"),
    ("extract-elf", ('memory values',), "terminus-2", "multi-pass"),
    ("extract-moves-from-video", ('video', 'zork'), "mini-swe-agent", "all-fail"),
    ("feal-differential-cryptanalysis", ('differential attack',), "pi", "all-fail"),
    ("feal-linear-cryptanalysis", ('linear attack',), "mini-swe-agent", "all-fail"),
    ("financial-document-processor", ('invoice',), "mini-swe-agent", "all-fail"),
    ("fix-git", ('checked out master', 'merge'), "pi", "multi-pass"),
    ("fix-ocaml-gc", ('OCaml', 'garbage collector'), "mini-swe-agent", "only-pass"),
    ("gcode-to-text", ('prusa',), "terminus-2", "all-fail"),
    ("git-leak-recovery", ('secret', 'rewriting history'), "mini-swe-agent", "multi-pass"),
    ("git-multibranch", ('Git server', 'branch'), "mini-swe-agent", "multi-pass"),
    ("gpt2-codegolf", ('GPT-2',), "terminus-2", "all-fail"),
    ("headless-terminal", ('headless terminal',), "mini-swe-agent", "only-pass"),
    ("hf-model-inference", ('sentiment analysis',), "terminus-2", "multi-pass"),
    ("install-windows-3.11", ('Windows 3.11',), "terminus-2", "all-fail"),
    ("kv-store-grpc", ('KV store',), "pi", "only-pass"),
    ("large-scale-text-editing", ('Vim macros',), "terminus-2", "all-fail"),
    ("largest-eigenval", ('eigenvalue',), "mini-swe-agent", "only-pass"),
    ("llm-inference-batching-scheduler", ('batching scheduler',), "terminus-2", "all-fail"),
    ("log-summary-date-ranges", ('severity', 'date ranges'), "pi", "multi-pass"),
    ("mailman", ('mailing list',), "terminus-2", "only-pass"),
    ("make-doom-for-mips", ('doom', 'build', 'for me'), "pi", "all-fail"),
    ("make-mips-interpreter", ('MIPS interpreter',), "terminus-2", "all-fail"),
    ("mcmc-sampling-stan", ('hierarchical Bayesian model',), "mini-swe-agent", "all-fail"),
    ("merge-diff-arc-agi-task", ('git bundle',), "mini-swe-agent", "only-pass"),
    ("model-extraction-relu-logits", ('ReLU neural network',), "pi", "only-pass"),
    ("modernize-scientific-stack", ('legacy Python',), "pi", "multi-pass"),
    ("mteb-leaderboard", ('MTEB leaderboard',), "pi", "all-fail"),
    ("mteb-retrieve", ('cosine similarity',), "pi", "all-fail"),
    ("multi-source-data-merger", ('source priority',), "pi", "multi-pass"),
    ("nginx-request-logging", ('request logging',), "mini-swe-agent", "multi-pass"),
    ("overfull-hbox", ('LaTeX',), "pi", "all-fail"),
    ("path-tracing", ('image', 'rendered'), "pi", "all-fail"),
    ("path-tracing-reverse", ('decompiling',), "pi", "all-fail"),
    ("polyglot-c-py", ('polyglot', 'gcc'), "pi", "all-fail"),
    ("polyglot-rust-c", ('rustc',), "mini-swe-agent", "only-pass"),
    ("portfolio-optimization", ('portfolio risk',), "mini-swe-agent", "only-pass"),
    ("protein-assembly", ('fusion protein',), "mini-swe-agent", "all-fail"),
    ("prove-plus-comm", ('Coq', 'proof'), "pi", "multi-pass"),
    ("pypi-server", ('pypi server',), "terminus-2", "multi-pass"),
    ("pytorch-model-cli", ('MNIST',), "mini-swe-agent", "all-fail"),
    ("pytorch-model-recovery", ('state dictionary',), "mini-swe-agent", "multi-pass"),
    ("qemu-alpine-ssh", ('qemu', 'ssh server'), "terminus-2", "all-fail"),
    ("qemu-startup", ('telnet',), "terminus-2", "all-fail"),
    ("query-optimize", ('sql query',), "mini-swe-agent", "multi-pass"),
    ("raman-fitting", ('Raman',), "terminus-2", "all-fail"),
    ("regex-chess", ('FEN',), "terminus-2", "all-fail"),
    ("regex-log", ('IPv4',), "pi", "all-fail"),
    ("reshard-c4-data", ('resharding',), "terminus-2", "all-fail"),
    ("rstan-to-pystan", ('PyStan',), "mini-swe-agent", "all-fail"),
    ("sam-cell-seg", ('Segment Anything',), "pi", "all-fail"),
    ("schemelike-metacircular-eval", ('metacircular evaluator',), "mini-swe-agent", "all-fail"),
    ("sparql-university", ('SPARQL',), "terminus-2", "all-fail"),
    ("sqlite-db-truncate", ('binary truncation',), "pi", "all-fail"),
    ("sqlite-with-gcov", ('gcov',), "mini-swe-agent", "multi-pass"),
    ("torch-pipeline-parallelism", ('pipeline parallel',), "mini-swe-agent", "all-fail"),
    ("torch-tensor-parallelism", ('tensor parallelism',), "terminus-2", "all-fail"),
    ("train-fasttext", ('fasttext',), "pi", "all-fail"),
    ("tune-mjcf", ('MuJoCo',), "pi", "all-fail"),
    ("video-processing", ('hurdle',), "pi", "all-fail"),
    ("winning-avg-corewars", ('CoreWars',), "terminus-2", "all-fail"),
    ("write-compressor", ('decompressor',), "mini-swe-agent", "all-fail"),
    ("break-filter-js-from-html", ('html', 'trigger a JavaScript alert()',), "pi", "all-fail"),
    ("crack-7z-hash", ('create a file', '/app/solution.txt', 'secret_file.txt',), "pi", "only-pass"),  
    ("filter-js-from-html", ('removes', 'JavaScript', 'from HTML files',), "terminus-2", "all-fail"),  
    ("fix-code-vulnerability", ('identify', 'fix', 'vulnerability', 'Common Weakness Enumeration', 'CWE',), "mini-swe-agent", "multi-pass"),  
    ("openssl-selfsigned-cert", ('create', 'self-signed certificate', 'OpenSSL',), "mini-swe-agent", "multi-pass"),  
    ("password-recovery", ('digital forensic recovery', 'password',), "mini-swe-agent", "all-fail"),  
    ("sanitize-git-repo", ('sanitize', 'github repository', 'sensitive values', 'replace', 'API keys', 'placeholder'), "terminus-2", "all-fail"),  
    ("vulnerable-secret", ('extract', 'secret key',), "pi", "multi-pass"),  
]


PLACEHOLDER = "TO_BE_FILLED"


def _keywords(keys):
    """Accept a tuple or a bare string. Return None (rule ignored) if any keyword
    is empty or still the placeholder, so an unfinished line never matches."""
    if isinstance(keys, str):
        keys = (keys,)
    keys = tuple(k.strip() for k in keys)
    if not keys or any(not k or k == PLACEHOLDER for k in keys):
        return None
    return keys


def _pattern(keyword):
    # whole word, case-insensitive; a space in a phrase also matches a line break
    body = re.escape(keyword).replace(r"\ ", r"\s+")
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])", re.I)


_COMPILED = [
    (tid, [_pattern(k) for k in ks], harness)
    for tid, keys, harness, _case in RULES
    if (ks := _keywords(keys))
]


def route(instruction: str) -> tuple[str, str]:
    """Return (harness, task_id). Both are "NA" when no rule matches."""
    for tid, pats, harness in _COMPILED:
        if all(p.search(instruction) for p in pats):
            return harness, tid
    return "NA", "NA"


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or sys.stdin.read()
    harness, task_id = route(text)
    print(f"{harness}\t{task_id}")
