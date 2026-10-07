"""Extract every Terminal-Bench task into one CSV: one row per task.

Usage:
    python extract_tasks.py <path-to-dataset-folder> [output.csv]

<path-to-dataset-folder> is the folder that CONTAINS the task folders,
i.e. the one where each subfolder has its own instruction.md and task.toml.

Needs Python 3.11+ (uses the built-in tomllib). No other dependencies.
"""

import csv
import sys
import tomllib
from pathlib import Path

COLUMNS = [
    "task_id",
    "description",  # task.toml [task].description (written by the benchmark author)
    "category",  # task.toml [metadata].category
    "difficulty",  # task.toml [metadata].difficulty
    "tags",  # task.toml [metadata].tags, joined with ";"
    "keywords",  # task.toml [task].keywords, joined with ";"
    "agent_timeout_sec",  # task.toml [agent].timeout_sec
    "instruction",  # full text of instruction.md (the agent's prompt)
]


def read_task(task_dir: Path) -> dict:
    toml = tomllib.loads((task_dir / "task.toml").read_text())
    task = toml.get("task", {})
    meta = toml.get("metadata", {})
    agent = toml.get("agent", {})
    return {
        "task_id": task_dir.name,
        "description": task.get("description", ""),
        "category": meta.get("category", ""),
        "difficulty": meta.get("difficulty", ""),
        "tags": ";".join(meta.get("tags", [])),
        "keywords": ";".join(task.get("keywords", [])),
        "agent_timeout_sec": agent.get("timeout_sec", ""),
        "instruction": (task_dir / "instruction.md").read_text().strip(),
    }


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1]).expanduser()
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "tasks.csv")

    task_dirs = sorted(p.parent for p in root.glob("*/task.toml"))
    if not task_dirs:
        sys.exit(f"No task folders found in {root}. Point at the folder that "
                 f"contains the task folders, not at a single task.")

    rows, skipped = [], []
    for d in task_dirs:
        if not (d / "instruction.md").exists():
            skipped.append(d.name)
            continue
        rows.append(read_task(d))

    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} tasks -> {out}")
    if skipped:
        print(f"skipped (no instruction.md): {skipped}")


if __name__ == "__main__":
    main()
