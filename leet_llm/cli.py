"""The ``grade`` console script — a short wrapper around pytest.

    uv run grade            # grade the task in the current directory
    uv run grade 001        # grade task 001 from anywhere
    uv run grade 2          # grade every Level-2 task
    uv run grade all        # grade the whole course
    uv run grade -s 001     # grade the reference solution
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

from ._loader import REPO_ROOT

TASK_RE = re.compile(r"^\d{3}_")


def _current_task(start: pathlib.Path) -> pathlib.Path | None:
    for p in (start, *start.parents):
        if TASK_RE.match(p.name):
            return p
    return None


def _resolve_targets(task: str | None) -> list[str] | None:
    """Return pytest path args for the requested task selector, or None on error."""
    if task is None:
        current = _current_task(pathlib.Path.cwd())
        if current is None:
            print(
                "Not inside a task folder — pass a task id, e.g. `grade 001`.",
                file=sys.stderr,
            )
            return None
        return [str(current)]

    if task == "all":
        return [str(REPO_ROOT)]

    matches = sorted(
        p
        for p in REPO_ROOT.iterdir()
        if p.is_dir() and TASK_RE.match(p.name) and p.name.startswith(task)
    )
    if not matches:
        print(f"No task folders match {task!r}.", file=sys.stderr)
        return None
    return [str(p) for p in matches]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grade", description="Run leet-llm task tests."
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="task id/prefix (e.g. 001, or 2 for Level 2), or 'all'. "
        "Omit to grade the current directory's task.",
    )
    parser.add_argument(
        "-s",
        "--solution",
        action="store_true",
        help="grade the reference solution instead of your stub",
    )
    args, extra = parser.parse_known_args(argv)

    targets = _resolve_targets(args.task)
    if targets is None:
        return 2

    env = os.environ.copy()
    if args.solution:
        env["LEET_LLM_TARGET"] = "solution"

    cmd = [sys.executable, "-m", "pytest", *targets, *extra]
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
