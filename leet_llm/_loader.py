"""Path-based loader for task modules.

A task lives in a folder like ``001_numpy_array_basics/`` containing the
learner's stub (``array_basics.py``) and the reference (``solution.py``). The
``LEET_LLM_TARGET`` env var selects which one to load:

- unset / ``"stub"`` (default): the learner's stub — the single ``.py`` in the
  task root that is neither ``solution`` nor ``conftest``.
- ``"solution"``: the reference ``solution.py``.

Modules are loaded by file path under a unique name (``<folder>_<stem>``) so a
full-course run never collides two different ``solution.py`` files in
``sys.modules``. Loaded modules are cached per (folder, target).
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
from types import ModuleType

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

_cache: dict[tuple[str, str], ModuleType] = {}


def _target() -> str:
    return os.environ.get("LEET_LLM_TARGET", "stub")


def _stub_path(task_dir: pathlib.Path) -> pathlib.Path:
    candidates = [
        p
        for p in sorted(task_dir.glob("*.py"))
        if p.stem not in ("solution", "conftest", "convert", "utils")
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one stub .py in {task_dir.name}/, found: "
            f"{[p.name for p in candidates]}"
        )
    return candidates[0]


def load_task(folder: str) -> ModuleType:
    """Load the module for task folder ``folder`` (e.g. '001_numpy_array_basics')."""
    target = _target()
    key = (folder, target)
    if key in _cache:
        return _cache[key]

    task_dir = REPO_ROOT / folder
    if not task_dir.is_dir():
        raise FileNotFoundError(f"no task folder {folder!r} in {REPO_ROOT}")

    if target == "solution":
        path = task_dir / "solution.py"
        if not path.exists():
            raise FileNotFoundError(f"no solution.py in {folder}/")
    else:
        path = _stub_path(task_dir)

    module_name = f"{folder}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader, f"could not build import spec for {path}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module resolves its own name during execution
    # (dataclasses with bare string annotations look themselves up in sys.modules).
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _cache[key] = module
    return module
