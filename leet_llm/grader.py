"""Test-side helper: load the task-under-test.

Every task's ``tests/test_*.py`` starts with::

    from leet_llm.grader import load
    m = load(__file__)
    group_last_axis = m.group_last_axis

``load`` infers the task folder from the test file's path (its grandparent) and
returns the learner's stub by default, or the reference when
``LEET_LLM_TARGET=solution``.
"""

from __future__ import annotations

import pathlib
from types import ModuleType

from ._loader import load_task


def load(test_file: str) -> ModuleType:
    """Load the module for the task whose ``tests/`` directory holds ``test_file``."""
    task_dir = pathlib.Path(test_file).resolve().parent.parent
    return load_task(task_dir.name)
