from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    "mbs_experiments/abstention.py",
    "scripts/run_experiment_1_static_quality.py",
    "scripts/run_experiment_2_prevas_scaling.py",
    "scripts/run_experiment_3_planning_utility.py",
)


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


class RuntimeNormalizerUsageTests(unittest.TestCase):
    def test_runtime_files_do_not_call_selected_groups_normalizer_directly(self) -> None:
        offenders: list[str] = []
        for rel in RUNTIME_FILES:
            path = PROJECT_ROOT / rel
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _call_name(node) == "selected_groups_normalizer":
                    offenders.append(f"{rel}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_runtime_files_do_not_read_precomputed_normalizer_data(self) -> None:
        offenders: list[str] = []
        forbidden = ("normalizer_data", "compute_normalizer")
        for rel in RUNTIME_FILES:
            path = PROJECT_ROOT / rel
            text = path.read_text()
            for token in forbidden:
                if token in text:
                    offenders.append(f"{rel}: contains {token!r}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
