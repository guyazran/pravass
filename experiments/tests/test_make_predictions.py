from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import make_predictions as mp


class ModelToPathSegmentTests(unittest.TestCase):
    def test_openai_default_model(self) -> None:
        self.assertEqual(
            mp.model_to_path_segment("OpenAI/gpt-4.1-2025-04-14"),
            "OpenAI__gpt-4.1-2025-04-14",
        )

    def test_gemma_model(self) -> None:
        self.assertEqual(
            mp.model_to_path_segment("google/gemma-3-4b-it"),
            "google__gemma-3-4b-it",
        )

    def test_string_without_slash_is_unchanged(self) -> None:
        self.assertEqual(mp.model_to_path_segment("local-model"), "local-model")


class GetSplitOutputRootTests(unittest.TestCase):
    def test_test_split_with_openai_model(self) -> None:
        self.assertEqual(
            mp.get_split_output_root(
                "test", mp.OPENAI_MODEL, output_root=Path("preds")
            ),
            Path("preds/test/OpenAI__gpt-4.1-2025-04-14"),
        )

    def test_validation_split_with_gemma_model(self) -> None:
        self.assertEqual(
            mp.get_split_output_root(
                "validation", "google/gemma-3-4b-it", output_root=Path("preds")
            ),
            Path("preds/validation/google__gemma-3-4b-it"),
        )


class SplitFileLayoutTests(unittest.TestCase):
    def test_split_root(self) -> None:
        self.assertEqual(mp.get_split_root("test"), mp.SPLIT_DATA_DIR / "test")
        self.assertEqual(
            mp.get_split_root("validation"), mp.SPLIT_DATA_DIR / "validation"
        )

    def test_split_samples_file(self) -> None:
        self.assertEqual(
            mp.get_split_samples_file("test"),
            mp.SPLIT_DATA_DIR / "test_samples.txt",
        )
        self.assertEqual(
            mp.get_split_samples_file("validation"),
            mp.SPLIT_DATA_DIR / "validation_samples.txt",
        )


class LoadSplitSamplesTests(unittest.TestCase):
    @contextmanager
    def _temp_split_root(self, split: str, sample_lines: list[str]):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / split).mkdir()
            (tmp_path / f"{split}_samples.txt").write_text(
                "\n".join(sample_lines) + "\n"
            )
            with patch.object(mp, "SPLIT_DATA_DIR", tmp_path):
                yield tmp_path

    def test_groups_by_domain_and_skips_blank_lines(self) -> None:
        lines = [
            "alf/h01/sample_000001",
            "alf/h01/sample_000002",
            "gw/h01/sample_000001",
            "",
        ]
        with self._temp_split_root("test", lines) as tmp:
            grouped = mp.load_split_samples("test")
        self.assertEqual(set(grouped), {"alf", "gw"})
        self.assertEqual(len(grouped["alf"]), 2)
        self.assertEqual(len(grouped["gw"]), 1)
        self.assertEqual(
            grouped["alf"][0], tmp / "test" / "alf" / "h01" / "sample_000001"
        )

    def test_validation_split_uses_validation_samples_file(self) -> None:
        lines = ["prb/cfg_a/sample_001", "prb/cfg_a/sample_002"]
        with self._temp_split_root("validation", lines) as tmp:
            grouped = mp.load_split_samples("validation")
        self.assertEqual(set(grouped), {"prb"})
        self.assertEqual(
            grouped["prb"][0],
            tmp / "validation" / "prb" / "cfg_a" / "sample_001",
        )


class MainCLITests(unittest.TestCase):
    """End-to-end argparse plumbing: --splits and --model reach predict_domain."""

    def _run_main(self, argv: list[str]) -> list[dict]:
        captured: list[dict] = []

        def fake_predict_domain(domain, sample_dirs, *, split, model, **kwargs):
            captured.append(
                {
                    "domain": domain,
                    "split": split,
                    "model": model,
                    "n_samples": len(sample_dirs),
                }
            )

        def fake_load_split_samples(split: str):
            return {
                "alf": [Path(f"split_data/{split}/alf/cfg/sample_000001")],
                "gw": [Path(f"split_data/{split}/gw/cfg/sample_000001")],
            }

        with patch.object(mp, "predict_domain", fake_predict_domain), patch.object(
            mp, "load_split_samples", fake_load_split_samples
        ), patch.object(mp, "build_vlm", lambda model, use_flash=False: object()), patch.object(
            sys, "argv", ["make_predictions.py", *argv]
        ):
            mp.main()
        return captured

    def test_defaults_to_all_splits_and_default_model(self) -> None:
        captured = self._run_main([])
        self.assertEqual(
            {c["split"] for c in captured}, {"test", "validation"}
        )
        for c in captured:
            self.assertEqual(c["model"], mp.GEMMA4_2B_MODEL)

    def test_single_split(self) -> None:
        captured = self._run_main(["--splits", "validation"])
        self.assertEqual({c["split"] for c in captured}, {"validation"})

    def test_two_splits(self) -> None:
        captured = self._run_main(["--splits", "test", "validation"])
        splits_seen = [c["split"] for c in captured]
        self.assertIn("test", splits_seen)
        self.assertIn("validation", splits_seen)

    def test_invalid_split_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self._run_main(["--splits", "train"])

    def test_gemma_model_is_threaded_through(self) -> None:
        captured = self._run_main(["--model", "google/gemma-4-E4B-it"])
        self.assertTrue(captured)
        for c in captured:
            self.assertEqual(c["model"], "google/gemma-4-E4B-it")

    def test_domain_filter_applies_per_split(self) -> None:
        captured = self._run_main(
            ["--splits", "test", "validation", "--domains", "alf"]
        )
        self.assertEqual({c["domain"] for c in captured}, {"alf"})
        self.assertEqual({c["split"] for c in captured}, {"test", "validation"})


if __name__ == "__main__":
    unittest.main()
