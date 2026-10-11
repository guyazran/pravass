from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace

from mbs_experiments.data import SampleId

MODEL = "OpenAI/gpt-4.1-2025-04-14"

# Fields every paper-era Experiment-2 ``prevas`` row carries; the renderers in
# ``results_display`` (cactus / sfc / performance table) read a subset of them.
PAPER_PREVAS_FIELDS = {
    "calibration",
    "compact_set_size",
    "configuration",
    "domain",
    "exact_match",
    "exhaustive_runtime_seconds",
    "feature_count",
    "full_normalizer",
    "jaccard",
    "method",
    "normalizer",
    "omitted_feature_count",
    "peak_rss_mb",
    "runtime_seconds",
    "sample",
    "search_feature_count",
    "search_selected_group_count",
    "set_size",
    "split_relative_path",
    "states_popped",
    "states_scored",
    "status",
    "theta",
    "timeout",
}


def _sid() -> SampleId:
    return SampleId("prb", "h01_n03", "sample_000001", "prb/h01_n03/sample_000001")


def _fake_bundle() -> SimpleNamespace:
    # Three visible features with one at-most-one group over the first two.
    return SimpleNamespace(
        feature_names=("a", "b", "c"),
        probabilities=(0.9, 0.8, 0.3),
        none_is_max=(False, False, False),
        selected_groups=((0, 1),),
    )


class Experiment2RunnerTests(unittest.TestCase):
    def test_defaults_match_the_paper_settings(self) -> None:
        from scripts import run_experiment_2_prevas_scaling as runner

        args = runner.parse_args([])
        self.assertEqual(args.model, MODEL)
        self.assertEqual(args.thetas, (0.5, 0.7, 0.8, 0.9, 0.95, 0.99))
        self.assertEqual(args.method_timeout, 10.0)
        self.assertEqual(tuple(args.domains), ("prb", "gw", "alf"))
        self.assertEqual(runner.TEST_SPLIT, "test")

    def test_thetas_are_deduplicated_and_validated(self) -> None:
        from scripts import run_experiment_2_prevas_scaling as runner

        args = runner.parse_args(["--thetas", "0.9", "0.5", "0.9"])
        self.assertEqual(args.thetas, (0.9, 0.5))
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            runner.parse_args(["--thetas", "1.5"])

    def test_outputs_are_sharded_by_model(self) -> None:
        from scripts import run_experiment_2_prevas_scaling as runner

        output_dir, results_path, agg_path = runner.experiment_paths(MODEL)
        base = Path("experiments_data/experiment_2_prevas_scaling/OpenAI__gpt-4.1-2025-04-14")
        self.assertEqual(output_dir, base)
        self.assertEqual(results_path, base / "sample_results.jsonl")
        self.assertEqual(agg_path, base / "aggregate_metrics.json")
        self.assertEqual(
            runner.RESULT_KEY_FIELDS[:4], ("domain", "configuration", "sample", "calibration")
        )
        self.assertEqual(runner.RESULT_KEY_FIELDS[-1], "model")

    def test_calibration_task_uses_the_multi_model_bundle_api(self) -> None:
        from scripts import run_experiment_2_prevas_scaling as runner

        calls: list[tuple] = []

        def fake_load_calibration_bundle(sid, calibration, *, model, split):
            calls.append((sid, calibration, model, split))
            return _fake_bundle()

        task = runner.Experiment2Task(
            sid=_sid(),
            calibration="global",
            config=runner.Experiment2Config(method_timeout=5.0, thetas=(0.5, 0.99)),
            model=MODEL,
            existing_records={},
        )
        original = runner.load_calibration_bundle
        runner.load_calibration_bundle = fake_load_calibration_bundle
        try:
            records = runner._run_experiment_2_calibration_task(task)
        finally:
            runner.load_calibration_bundle = original

        self.assertEqual(calls, [(task.sid, "global", MODEL, "test")])
        self.assertEqual(len(records), 6)
        self.assertEqual({r["status"] for r in records}, {"ok"})
        self.assertEqual(
            {r["method"] for r in records},
            {"exhaustive", "prevas", "random_theta_confident"},
        )
        self.assertEqual({r["theta"] for r in records}, {0.5, 0.99})
        self.assertTrue(all(r["model"] == MODEL for r in records))
        keys = {runner.result_key(r, runner.RESULT_KEY_FIELDS) for r in records}
        self.assertEqual(len(keys), 6)

        prevas = [r for r in records if r["method"] == "prevas"]
        self.assertTrue(PAPER_PREVAS_FIELDS <= set(prevas[0]))
        self.assertTrue(all(r["exact_match"] for r in prevas))
        self.assertTrue(all(r["jaccard"] == 1.0 for r in prevas))
        self.assertTrue(all(r["search_feature_count"] == 3 for r in records))
        self.assertTrue(all(r["search_selected_group_count"] == 1 for r in records))

    def test_bundle_failure_becomes_a_single_error_record(self) -> None:
        from scripts import run_experiment_2_prevas_scaling as runner

        def failing_bundle(sid, calibration, *, model, split):
            raise FileNotFoundError("missing prediction.json")

        task = runner.Experiment2Task(
            sid=_sid(),
            calibration="global",
            config=runner.Experiment2Config(method_timeout=5.0),
            model=MODEL,
            existing_records={},
        )
        original = runner.load_calibration_bundle
        runner.load_calibration_bundle = failing_bundle
        try:
            records = runner._run_experiment_2_calibration_task(task)
        finally:
            runner.load_calibration_bundle = original

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "error")
        self.assertEqual(records[0]["method"], "bundle")
        self.assertEqual(records[0]["model"], MODEL)
        self.assertEqual(records[0]["error_type"], "FileNotFoundError")

    def test_existing_rows_are_not_recomputed(self) -> None:
        from scripts import run_experiment_2_prevas_scaling as runner

        sid = _sid()
        existing = {}
        for method in ("exhaustive", "prevas", "random_theta_confident"):
            key = runner.experiment_key(sid, "global", method, 0.5, model=MODEL)
            existing[key] = {"status": "ok", "method": method, "theta": 0.5}
        task = runner.Experiment2Task(
            sid=sid,
            calibration="global",
            config=runner.Experiment2Config(method_timeout=5.0, thetas=(0.5, 0.99)),
            model=MODEL,
            existing_records=existing,
        )
        original = runner.load_calibration_bundle
        runner.load_calibration_bundle = lambda sid, calibration, *, model, split: _fake_bundle()
        try:
            records = runner._run_experiment_2_calibration_task(task)
        finally:
            runner.load_calibration_bundle = original

        self.assertEqual(len(records), 3)
        self.assertEqual({r["theta"] for r in records}, {0.99})


if __name__ == "__main__":
    unittest.main()
