from __future__ import annotations

import unittest

import pandas as pd


class NotebookDataExp3Tests(unittest.TestCase):
    def test_exp3_success_table_uses_default_operating_thetas_when_theta_is_none(self) -> None:
        from mbs_experiments import notebook_data as nd

        summary = pd.DataFrame(
            [
                {
                    "Model": "m",
                    "Domain": "prb",
                    "Calibration": "global",
                    "Method": "mbs_prevas",
                    "Method Label": "MBSθ (Ours)",
                    "Theta": 0.01,
                    "N": 2,
                    "Success Rate": 0.5,
                    "Success Rate CI Low": 0.5,
                    "Success Rate CI High": 0.5,
                    "Timeout Rate": 0.0,
                },
                {
                    "Model": "m",
                    "Domain": "prb",
                    "Calibration": "global",
                    "Method": "mbs_prevas",
                    "Method Label": "MBSθ (Ours)",
                    "Theta": 0.99,
                    "N": 2,
                    "Success Rate": 1.0,
                    "Success Rate CI Low": 1.0,
                    "Success Rate CI High": 1.0,
                    "Timeout Rate": 0.0,
                },
                {
                    "Model": "m",
                    "Domain": "prb",
                    "Calibration": "global",
                    "Method": "mbs_prevas_prefix_search_constraints_relaxed",
                    "Method Label": "MBSθ prefix-search (relaxed)",
                    "Theta": 0.99,
                    "N": 2,
                    "Success Rate": 0.25,
                    "Success Rate CI Low": 0.25,
                    "Success Rate CI High": 0.25,
                    "Timeout Rate": 0.0,
                },
            ]
        )

        table = nd.exp3_success_table(
            summary,
            model="m",
            calibration="global",
            methods=[
                "mbs_prevas",
                "mbs_prevas_prefix_search_constraints_relaxed",
            ],
            theta=None,
            domains=("prb",),
            show_timeout=False,
        )

        mbs_label = nd.EXP3_METHOD_LABELS["mbs_prevas"]
        self.assertIn("50.0%", table.loc[mbs_label, "PRB"])
        self.assertIn("(n=2)", table.loc[mbs_label, "PRB"])
        prefix_label = nd.EXP3_METHOD_LABELS[
            "mbs_prevas_prefix_search_constraints_relaxed"
        ]
        self.assertIn("25.0%", table.loc[prefix_label, "PRB"])


if __name__ == "__main__":
    unittest.main()
