from __future__ import annotations

import unittest

import pandas as pd


class NotebookDataExp1Tests(unittest.TestCase):
    @staticmethod
    def _plot_summary() -> pd.DataFrame:
        rows = []
        for model, domains in (("m1", ("prb",)), ("m2", ("prb", "gw"))):
            for domain in domains:
                for theta, mass, coverage, size in (
                    (0.5, 0.48, 0.44, 2.0),
                    (0.9, 0.86, 0.82, 5.0),
                ):
                    for method in ("mbs_prevas", "random_theta_confident"):
                        rows.append(
                            {
                                "Model": model,
                                "Domain": domain,
                                "Calibration": "global",
                                "Method": method,
                                "Method Label": method,
                                "Theta": theta,
                                "Completion": 1.0,
                                "N": 10,
                                "Avg Compact Size": size,
                                "IQM Compact Size": size,
                                "Avg Mass": mass,
                                "Coverage": coverage,
                            }
                        )
                    rows.append(
                        {
                            "Model": model,
                            "Domain": domain,
                            "Calibration": "global",
                            "Method": "map",
                            "Method Label": "MAP",
                            "Theta": theta,
                            "Completion": 1.0,
                            "N": 10,
                            "Avg Compact Size": 1.0,
                            "IQM Compact Size": 1.0,
                            "Avg Mass": 0.3,
                            "Coverage": 0.25,
                        }
                    )
        return pd.DataFrame(rows)

    @staticmethod
    def _figure_legend_labels(fig) -> list[str]:
        return [
            text.get_text()
            for legend in fig.legends
            for text in legend.get_texts()
        ]

    def test_model_comparison_legend_unions_series_across_domain_panels(self) -> None:
        import matplotlib.pyplot as plt

        from mbs_experiments import notebook_data as nd

        fig = nd.exp1_model_comparison_plot(
            self._plot_summary(),
            method="mbs_prevas",
            calibration="global",
            domains=("prb", "gw"),
        )

        try:
            labels = self._figure_legend_labels(fig)
            self.assertIn("m1", labels)
            self.assertIn("m2", labels)
            self.assertFalse(any("operating point" in label for label in labels))
            self.assertNotIn("(n=", " ".join(labels))
        finally:
            plt.close(fig)

    def test_mass_coverage_legends_match_plotted_models_and_methods(self) -> None:
        import matplotlib.pyplot as plt

        from mbs_experiments import notebook_data as nd

        summary = self._plot_summary()
        by_domain = nd.exp1_mass_coverage_by_domain_plot(
            summary,
            method="mbs_prevas",
            calibration="global",
            models=None,
            domains=("prb", "gw"),
        )
        by_model = nd.exp1_mass_coverage_by_model_plot(
            summary,
            methods=("map", "random_theta_confident", "mbs_prevas"),
            calibration="global",
            models=None,
            domains=("prb",),
        )

        try:
            domain_labels = self._figure_legend_labels(by_domain)
            self.assertIn("m1", domain_labels)
            self.assertIn("m2", domain_labels)
            self.assertIn("Ideal calibration", domain_labels)
            self.assertFalse(any("operating point" in label for label in domain_labels))

            model_labels = self._figure_legend_labels(by_model)
            self.assertIn(nd.method_label("map"), model_labels)
            self.assertIn(nd.method_label("random_theta_confident"), model_labels)
            self.assertIn(nd.method_label("mbs_prevas"), model_labels)
            self.assertIn("Ideal calibration", model_labels)
            self.assertFalse(any("operating point" in label for label in model_labels))
        finally:
            plt.close(by_domain)
            plt.close(by_model)

    def test_coverage_size_by_model_plot_overlays_all_requested_methods(self) -> None:
        import matplotlib.pyplot as plt

        from mbs_experiments import notebook_data as nd

        fig = nd.exp1_coverage_size_by_model_plot(
            self._plot_summary(),
            methods=("map", "random_theta_confident", "mbs_prevas"),
            calibration="global",
            models=None,
            domains=("prb",),
            size_field="compact",
            size_stat="iqm",
        )

        try:
            labels = self._figure_legend_labels(fig)
            self.assertIn(nd.method_label("map"), labels)
            self.assertIn(nd.method_label("random_theta_confident"), labels)
            self.assertIn(nd.method_label("mbs_prevas"), labels)
            self.assertFalse(any("operating point" in label for label in labels))
        finally:
            plt.close(fig)

    def test_current_method_labels_have_non_fallback_styles(self) -> None:
        from mbs_experiments import notebook_data as nd

        labels = set(nd.EXP1_METHOD_LABELS.values()) | set(nd.EXP3_METHOD_LABELS.values())
        missing = sorted(label for label in labels if nd._color(label) == "0.4")
        self.assertEqual(missing, [])

    def test_theta_size_plot_style_survives_runtime_label_edit(self) -> None:
        import matplotlib.pyplot as plt

        from mbs_experiments import notebook_data as nd

        method = "mbs_prevas"
        original = nd.EXP1_METHOD_LABELS[method]
        renamed = "MBSθ (Renamed)"
        summary = pd.DataFrame(
            [
                {
                    "Model": "m",
                    "Domain": "gw",
                    "Calibration": "global",
                    "Method": method,
                    "Method Label": original,
                    "Theta": 0.9,
                    "Completion": 1.0,
                    "IQM Compact Size": 10.0,
                }
            ]
        )

        try:
            nd.EXP1_METHOD_LABELS[method] = renamed
            fig = nd.exp1_theta_size_plot(
                summary,
                model="m",
                calibration="global",
                methods=[method],
                domains=("gw",),
                size_stat="iqm",
            )
            expected_color = nd._color_for_method(method)
            line = fig.axes[0].lines[0]
            self.assertEqual(line.get_label(), renamed)
            self.assertEqual(line.get_color(), expected_color)
            self.assertNotEqual(line.get_color(), nd._color(renamed))
        finally:
            nd.EXP1_METHOD_LABELS[method] = original
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
