import unittest
import gzip
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.analysis_service import (
    analyze_traces,
    calculate_lengths,
    detect_alltrace_boundary,
    parse_traces,
    recalculate_boundary_outputs,
)


class AnalysisServiceTests(unittest.TestCase):
    def test_parse_csv_pairs_without_row_misalignment(self):
        frame = pd.DataFrame([[0.0, -0.5, 0.0, -0.4], [0.1, -3.0, 0.1, -3.1]])
        traces, ids = parse_traces(frame.to_csv(index=False, header=False).encode(), "sample.csv")
        self.assertEqual(ids, [1, 2])
        self.assertEqual(len(traces), 2)
        np.testing.assert_allclose(traces[0][1], [-0.5, -3.0])

    def test_parse_gzip_csv_and_zip_csv(self):
        frame = pd.DataFrame([[0.0, -0.5], [0.1, -3.0]])
        csv_data = frame.to_csv(index=False, header=False).encode()
        gzip_data = gzip.compress(csv_data)
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w") as archive:
            archive.writestr("trace.csv", csv_data)

        gzip_traces, _ = parse_traces(gzip_data, "trace.csv.gz")
        zip_traces, _ = parse_traces(zip_buffer.getvalue(), "trace.zip")
        self.assertEqual(len(gzip_traces), 1)
        self.assertEqual(len(zip_traces), 1)

    def test_legacy_xls_is_not_supported(self):
        with self.assertRaisesRegex(ValueError, "仅支持 CSV、XLSX"):
            parse_traces(b"not-an-xls-workbook", "trace.xls")

    def test_alltrace_valley_is_between_background_and_platform(self):
        traces = []
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = np.linspace(-0.2, 1.5, 240)
            y = np.concatenate([
                rng.normal(-3.45, 0.04, 150),
                rng.normal(-6.0, 0.05, 90),
            ])
            traces.append((x, y))
        result = detect_alltrace_boundary(traces)
        self.assertLess(result["boundary_log_g_g0"], result["platform_peak_log_g_g0"])
        self.assertGreater(result["boundary_log_g_g0"], result["background_peak_log_g_g0"])

    def test_rightmost_boundary_window_and_slope_filter(self):
        x = np.arange(0.0, 1.0, 0.1)
        y = np.array([0.0, -1.2, -3.0, -3.1, -4.0, -3.4, -4.0, -4.8, -6.0, -6.2])
        accepted, rejected = calculate_lengths(
            [(x, y)], np.array([0]), boundary_log_g_g0=-4.7, trim_points_each_side=0
        )
        self.assertEqual(len(accepted[0]), 1)
        self.assertAlmostEqual(accepted[0][0]["x_end"], 0.6)
        self.assertEqual(rejected[0], {})

    def test_complete_analysis_returns_three_ordered_clusters(self):
        traces = []
        for cluster_peak in (-2.7, -3.4, -4.2):
            for replicate in range(4):
                x = np.linspace(-0.1, 1.0, 80)
                y = np.full_like(x, cluster_peak)
                y[:8] = 0.0
                y[-8:] = -6.0
                y += (replicate - 1.5) * 0.002
                traces.append((x, y))
        result = analyze_traces(traces)
        self.assertEqual(len(result.cluster_stats), 3)
        self.assertEqual(sum(item["count"] for item in result.cluster_stats), 12)
        self.assertGreater(result.cluster_stats[0]["peak_mean"], result.cluster_stats[1]["peak_mean"])
        self.assertGreater(result.cluster_stats[1]["peak_mean"], result.cluster_stats[2]["peak_mean"])

    def test_analysis_supports_dynamic_k_and_roi(self):
        traces = []
        for cluster_peak in (-2.5, -3.0, -3.5, -4.0):
            for replicate in range(4):
                x = np.linspace(-0.1, 1.2, 90)
                y = np.full_like(x, cluster_peak + replicate * 0.002)
                y[:8] = 0.0
                y[-10:] = -6.0
                traces.append((x, y))
        result = analyze_traces(
            traces,
            n_clusters=4,
            roi_x_range=(0.0, 1.0),
            roi_y_range=(-4.5, -2.0),
        )
        self.assertEqual(result.n_clusters, 4)
        self.assertEqual(result.roi_x_range, (0.0, 1.0))
        self.assertEqual(len(result.cluster_stats), 4)
        self.assertEqual(set(result.lengths), {0, 1, 2, 3})

        updated = recalculate_boundary_outputs(result, -4.6)
        self.assertEqual(updated.n_clusters, 4)
        self.assertAlmostEqual(updated.boundary["boundary_log_g_g0"], -4.6)
        self.assertEqual(len(updated.cluster_stats), 4)


if __name__ == "__main__":
    unittest.main()
