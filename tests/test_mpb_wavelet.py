"""Pruebas sintéticas de CWT de MAG: sin BigQuery ni Parquet externos.

Ejecutar desde la raíz:
    python3 -m unittest discover -s tests -p "test_mpb_wavelet.py" -v
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" / "wavelets"
          / "05a_MPB_MAG_wavelet_10min.py")
spec = importlib.util.spec_from_file_location("mpb_wavelet_module", str(SCRIPT))
wavelets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wavelets)


class TestMpbWavelet(unittest.TestCase):
    def setUp(self):
        self.mpb = pd.Timestamp("2016-03-16T18:00:00Z")
        self.t = np.arange(-300., 301., 1.)
        self.periods = np.geomspace(4., 120., 48)
        # Onda conocida de período 20 s con tendencia lineal.
        self.B = 20. + np.sin(2 * np.pi * self.t / 20.) + self.t * 0.001
        self.data = pd.DataFrame({
            "datetime_utc": self.mpb + pd.to_timedelta(self.t, unit="s"),
            "b_total_nt": self.B, "is_interpolated": False
        })

    def test_known_period(self):
        trace, power, avg, count, stats = wavelets.cwt_event(
            self.data, self.mpb, "b_total_nt", self.t, self.periods)
        self.assertEqual(stats["status"], "ok")
        self.assertAlmostEqual(stats["fraction_window_covered"], 1.0)
        self.assertGreater(stats["n_valid_cwt_pixels"], 0)
        self.assertLess(abs(stats["dominant_period_s"] - 20.), 5.)
        self.assertTrue(np.all(np.isfinite(trace)))
        self.assertGreater(np.max(count), 0)

    def test_gap_not_crossed(self):
        # Treinta segundos de datos faltantes en torno a la MPB.
        data = self.data.loc[(self.t <= -20.) | (self.t >= 20.)].copy()
        trace, power, avg, count, stats = wavelets.cwt_event(
            data, self.mpb, "b_total_nt", self.t, self.periods)
        self.assertTrue(np.all(np.isnan(trace[(self.t > -20.) & (self.t < 20.)])))
        self.assertTrue(np.all(np.isnan(power[:, (self.t > -20.) & (self.t < 20.)])))
        self.assertEqual(stats["n_contiguous_segments"], 2)
        self.assertLess(stats["fraction_window_covered"], 1.)

    def test_before_after_ratio_excludes_crossing_instant(self):
        # Dos tiempos válidos por lado. La muestra central no debe entrar.
        t = np.array([-2., -1., 0., 1., 2.])
        power = np.array([
            [2., 2., 999., 8., 8.],
            [np.nan, 2., 999., 8., np.nan],
        ])
        before, after, n_before, n_after, delta = (
            wavelets.paired_power_by_period(power, t, min_side_points=2)
        )
        self.assertEqual(n_before.tolist(), [2, 1])
        self.assertEqual(n_after.tolist(), [2, 1])
        self.assertAlmostEqual(before[0], 2.)
        self.assertAlmostEqual(after[0], 8.)
        self.assertAlmostEqual(delta[0], np.log10(4.))
        self.assertTrue(np.isnan(delta[1]))

    def test_pair_with_no_overlap_with_coi(self):
        # Una fila sin ningún tiempo válido después de la frontera.
        t = np.array([-2., -1., 0., 1., 2.])
        p = np.array([[2., 2., np.nan, np.nan, np.nan]])
        before, after, n_before, n_after, delta = (
            wavelets.paired_power_by_period(p, t, min_side_points=2)
        )
        self.assertEqual(n_before[0], 2)
        self.assertEqual(n_after[0], 0)
        self.assertTrue(np.isnan(delta[0]))

    def test_catalog_preserves_manual_mpb(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "MAVEN_MPB_Data_Group1.csv"
            file.write_text(
                "yyyy-mm-dd,BS time,MPB time\n"
                "2016-03-16,17:55:00,18:00:00\n"
                "2016-03-16,17:55:00,18:00:00\n",
                encoding="utf-8")
            selected = wavelets.load_mpb_catalog(Path(directory))
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected.iloc[0]["mpb_utc"], self.mpb)
            self.assertEqual(selected.iloc[0]["case_id"], "G1_row0002")


if __name__ == "__main__":
    unittest.main()
