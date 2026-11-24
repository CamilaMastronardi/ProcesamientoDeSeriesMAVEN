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
