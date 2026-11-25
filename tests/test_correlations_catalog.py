"""Validación local del catálogo normalizado MAG–SWEA, sin consultas BigQuery.

Ejecutar:
python3 -m unittest discover -s tests -p 'test_correlations_catalog.py' -v
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/correlations/04a_boundary_energy_correlation_maps.py"
spec = importlib.util.spec_from_file_location("boundary_correlations", str(SCRIPT))
correlations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(correlations)


def args():
    return SimpleNamespace(case_col=None, event_type_col=None, event_time_col=None,
                           bs_col=None, mpb_col=None, start=None, end=None,
                           max_events=None)


class TestNormalizedCatalog(unittest.TestCase):
    def test_two_groups_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "crossings_catalog_normalized_4groups.csv"
            pd.DataFrame({
                "case_id": ["G1_row0002", "G2_row0003"],
                "group": ["G1", "G2"],
                "source_file": ["MAVEN_MPB_Data_Group1.csv",
                                "MAVEN_MPB_Data_Group2.csv"],
                "source_row": [2, 3],
                "bow_shock_datetime_utc": ["2016-03-16T17:30:00+00:00",
                                            "2016-03-17T17:30:00+00:00"],
                "mpb_datetime_utc": ["2016-03-16T18:00:00+00:00",
                                     "2016-03-17T18:00:00+00:00"],
                "theta_bn_deg": [45.0, 70.0],
                "beta_proton": [0.5, 0.7],
                "pdyn_proton": [1.1, 2.3],
            }).to_csv(csv, index=False)
            events, catalog, issues, source = correlations.load_events([csv], args())
            self.assertEqual(len(events), 4)
            self.assertEqual(events.group.value_counts().to_dict(), {"G1": 2, "G2": 2})
            self.assertEqual(set(events.source_file), {
                "MAVEN_MPB_Data_Group1.csv", "MAVEN_MPB_Data_Group2.csv"})
            self.assertEqual(set(events.source_row), {2, 3})
            self.assertEqual(set(events.theta_bn_deg), {45.0, 70.0})
            self.assertEqual(set(events.event_type), {"BS", "MPB"})
            self.assertEqual(len(catalog), 0)
            self.assertEqual(len(issues), 0)

    def test_repository_catalog_has_569_manual_cases(self):
        catalog = REPO / "data/classified_crossings/crossings_catalog_normalized_4groups.csv"
        if not catalog.exists():
            self.skipTest("Catálogo normalizado no disponible")
        events, _, _, _ = correlations.load_events([catalog], args())
        self.assertEqual(len(events), 1138)
        self.assertEqual(set(events.group), {"G1", "G2", "G3", "G4"})
        self.assertEqual(events.event_type.value_counts().to_dict(), {"BS": 569, "MPB": 569})


if __name__ == "__main__":
    unittest.main()
