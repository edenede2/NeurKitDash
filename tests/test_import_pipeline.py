import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from pages.data_manager import DataManager, _time_to_index
from pages.drive_import import find_recording, read_recording, subject_folders


class ImportPipelineTests(unittest.TestCase):
    def test_find_recording_prefers_data_export_and_reads_both_header_styles(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "sub_001" / "hrv"
            folder.mkdir(parents=True)
            (folder / "a_hrv.txt").write_text("unrelated", encoding="utf-8")
            (folder / "a_markers.txt").write_text("Time\tEvent\n0\tStart\n", encoding="utf-8")
            (folder / "z_EVENTS.TXT").write_text("Time\tEvent\n0\tStart\n", encoding="utf-8")
            signal = folder / "z_DATA.TXT"
            for preamble in ("", "Sample Rate: 500\n", "Mindware export\n"):
                signal.write_text(preamble + "Time (s)\tBio\n0\t1\n0.002\t2\n", encoding="utf-8")
                folders = subject_folders(root)
                kind, signal_path, event_path = find_recording(folders["sub_001"])
                self.assertEqual(kind, "HRV")
                self.assertEqual(Path(signal_path), signal)
                self.assertEqual(Path(event_path).name, "z_EVENTS.TXT")
                frame, _ = read_recording(signal_path)
                self.assertEqual(frame["Bio"].tolist(), [1, 2])

    def test_batch_retries_a_partial_write_and_preserves_legacy_data(self):
        with tempfile.TemporaryDirectory() as root:
            manager = DataManager("demo", root)
            pl.DataFrame({"subject_id": ["old"], "time": [0.0], "signal": [1.0]}).write_parquet(
                manager.signal_parquet_path
            )
            pd.DataFrame([{"subject_id": "old", "attributes_json": json.dumps({"rpeaks": [0]})}]).to_csv(
                manager.attributes_csv_path, index=False
            )
            samples = {"time": np.arange(3), "signal": np.ones(3)}
            attrs = {"quality": np.array([0.3, 0.8, 1.0]), "rpeaks": [1]}
            events = [{"name": "rest", "Start_Time": 0.0, "End_Time": 2.0}]
            pending = {"new": manager.stage_subject("new", samples, events, attrs),
                       "signal-only": manager.stage_subject("signal-only", samples, [], {})}
            from pages import data_manager

            real_replace = data_manager.replace_file

            def fail_signal(source, target):
                if str(target).endswith("new.parquet"):
                    raise PermissionError("temporary sharing violation")
                return real_replace(source, target)

            with patch.object(data_manager, "replace_file", side_effect=fail_signal):
                with self.assertLogs("pages.data_manager", "ERROR"):
                    self.assertEqual(manager.commit_subjects(pending), {"signal-only"})
            self.assertEqual(manager.get_all_subjects(), ["old", "signal-only"])
            self.assertTrue(manager.save_subject_data("new", samples, events, attrs))
            self.assertEqual(manager.get_all_subjects(), ["new", "old", "signal-only"])
            self.assertEqual(manager.load_subject_attributes("old"), {"rpeaks": [0]})
            self.assertEqual(manager.load_subject_attributes("new"), {"rpeaks": [1], "quality": [0.3, 0.8, 1.0]})
            self.assertEqual(manager.load_subject_attributes("new", include_quality=False), {"rpeaks": [1]})
            self.assertEqual(len(manager.load_subject_events_data("new")), 1)

    def test_binary_time_lookup_matches_linear_lookup(self):
        times = np.array([0.0, 0.5, 1.25, 2.0])
        for target in [-2.0, 0.0, 0.25, 0.9, 1.25, 2.4, 4.0]:
            self.assertEqual(_time_to_index(target, times, True), _time_to_index(target, times))


if __name__ == "__main__":
    unittest.main()
