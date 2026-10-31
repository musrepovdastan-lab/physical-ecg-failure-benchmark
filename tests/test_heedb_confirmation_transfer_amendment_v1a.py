import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from src import heedb_confirmation_transfer_amendment_v1a as amendment
from src import heedb_independent_confirmation_v1 as original


class ContainedPathTests(unittest.TestCase):
    def test_valid_nested_waveform_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "waveforms"
            with patch.object(original, "WAVES", root):
                for relative in ("ECG/a/b.hea", "ECG/a/b.dat", "one.dat"):
                    result = amendment.validated_path(relative)
                    self.assertEqual(result, root / Path(relative))

    def test_traversal_absolute_and_windows_path_attacks(self):
        attacks = (
            "../escape.dat", "ECG/../escape.dat", "ECG//x.dat", "ECG/./x.dat",
            "/tmp/x.dat", "\\root\\x.dat", "C:\\temp\\x.dat", "C:/temp/x.dat",
            "\\\\server\\share\\x.dat", "ECG\\x.dat", "ECG/a.txt", "ECG/NUL.dat",
            "ECG/a.dat ", "ECG/a.dat/", "ECG/a.dat\x00",
        )
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(original, "WAVES", Path(temporary) / "waveforms"):
                for relative in attacks:
                    with self.subTest(relative=repr(relative)):
                        with self.assertRaises(ValueError):
                            amendment.validated_path(relative)

    def test_concurrent_directory_creation_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "waveforms"
            with patch.object(original, "WAVES", root):
                def work(index):
                    relative = f"ECG/{index % 17}/{index % 31}/record_{index}.dat"
                    first = amendment.validated_path(relative)
                    first.parent.mkdir(parents=True, exist_ok=True)
                    second = amendment.validated_path(relative)
                    return first == second and root in second.parents

                with ThreadPoolExecutor(max_workers=24) as executor:
                    outcomes = list(executor.map(work, range(1500)))
                self.assertEqual(len(outcomes), 1500)
                self.assertTrue(all(outcomes))

    def test_existing_link_component_is_rejected_when_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "waveforms"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            link = root / "linked"
            try:
                os.symlink(outside, link, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable on this Windows account")
            with patch.object(original, "WAVES", root):
                with self.assertRaises(ValueError):
                    amendment.validated_path("linked/record.dat")


if __name__ == "__main__":
    unittest.main()
