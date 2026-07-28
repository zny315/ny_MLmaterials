from __future__ import annotations

import importlib.util
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_pipeline():
    path = ROOT / "pipeline" / "eva_split_pipeline.py"
    spec = importlib.util.spec_from_file_location("eva_split_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ThreadSafetyTests(unittest.TestCase):
    def test_pending_part_rows_are_thread_local(self):
        module = load_pipeline()
        analyzer = object.__new__(module.EntityFormulationAnalyzer)
        analyzer._part_context = threading.local()
        barrier = threading.Barrier(3)
        observed = {}

        def worker(label):
            analyzer._set_pending_part_rows([{"worker": label}])
            barrier.wait()
            observed[label] = analyzer._get_pending_part_rows()

        threads = [threading.Thread(target=worker, args=(label,)) for label in ("A", "B")]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(observed, {"A": [{"worker": "A"}], "B": [{"worker": "B"}]})
        self.assertEqual(analyzer._get_pending_part_rows(), [])


if __name__ == "__main__":
    unittest.main()
