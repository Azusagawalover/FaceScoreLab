"""Local synthetic HTTP tests; no external dataset bytes or training are involved."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import random
import tempfile
import threading
import unittest

from parallel_public_download import transfer_ranges
from prepare_celeba_subset import atomic_json


class RangeSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = random.Random(42).randbytes(16 * 1024 * 1024 + 10000)
        cls.owner = Path(__file__).resolve().parent / "download_test_temp"
        cls.owner.mkdir(exist_ok=True)

    def exercise(self, mode="ok", cap=64 * 1024 * 1024):
        payload = self.payload

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if mode == "ignored_range":
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("ETag", '"fixture-etag"')
                    self.end_headers()
                    return
                start, end = map(int, self.headers["Range"][6:].split("-"))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("ETag", '"changed-etag"' if mode == "changed_etag" else '"fixture-etag"')
                self.end_headers()
                if mode == "changed_etag":
                    return
                try:
                    self.wfile.write(payload[start:end + 1])
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        temporary = tempfile.TemporaryDirectory(dir=self.owner)
        root = Path(temporary.name).resolve()
        self.assertTrue(root.is_relative_to(self.owner.resolve()))
        try:
            part = root / "synthetic.part"
            part.write_bytes(payload[:4096])
            state = {"etag": '"fixture-etag"'}
            budget = {"accounted_bytes": 4096}
            call = lambda: transfer_ranges(f"http://127.0.0.1:{server.server_port}/synthetic", part, root, state,
                                          root / "state.json", len(payload), budget, cap, atomic_json, workers=2)
            if mode == "ok":
                call()
                self.assertEqual(part.read_bytes(), payload)
                self.assertLessEqual(budget["accounted_bytes"], cap)
                self.assertEqual(list((root / "public_range_chunks").glob("*.chunk")), [])
            elif mode in ("ignored_range", "changed_etag"):
                with self.assertRaisesRegex(ValueError, "Range not honored|ETag/length changed"):
                    call()
                self.assertEqual(part.read_bytes(), payload[:4096])
            else:
                with self.assertRaisesRegex(ValueError, "budget prevents"):
                    call()
                self.assertLessEqual(budget["accounted_bytes"], cap)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            # Resolved target was checked within this task's test directory above.
            temporary.cleanup()

    def test_parallel_resume_preserves_ordered_bytes(self):
        self.exercise()

    def test_ignored_range_does_not_append(self):
        self.exercise("ignored_range")

    def test_changed_etag_does_not_append(self):
        self.exercise("changed_etag")

    def test_concurrent_workers_cannot_exceed_budget(self):
        self.exercise("budget", cap=9 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)
