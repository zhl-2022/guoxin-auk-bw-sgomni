"""Safety and transport tests for the bounded pilot; no model weights/GPU."""
from __future__ import annotations
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, HTTPServer
import runtime
import serve
import pilot
import owned_supervisor

class PortTests(unittest.TestCase):
    def test_actual_run_ids_accept_only_exact_32_hex(self):
        self.assertEqual(owned_supervisor.checked_run_id("a"*32),"a"*32)
        for bad in ["a"*33,"a"*31,"A"*32,"../"+"a"*32]:
            with self.assertRaises(ValueError):owned_supervisor.checked_run_id(bad)

    def test_short_runtime_path_is_bound_to_owner(self):
        config=runtime.configuration("b"*32)
        value=config["command"][config["command"].index("--runtime-dir")+1]
        self.assertEqual(value,"/tmp/a006-"+"b"*32)
        # Fixed base 46 + slash + random suffix 8 + longest socket suffix 30 = 85.
        self.assertEqual(len(value+"/ipc")+1+8+30,85)
        with self.assertRaises(ValueError):runtime.configuration(None)

    def test_runtime_path_rejects_traversal_and_nonowner_names(self):
        for value in ["/root/private_data/pipeline-"+"b"*32,"/tmp/a006-"+"B"*32,
                      "/tmp/a006-"+"b"*31,"/tmp/../tmp/a006-"+"b"*32,
                      "/tmp//a006-"+"b"*32]:
            with self.assertRaises(ValueError):serve.checked_runtime_directory(value)

    def test_runtime_path_refuses_existing_target_before_resolve(self):
        with patch.object(Path,"is_symlink",return_value=True),self.assertRaises(ValueError):
            serve.checked_runtime_directory("/tmp/a006-"+"c"*32)
        with patch.object(Path,"is_symlink",return_value=False),patch.object(Path,"exists",return_value=True),self.assertRaises(ValueError):
            serve.checked_runtime_directory("/tmp/a006-"+"c"*32)

    def test_runtime_directory_claim_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"owned"
            self.assertEqual(serve.claim_runtime_directory(path),path)
            marker=path/"keep";marker.write_text("unchanged")
            with self.assertRaises(ValueError):serve.claim_runtime_directory(path)
            self.assertEqual(marker.read_text(),"unchanged")

    @unittest.skipUnless(__import__("sys").platform=="linux","Linux symlink semantics")
    def test_runtime_directory_claim_refuses_actual_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);real=root/"real";real.mkdir();marker=real/"keep";marker.write_text("unchanged")
            link=root/"alias";link.symlink_to(real,target_is_directory=True)
            with self.assertRaises(ValueError):serve.claim_runtime_directory(link)
            with self.assertRaises(ValueError):serve.claim_runtime_directory(link/"child")
            self.assertEqual(marker.read_text(),"unchanged");self.assertFalse((real/"child").exists())

    def test_dry_stop_never_signals(self):
        with patch("sys.argv",["runtime.py","stop","--dry-run","--run-id","a"*32]), patch.object(owned_supervisor,"stop",side_effect=AssertionError("Must not stop")), contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runtime.main(),0)
        self.assertFalse(json.loads(stream.getvalue())["process_signalled"])

    def test_missing_run_identity_never_stops_current_task(self):
        with patch("sys.argv",["runtime.py","stop"]),patch.object(owned_supervisor,"stop",side_effect=AssertionError("Must not stop")),self.assertRaises(ValueError):runtime.main()

    def test_bounds_reject_invalid_or_excess_runtime(self):
        for value in ["0","3601","-2"]:
            with patch.dict("os.environ",{"AUK_MAX_RUNTIME_SECONDS":value}),self.assertRaises(ValueError):runtime.configuration("a"*32)

    def test_supervisor_source_is_exact_prior_implementation(self):
        import hashlib
        self.assertEqual(hashlib.sha256(Path(owned_supervisor.__file__).read_bytes()).hexdigest(),"0edc33da22c07a804c93047f918a4a39a58af90997390c49695d8b39a891d85b")

    def test_checkpoint_requires_real_files(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime,"MODEL",Path(temp)/"missing"), self.assertRaises(ValueError):runtime.verify_models()

    def test_http_error_body_is_preserved(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                self.send_response(400);self.send_header("Content-Type","application/json");self.end_headers();self.wfile.write(b'{"error":"bad"}')
        server=HTTPServer(("127.0.0.1",0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/"error.body";record,body=pilot.http(f"http://127.0.0.1:{server.server_port}/",path,{"test":"x"})
                self.assertEqual(record["status"],400);self.assertEqual(body,b'{"error":"bad"}')
                self.assertEqual(path.read_bytes(),body);self.assertTrue(path.with_suffix(".request.json").is_file())
        finally:server.shutdown();server.server_close();thread.join(3)

    def test_error_json_is_not_a_wav_pass(self):
        with self.assertRaises(ValueError):pilot.inspect_wav(b'{"error":"bad"}',sample_rate=24000,channels=1,seconds=3)

    def test_native_metadata_does_not_import_torch(self):
        # This test process only uses the standard-library controller.
        import sys
        self.assertNotIn("torch",sys.modules)
        metadata=pilot.metadata()
        self.assertIn("torch",metadata);self.assertNotIn("torch",sys.modules)

if __name__=="__main__":unittest.main()
