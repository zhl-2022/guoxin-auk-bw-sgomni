from __future__ import annotations
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import entry

class DeliveryTests(unittest.TestCase):
    def test_frozen_candidate_matches_all_326_source_bytes(self):
        self.assertEqual(entry.source_binding()["files_verified"],326)

    def test_all_12_cpu_wheels_have_exact_identities(self):
        self.assertEqual(len(entry.wheel_lock()["files"]),12)

    def test_dry_launch_does_not_install_or_spawn(self):
        with patch.object(entry,"source_binding",return_value={}),patch.object(entry,"prepare",side_effect=AssertionError("No prepare")),patch.object(subprocess,"run",side_effect=AssertionError("No spawn")),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(entry.main(["launch","--dry-run","--python","/missing/python"]),0)
        self.assertFalse(json.loads(output.getvalue())["environment_modified"])

    def test_stop_requires_explicit_owner(self):
        with patch.object(entry,"source_binding",return_value={}),patch.object(subprocess,"run",side_effect=AssertionError("No stop")),self.assertRaises(ValueError):entry.main(["stop"])

    def test_formal_audio_gates_do_not_fabricate_results(self):
        for action in ["verify","bench"]:
            with patch.dict(os.environ,{},clear=True),patch.object(entry,"source_binding",return_value={}),contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(entry.main([action,"--dry-run"]),2)
            plan=json.loads(output.getvalue())
            self.assertEqual(plan["track"],"T04");self.assertFalse(plan["official_acceptance"])
            self.assertFalse(plan["filesystem_mutated"]);self.assertFalse(plan["subprocess_executed"])

    def test_canonical_log_keeps_old_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(entry,"ROOT",Path(temp)):
            path=Path(temp)/"env.log";path.write_bytes(b"original raw log\n")
            entry.canonical_log("env.log",{"new":True})
            archived=Path(temp)/"runtime/log-archive"/(hashlib.sha256(b"original raw log\n").hexdigest()+"-env.log")
            self.assertEqual(archived.read_bytes(),b"original raw log\n")
            self.assertTrue(json.loads(path.read_text())["new"])

    @unittest.skipUnless(os.name=="posix","Linux venv interpreter symlink regression")
    def test_existing_venv_interpreter_symlink_keeps_venv_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            venv=Path(temp)/"env"
            subprocess.run([sys.executable,"-B","-m","venv","--without-pip","--system-site-packages",str(venv)],check=True,timeout=60)
            executable=venv/"bin/python"
            self.assertTrue(executable.is_symlink())
            with patch.object(entry,"wheel_lock",return_value={"files":[]}),patch.object(entry,"CORE",{}):
                result=entry.prepare(executable)
            self.assertTrue(result["existing_venv_checked_readonly"])
            self.assertEqual(Path(result["metadata"]["prefix"]),venv)

if __name__=="__main__":unittest.main()
