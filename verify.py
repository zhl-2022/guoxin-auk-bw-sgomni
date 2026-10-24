"""Run only the supplied official AuK evaluator; development WAV is not accuracy."""
from __future__ import annotations
import sys
from entry import main
if __name__=="__main__":raise SystemExit(main(["verify",*sys.argv[1:]]))
