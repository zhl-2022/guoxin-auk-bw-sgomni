"""Run the measured AuK fixed-root verifier through the template entrypoint."""
import sys
from entry031 import main

if __name__ == '__main__':
    sys.argv.insert(1, 'verify')
    raise SystemExit(main())
