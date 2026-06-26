"""Put the repo root on sys.path so tests can import `src`, `eval`, and the
root-level `validate_submission` module."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
