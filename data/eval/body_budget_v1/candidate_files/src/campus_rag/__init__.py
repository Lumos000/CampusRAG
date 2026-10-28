"""Offline campus retrieval-augmented question answering."""

import os
from pathlib import Path
import tempfile

# Keep process-local temporary files beside the project, including libraries
# that consult TEMP/TMP instead of Python's cached temporary directory.
_TEMP_DIR = Path(__file__).resolve().parents[2] / '.tmp' / 'runtime'
_TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.update(dict.fromkeys(('TEMP', 'TMP', 'TMPDIR'), str(_TEMP_DIR)))
tempfile.tempdir = str(_TEMP_DIR)

__version__ = "0.1.0"

