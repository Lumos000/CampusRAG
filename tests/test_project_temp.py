import os
from pathlib import Path
import tempfile

from campus_rag.config import ROOT


def test_python_and_library_temp_files_stay_in_project():
    expected = (ROOT/'.tmp/runtime').resolve()
    assert Path(tempfile.gettempdir()).resolve() == expected
    assert all(Path(os.environ[key]).resolve() == expected for key in ('TEMP', 'TMP', 'TMPDIR'))
    with tempfile.TemporaryDirectory() as directory:
        assert Path(directory).resolve().is_relative_to(expected)


def test_pytest_database_directory_stays_in_project(tmp_path):
    assert tmp_path.resolve().is_relative_to((ROOT/'.tmp').resolve())
