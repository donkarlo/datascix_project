import pathlib
import pytest


@pytest.fixture(autouse=True)
def chdir_to_test_dir(request, monkeypatch):
    # Change CWD to the directory containing the current test file so that relative paths work correctly
    monkeypatch.chdir(pathlib.Path(request.fspath).parent)