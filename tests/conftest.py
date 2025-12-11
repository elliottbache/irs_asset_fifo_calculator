import pytest
import os
import sys

# Repo root = one level up from tests/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture
def readout(capsys) -> str:
    def _():
        return capsys.readouterr().out.replace("\r\n", "\n").rstrip("\n")
    return _
