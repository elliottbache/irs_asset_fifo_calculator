import pytest

@pytest.fixture
def readout(capsys) -> str:
    def _():
        return capsys.readouterr().out.replace("\r\n", "\n").rstrip("\n")
    return _

"""
@pytest.fixture
def path_to_pow_benchmark():
    return "path/to/pow_benchmark"
"""
