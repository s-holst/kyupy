import pytest


@pytest.fixture
def mydir():
    import os
    from pathlib import Path
    return Path(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))))
