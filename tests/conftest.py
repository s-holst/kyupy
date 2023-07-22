import pytest


@pytest.fixture(scope='session')
def mydir():
    import os
    from pathlib import Path
    return Path(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))))

@pytest.fixture(scope='session')
def b15_2ig_circuit(mydir):
    from kyupy import verilog
    from kyupy.techlib import SAED32
    return verilog.load(mydir / 'b15_2ig.v.gz', branchforks=True, tlib=SAED32)

@pytest.fixture(scope='session')
def b15_2ig_delays(mydir, b15_2ig_circuit):
    from kyupy import sdf
    from kyupy.techlib import SAED32
    return sdf.load(mydir / 'b15_2ig.sdf.gz').iopaths(b15_2ig_circuit, tlib=SAED32)[1:2]
