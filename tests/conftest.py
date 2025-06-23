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
def b15_2ig_circuit_resolved(b15_2ig_circuit):
    from kyupy.techlib import SAED32
    cr = b15_2ig_circuit.copy()
    cr.resolve_tlib_cells(SAED32)
    return cr

@pytest.fixture(scope='session')
def b15_4ig_circuit(mydir):
    from kyupy import verilog
    from kyupy.techlib import SAED32
    return verilog.load(mydir / 'b15_4ig.v.gz', branchforks=True, tlib=SAED32)

@pytest.fixture(scope='session')
def b15_4ig_circuit_resolved(b15_4ig_circuit):
    from kyupy.techlib import SAED32
    cr = b15_4ig_circuit.copy()
    cr.resolve_tlib_cells(SAED32)
    return cr

@pytest.fixture(scope='session')
def b15_2ig_delays(mydir, b15_2ig_circuit):
    from kyupy import sdf
    from kyupy.techlib import SAED32
    return sdf.load(mydir / 'b15_2ig.sdf.gz').iopaths(b15_2ig_circuit, tlib=SAED32)[1:2]

@pytest.fixture(scope='session')
def b15_2ig_sa_nf_test_resp(mydir, b15_2ig_circuit_resolved):
    from kyupy import stil
    s = stil.load(mydir / 'b15_2ig.sa_nf.stil.gz')
    tests = s.tests(b15_2ig_circuit_resolved)[:,1:]
    resp = s.responses(b15_2ig_circuit_resolved)[:,1:]
    return (tests, resp)

@pytest.fixture(scope='session')
def b15_4ig_sa_rf_test_resp(mydir, b15_4ig_circuit_resolved):
    from kyupy import stil
    s = stil.load(mydir / 'b15_4ig.sa_rf.stil.gz')
    tests = s.tests(b15_4ig_circuit_resolved)[:,1:]
    resp = s.responses(b15_4ig_circuit_resolved)[:,1:]
    return (tests, resp)
