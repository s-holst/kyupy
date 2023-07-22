from kyupy import stil, verilog
from kyupy.techlib import SAED32

def test_b15(mydir):
    b15 = verilog.load(mydir / 'b15_2ig.v.gz', tlib=SAED32)

    s = stil.load(mydir / 'b15_2ig.sa_nf.stil.gz')
    assert len(s.signal_groups) == 10
    assert len(s.scan_chains) == 1
    assert len(s.calls) == 1357
    tests = s.tests(b15)
    resp = s.responses(b15)
    assert len(tests) > 0
    assert len(resp) > 0

    s2 = stil.load(mydir / 'b15_2ig.tf_nf.stil.gz')
    tests = s2.tests_loc(b15)
    resp = s2.responses(b15)
    assert len(tests) > 0
    assert len(resp) > 0

