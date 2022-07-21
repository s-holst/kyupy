from kyupy import stil, verilog


def test_b14(mydir):
    b14 = verilog.load(mydir / 'b14.v.gz')
    
    s = stil.load(mydir / 'b14.stuck.stil.gz')
    assert len(s.signal_groups) == 10
    assert len(s.scan_chains) == 1
    assert len(s.calls) == 2163
    tests = s.tests(b14)
    resp = s.responses(b14)
    assert len(tests) > 0
    assert len(resp) > 0
    
    s2 = stil.load(mydir / 'b14.transition.stil.gz')
    tests = s2.tests_loc(b14)
    resp = s2.responses(b14)
    assert len(tests) > 0
    assert len(resp) > 0

