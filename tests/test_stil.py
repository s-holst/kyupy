from kyupy import stil


def test_b14(mydir):
    s = stil.load(mydir / 'b14.stuck.stil.gz')
    assert len(s.signal_groups) == 10
    assert len(s.scan_chains) == 1
    assert len(s.calls) == 2163
