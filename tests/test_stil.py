from kyupy import stil


def test_b14(mydir):
    s = stil.parse(mydir / 'b14.stil.gz')
    assert 10 == len(s.signal_groups)
    assert 1 == len(s.scan_chains)
    assert 3259 == len(s.calls)

