from kyupy import bench


def test_b01(mydir):
    with open(mydir / 'b01.bench', 'r') as f:
        c = bench.parse(f.read())
        assert 92 == len(c.nodes)
    c = bench.load(mydir / 'b01.bench')
    assert 92 == len(c.nodes)


def test_simple():
    c = bench.parse('input(a, b) output(z) z=and(a,b)')
    assert len(c.nodes) == 4
    assert len(c.interface) == 3
