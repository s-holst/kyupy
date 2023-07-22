from kyupy import bench


def test_b01(mydir):
    with open(mydir / 'b01.bench', 'r') as f:
        c = bench.parse(f.read())
        assert len(c.nodes) == 92
    c = bench.load(mydir / 'b01.bench')
    assert len(c.nodes) == 92


def test_simple():
    c = bench.parse('input(a, b) output(z) z=and(a,b)')
    assert len(c.nodes) == 4
    assert len(c.io_nodes) == 3
