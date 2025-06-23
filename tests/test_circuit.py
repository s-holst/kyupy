import pickle

from kyupy.circuit import GrowingList, Circuit, Node, Line
from kyupy import verilog, bench
from kyupy.techlib import SAED32

def test_growing_list():
    gl = GrowingList()
    assert gl.free_idx == 0
    gl[0] = 1
    assert gl.free_idx == 1
    gl[2] = 1
    assert gl.free_idx == 1
    gl[0] = None
    assert gl.free_idx == 0
    gl[0] = 1
    assert gl.free_idx == 1
    gl[1] = 1
    assert gl.free_idx == 3
    gl.append(1)
    assert gl.free_idx == 4
    gl[2] = None
    assert gl.free_idx == 2
    gl[2] = 1
    gl[1] = None
    assert gl.free_idx == 1

def test_lines():
    c = Circuit()
    n1 = Node(c, 'n1')
    n2 = Node(c, 'n2')
    line = Line(c, n1, n2)

    assert line.driver == n1
    assert line.reader == n2
    assert line.driver_pin == 0
    assert line.reader_pin == 0
    assert n1.outs[0] == line
    assert n2.ins[0] == line

    line2 = Line(c, n1, (n2, 2))

    assert line2.driver == n1
    assert line2.reader == n2
    assert line2.driver_pin == 1
    assert line2.reader_pin == 2
    assert n1.outs[0] == line
    assert n1.outs[1] == line2
    assert n2.ins[1] is None
    assert n2.ins[2] == line2

    line3 = Line(c, n1, n2)

    assert line3.driver_pin == 2
    assert line3.reader_pin == 1
    assert n1.outs[2] == line3
    assert n2.ins[1] == line3
    assert n2.ins[2] == line2

    assert len(c.lines) == 3

    line3.remove()

    assert len(c.lines) == 2
    assert c.lines[0].index == 0
    assert c.lines[1].index == 1

    assert len(n1.outs) == 2
    assert n2.ins[1] is None
    assert n2.ins[2] == line2


def test_circuit():
    c = Circuit()
    in1 = Node(c, 'in1', 'buf')
    in2 = Node(c, 'in2', 'buf')
    out1 = Node(c, 'out1', 'buf')

    assert 'in1' in c.cells
    assert 'and1' not in c.cells

    c.io_nodes[0] = in1
    c.io_nodes[1] = in2
    c.io_nodes[2] = out1

    and1 = Node(c, 'and1', kind='and')
    Line(c, in1, and1)
    Line(c, in2, and1)
    Line(c, and1, out1)

    assert len(in1.ins) == 0
    assert len(in1.outs) == 1
    assert len(in2.outs) == 1

    assert in1.outs[0].reader == and1
    assert in1.outs[0].driver == in1

    assert len(and1.ins) == 2
    assert len(and1.outs) == 1

    or1 = Node(c, 'or1', 'or')
    Line(c, and1, (or1, 1))

    or2 = Node(c, 'or2', 'or')
    or3 = Node(c, 'or3', 'or')

    assert or2.index == 5
    assert or3.index == 6

    assert len(c.nodes) == 7
    or2.remove()
    or3 = c.cells['or3']
    assert or3.index == 5
    assert 'or2' not in c.cells
    assert len(c.nodes) == 6

    c.cells['or3'].remove()
    assert 'or3' not in c.cells
    assert len(c.nodes) == 5

    repr(c)
    str(c)

    for n in c.topological_order():
        repr(n)


def test_pickle(mydir):
    c = verilog.load(mydir / 'b15_4ig.v.gz', tlib=SAED32)
    assert c is not None
    cs = pickle.dumps(c)
    assert cs is not None
    c2 = pickle.loads(cs)
    assert c == c2


def test_substitute():
    c = bench.parse('input(i1, i2, i3, i4, i5) output(o1) aoi=AOI221(i1, i2, i3, i4, i5) o1=not(aoi)')
    assert len(c.cells) == 2
    assert len(c.io_nodes) == 6
    aoi221_impl = bench.parse('input(in1, in2, in3, in4, in5) output(q) a1=and(in1, in2) a2=and(in3, in4) q=or(a1, a2, in5)')
    assert len(aoi221_impl.cells) == 3
    assert len(aoi221_impl.io_nodes) == 6
    c.substitute(c.cells['aoi'], aoi221_impl)
    assert len(c.cells) == 4
    assert len(c.io_nodes) == 6


def test_resolve(mydir):
    c = verilog.load(mydir / 'b15_4ig.v.gz', tlib=SAED32)
    s_names = [n.name for n in c.s_nodes]
    c.resolve_tlib_cells(SAED32)
    s_names_prim = [n.name for n in c.s_nodes]
    assert s_names == s_names_prim, 'resolve_tlib_cells does not preserve names or order of s_nodes'
