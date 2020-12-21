from kyupy.circuit import Circuit, Node, Line


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

    assert n1.outs[2] is None
    assert n2.ins[1] is None
    assert n2.ins[2] == line2


def test_circuit():
    c = Circuit()
    in1 = Node(c, 'in1', 'buf')
    in2 = Node(c, 'in2', 'buf')
    out1 = Node(c, 'out1', 'buf')

    assert 'in1' in c.cells
    assert 'and1' not in c.cells

    c.interface[0] = in1
    c.interface[1] = in2
    c.interface[2] = out1

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
