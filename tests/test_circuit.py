from kyupy.circuit import Circuit, Node, Line


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
