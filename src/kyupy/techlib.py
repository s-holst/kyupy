from .circuit import Node, Line


def add_and_connect(circuit, name, kind, in1=None, in2=None, out=None):
    n = Node(circuit, name, kind)
    if in1 is not None:
        n.ins[0] = in1
        in1.reader = n
        in1.reader_pin = 0
    if in2 is not None:
        n.ins[1] = in2
        in2.reader = n
        in2.reader_pin = 1
    if out is not None:
        n.outs[0] = out
        out.driver = n
        out.driver_pin = 0
    return n


class TechLib:
    """Provides some information specific to standard cell libraries necessary
    for loading gate-level designs. :py:class:`~kyupy.circuit.Node` objects do not
    have pin names. The methods defined here map pin names to pin directions and defined
    positions in the ``node.ins`` and ``node.outs`` lists. The default implementation
    provides mappings for SAED-inspired standard cell libraries.
    """

    @staticmethod
    def pin_index(kind, pin):
        """Returns a pin list position for a given node kind and pin name."""
        for prefix, pins, index in [('HADD', ('B0', 'SO'), 1),
                                    ('MUX21', ('S',), 2),
                                    ('DFF', ('QN',), 1),
                                    ('SDFF', ('QN',), 1),
                                    ('SDFF', ('CLK',), 3),
                                    ('SDFF', ('RSTB',), 4),
                                    ('SDFF', ('SETB',), 5)]:
            if kind.startswith(prefix) and pin in pins: return index
        for index, pins in enumerate([('A1', 'IN1', 'D', 'S', 'INP', 'A', 'Q', 'QN', 'Y', 'Z', 'ZN'),
                                      ('A2', 'IN2', 'CLK', 'CO', 'SE', 'B'),
                                      ('A3', 'IN3', 'RSTB', 'CI', 'SI'),
                                      ('A4', 'IN4', 'SETB'),
                                      ('A5', 'IN5'),
                                      ('A6', 'IN6')]):
            if pin in pins: return index
        raise ValueError(f'Unknown pin index for {kind}.{pin}')

    @staticmethod
    def pin_is_output(kind, pin):
        """Returns True, if given pin name of a node kind is an output."""
        if 'MUX' in kind and pin == 'S': return False
        return pin in ('Q', 'QN', 'Z', 'ZN', 'Y', 'CO', 'S', 'SO', 'C1')

    @staticmethod
    def split_complex_gates(circuit):
        node_list = circuit.nodes
        for n in node_list:
            name = n.name
            ins = n.ins
            outs = n.outs
            if n.kind.startswith('AO21X'):
                n.remove()
                n_and = add_and_connect(circuit, name+'~and', 'AND2', ins[0], ins[1], None)
                n_or = add_and_connect(circuit, name+'~or', 'OR2', None, ins[2], outs[0])
                Line(circuit, n_and, n_or)
            elif n.kind.startswith('AOI21X'):
                n.remove()
                n_and = add_and_connect(circuit, name+'~and', 'AND2', ins[0], ins[1], None)
                n_nor = add_and_connect(circuit, name+'~nor', 'NOR2', None, ins[2], outs[0])
                Line(circuit, n_and, n_nor)
            elif n.kind.startswith('OA21X'):
                n.remove()
                n_or = add_and_connect(circuit, name+'~or', 'OR2', ins[0], ins[1], None)
                n_and = add_and_connect(circuit, name+'~and', 'AND2', None, ins[2], outs[0])
                Line(circuit, n_or, n_and)
            elif n.kind.startswith('OAI21X'):
                n.remove()
                n_or = add_and_connect(circuit, name+'~or', 'OR2', ins[0], ins[1], None)
                n_nand = add_and_connect(circuit, name+'~nand', 'NAND2', None, ins[2], outs[0])
                Line(circuit, n_or, n_nand)
            elif n.kind.startswith('OA22X'):
                n.remove()
                n_or0 = add_and_connect(circuit, name+'~or0', 'OR2', ins[0], ins[1], None)
                n_or1 = add_and_connect(circuit, name+'~or1', 'OR2', ins[2], ins[3], None)
                n_and = add_and_connect(circuit, name+'~and', 'AND2', None, None, outs[0])
                Line(circuit, n_or0, n_and)
                Line(circuit, n_or1, n_and)
            elif n.kind.startswith('OAI22X'):
                n.remove()
                n_or0 = add_and_connect(circuit, name+'~or0', 'OR2', ins[0], ins[1], None)
                n_or1 = add_and_connect(circuit, name+'~or1', 'OR2', ins[2], ins[3], None)
                n_nand = add_and_connect(circuit, name+'~nand', 'NAND2', None, None, outs[0])
                Line(circuit, n_or0, n_nand)
                Line(circuit, n_or1, n_nand)
            elif n.kind.startswith('AO22X'):
                n.remove()
                n_and0 = add_and_connect(circuit, name+'~and0', 'AND2', ins[0], ins[1], None)
                n_and1 = add_and_connect(circuit, name+'~and1', 'AND2', ins[2], ins[3], None)
                n_or = add_and_connect(circuit, name+'~or', 'OR2', None, None, outs[0])
                Line(circuit, n_and0, n_or)
                Line(circuit, n_and1, n_or)
            elif n.kind.startswith('AOI22X'):
                n.remove()
                n_and0 = add_and_connect(circuit, name+'~and0', 'AND2', ins[0], ins[1], None)
                n_and1 = add_and_connect(circuit, name+'~and1', 'AND2', ins[2], ins[3], None)
                n_nor = add_and_connect(circuit, name+'~nor', 'NOR2', None, None, outs[0])
                Line(circuit, n_and0, n_nor)
                Line(circuit, n_and1, n_nor)
            elif n.kind.startswith('AO221X'):
                n.remove()
                n_and0 = add_and_connect(circuit, name+'~and0', 'AND2', ins[0], ins[1], None)
                n_and1 = add_and_connect(circuit, name+'~and1', 'AND2', ins[2], ins[3], None)
                n_or0 = add_and_connect(circuit, name+'~or0', 'OR2', None, None, None)
                n_or1 = add_and_connect(circuit, name+'~or1', 'OR2', None, ins[4], outs[0])
                Line(circuit, n_and0, n_or0)
                Line(circuit, n_and1, n_or0)
                Line(circuit, n_or0, n_or1)
            elif n.kind.startswith('AOI221X'):
                n.remove()
                n_and0 = add_and_connect(circuit, name+'~and0', 'AND2', ins[0], ins[1], None)
                n_and1 = add_and_connect(circuit, name+'~and1', 'AND2', ins[2], ins[3], None)
                n_or = add_and_connect(circuit, name+'~or', 'OR2', None, None, None)
                n_nor = add_and_connect(circuit, name+'~nor', 'NOR2', None, ins[4], outs[0])
                Line(circuit, n_and0, n_or)
                Line(circuit, n_and1, n_or)
                Line(circuit, n_or, n_nor)
            elif n.kind.startswith('OA221X'):
                n.remove()
                n_or0 = add_and_connect(circuit, name+'~or0', 'OR2', ins[0], ins[1], None)
                n_or1 = add_and_connect(circuit, name+'~or1', 'OR2', ins[2], ins[3], None)
                n_and0 = add_and_connect(circuit, name+'~and0', 'AND2', None, None, None)
                n_and1 = add_and_connect(circuit, name+'~and1', 'AND2', None, ins[4], outs[0])
                Line(circuit, n_or0, n_and0)
                Line(circuit, n_or1, n_and0)
                Line(circuit, n_and0, n_and1)
            elif n.kind.startswith('OAI221X'):
                n.remove()
                n_or0 = add_and_connect(circuit, name+'~or0', 'OR2', ins[0], ins[1], None)
                n_or1 = add_and_connect(circuit, name+'~or1', 'OR2', ins[2], ins[3], None)
                n_and0 = add_and_connect(circuit, name+'~and0', 'AND2', None, None, None)
                n_nand1 = add_and_connect(circuit, name+'~nand1', 'NAND2', None, ins[4], outs[0])
                Line(circuit, n_or0, n_and0)
                Line(circuit, n_or1, n_and0)
                Line(circuit, n_and0, n_nand1)
            elif n.kind.startswith('AO222X'):
                n.remove()
                n_and0 = add_and_connect(circuit, name+'~and0', 'AND2', ins[0], ins[1], None)
                n_and1 = add_and_connect(circuit, name+'~and1', 'AND2', ins[2], ins[3], None)
                n_and2 = add_and_connect(circuit, name+'~and2', 'AND2', ins[4], ins[5], None)
                n_or0 = add_and_connect(circuit, name+'~or0', 'OR2', None, None, None)
                n_or1 = add_and_connect(circuit, name+'~or1', 'OR2', None, None, outs[0])
                Line(circuit, n_and0, n_or0)
                Line(circuit, n_and1, n_or0)
                Line(circuit, n_and2, n_or1)
                Line(circuit, n_or0, n_or1)
            elif n.kind.startswith('AOI222X'):
                n.remove()
                n_and0 = add_and_connect(circuit, name+'~and0', 'AND2', ins[0], ins[1], None)
                n_and1 = add_and_connect(circuit, name+'~and1', 'AND2', ins[2], ins[3], None)
                n_and2 = add_and_connect(circuit, name+'~and2', 'AND2', ins[4], ins[5], None)
                n_or0 = add_and_connect(circuit, name+'~or0', 'OR2', None, None, None)
                n_nor1 = add_and_connect(circuit, name+'~nor1', 'NOR2', None, None, outs[0])
                Line(circuit, n_and0, n_or0)
                Line(circuit, n_and1, n_or0)
                Line(circuit, n_and2, n_nor1)
                Line(circuit, n_or0, n_nor1)
            elif n.kind.startswith('OA222X'):
                n.remove()
                n_or0 = add_and_connect(circuit, name+'~or0', 'OR2', ins[0], ins[1], None)
                n_or1 = add_and_connect(circuit, name+'~or1', 'OR2', ins[2], ins[3], None)
                n_or2 = add_and_connect(circuit, name+'~or2', 'OR2', ins[4], ins[5], None)
                n_and0 = add_and_connect(circuit, name+'~and0', 'AND2', None, None, None)
                n_and1 = add_and_connect(circuit, name+'~and1', 'AND2', None, None, outs[0])
                Line(circuit, n_or0, n_and0)
                Line(circuit, n_or1, n_and0)
                Line(circuit, n_or2, n_and1)
                Line(circuit, n_and0, n_and1)
            elif n.kind.startswith('OAI222X'):
                n.remove()
                n0 = add_and_connect(circuit, name+'~or0', 'OR2', ins[0], ins[1], None)
                n1 = add_and_connect(circuit, name+'~or1', 'OR2', ins[2], ins[3], None)
                n2 = add_and_connect(circuit, name+'~or2', 'OR2', ins[4], ins[5], None)
                n3 = add_and_connect(circuit, name+'~and0', 'AND2', None, None, None)
                n4 = add_and_connect(circuit, name+'~nand1', 'NAND2', None, None, outs[0])
                Line(circuit, n0, n3)
                Line(circuit, n1, n3)
                Line(circuit, n2, n4)
                Line(circuit, n3, n4)
            elif n.kind.startswith('AND3X'):
                n.remove()
                n0 = add_and_connect(circuit, name+'~and0', 'AND2', ins[0], ins[1], None)
                n1 = add_and_connect(circuit, name+'~and1', 'AND2', None, ins[2], outs[0])
                Line(circuit, n0, n1)
            elif n.kind.startswith('OR3X'):
                n.remove()
                n0 = add_and_connect(circuit, name+'~or0', 'OR2', ins[0], ins[1], None)
                n1 = add_and_connect(circuit, name+'~or1', 'OR2', None, ins[2], outs[0])
                Line(circuit, n0, n1)
            elif n.kind.startswith('XOR3X'):
                n.remove()
                n0 = add_and_connect(circuit, name+'~xor0', 'XOR2', ins[0], ins[1], None)
                n1 = add_and_connect(circuit, name+'~xor1', 'XOR2', None, ins[2], outs[0])
                Line(circuit, n0, n1)
            elif n.kind.startswith('NAND3X'):
                n.remove()
                n0 = add_and_connect(circuit, name+'~and', 'AND2', ins[0], ins[1], None)
                n1 = add_and_connect(circuit, name+'~nand', 'NAND2', None, ins[2], outs[0])
                Line(circuit, n0, n1)
            elif n.kind.startswith('NOR3X'):
                n.remove()
                n0 = add_and_connect(circuit, name+'~or', 'OR2', ins[0], ins[1], None)
                n1 = add_and_connect(circuit, name+'~nor', 'NOR2', None, ins[2], outs[0])
                Line(circuit, n0, n1)
            elif n.kind.startswith('XNOR3X'):
                n.remove()
                n0 = add_and_connect(circuit, name+'~xor', 'XOR2', ins[0], ins[1], None)
                n1 = add_and_connect(circuit, name+'~xnor', 'XNOR2', None, ins[2], outs[0])
                Line(circuit, n0, n1)
            elif n.kind.startswith('AND4X'):
                n.remove()
                n0 = add_and_connect(circuit, name+'~and0', 'AND2', ins[0], ins[1], None)
                n1 = add_and_connect(circuit, name+'~and1', 'AND2', ins[2], ins[3], None)
                n2 = add_and_connect(circuit, name+'~and2', 'AND2', None, None, outs[0])
                Line(circuit, n0, n2)
                Line(circuit, n1, n2)
            elif n.kind.startswith('OR4X'):
                n.remove()
                n0 = add_and_connect(circuit, name+'~or0', 'OR2', ins[0], ins[1], None)
                n1 = add_and_connect(circuit, name+'~or1', 'OR2', ins[2], ins[3], None)
                n2 = add_and_connect(circuit, name+'~or2', 'OR2', None, None, outs[0])
                Line(circuit, n0, n2)
                Line(circuit, n1, n2)
            elif n.kind.startswith('NAND4X'):
                n.remove()
                n0 = add_and_connect(circuit, name+'~and0', 'AND2', ins[0], ins[1], None)
                n1 = add_and_connect(circuit, name+'~and1', 'AND2', ins[2], ins[3], None)
                n2 = add_and_connect(circuit, name+'~nand2', 'NAND2', None, None, outs[0])
                Line(circuit, n0, n2)
                Line(circuit, n1, n2)
            elif n.kind.startswith('NOR4X'):
                n.remove()
                n0 = add_and_connect(circuit, name+'~or0', 'OR2', ins[0], ins[1], None)
                n1 = add_and_connect(circuit, name+'~or1', 'OR2', ins[2], ins[3], None)
                n2 = add_and_connect(circuit, name+'~nor2', 'NOR2', None, None, outs[0])
                Line(circuit, n0, n2)
                Line(circuit, n1, n2)
            elif n.kind.startswith('FADDX'):
                n.remove()
                # forks for fan-outs
                f_a = add_and_connect(circuit, name + '~fork0', '__fork__', ins[0])
                f_b = add_and_connect(circuit, name + '~fork1', '__fork__', ins[1])
                f_ci = add_and_connect(circuit, name + '~fork2', '__fork__', ins[2])
                f_ab = Node(circuit, name + '~fork3')
                # sum-block
                n_xor0 = Node(circuit, name + '~xor0', 'XOR2')
                Line(circuit, f_a, n_xor0)
                Line(circuit, f_b, n_xor0)
                Line(circuit, n_xor0, f_ab)
                if len(outs) > 0 and outs[0] is not None:
                    n_xor1 = add_and_connect(circuit, name + '~xor1', 'XOR2', None, None, outs[0])
                    Line(circuit, f_ab, n_xor1)
                    Line(circuit, f_ci, n_xor1)
                # carry-block
                if len(outs) > 1 and outs[1] is not None:
                    n_and0 = Node(circuit, name + '~and0', 'AND2')
                    Line(circuit, f_ab, n_and0)
                    Line(circuit, f_ci, n_and0)
                    n_and1 = Node(circuit, name + '~and1', 'AND2')
                    Line(circuit, f_a, n_and1)
                    Line(circuit, f_b, n_and1)
                    n_or = add_and_connect(circuit, name + '~or0', 'OR2', None, None, outs[1])
                    Line(circuit, n_and0, n_or)
                    Line(circuit, n_and1, n_or)
            elif n.kind.startswith('HADDX'):
                n.remove()
                # forks for fan-outs
                f_a = add_and_connect(circuit, name + '~fork0', '__fork__', ins[0])
                f_b = add_and_connect(circuit, name + '~fork1', '__fork__', ins[1])
                n_xor0 = add_and_connect(circuit, name + '~xor0', 'XOR2', None, None, outs[1])
                Line(circuit, f_a, n_xor0)
                Line(circuit, f_b, n_xor0)
                n_and0 = add_and_connect(circuit, name + '~and0', 'AND2', None, None, outs[0])
                Line(circuit, f_a, n_and0)
                Line(circuit, f_b, n_and0)
            elif n.kind.startswith('MUX21X'):
                n.remove()
                f_s = add_and_connect(circuit, name + '~fork0', '__fork__', ins[2])
                n_not = Node(circuit, name + '~not', 'INV')
                Line(circuit, f_s, n_not)
                n_and0 = add_and_connect(circuit, name + '~and0', 'AND2', ins[0])
                n_and1 = add_and_connect(circuit, name + '~and1', 'AND2', ins[1])
                n_or0 = add_and_connect(circuit, name + '~or0', 'OR2', None, None, outs[0])
                Line(circuit, n_not, n_and0)
                Line(circuit, f_s, n_and1)
                Line(circuit, n_and0, n_or0)
                Line(circuit, n_and1, n_or0)
            elif n.kind.startswith('DFFSSR'):
                n.kind = 'DFFX1'
                n_and0 = add_and_connect(circuit, name + '~and0', 'AND2', ins[0], ins[2], None)
                Line(circuit, n_and0, (n, 0))
