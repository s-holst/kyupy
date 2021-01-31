"""A simple and incomplete parser for Verilog files.

The main purpose of this parser is to load synthesized, non-hierarchical (flat) gate-level netlists.
It supports only a very limited subset of Verilog.
"""

from collections import namedtuple

from lark import Lark, Transformer

from . import log, readtext
from .circuit import Circuit, Node, Line
from .techlib import TechLib

Instantiation = namedtuple('Instantiation', ['type', 'name', 'pins'])


class SignalDeclaration:

    def __init__(self, kind, name, rnge=None):
        self.left = None
        self.right = None
        self.kind = kind
        self.basename = name
        self.rnge = rnge

    @property
    def names(self):
        if self.rnge is None:
            return [self.basename]
        return [f'{self.basename}[{i}]' for i in self.rnge]

    def __repr__(self):
        return f"{self.kind}:{self.basename}[{self.rnge}]"


class VerilogTransformer(Transformer):
    def __init__(self, branchforks=False, tlib=TechLib()):
        super().__init__()
        self._signal_declarations = {}
        self.branchforks = branchforks
        self.tlib = tlib

    @staticmethod
    def name(args):
        s = args[0].value
        if s[0] == '\\':
            s = s[1:-1]
        return s

    @staticmethod
    def instantiation(args):
        return Instantiation(args[0], args[1],
                             dict((pin.children[0],
                                   pin.children[1]) for pin in args[2:] if len(pin.children) > 1))

    def range(self, args):
        left = int(args[0].value)
        right = int(args[1].value)
        return range(left, right+1) if left <= right else range(left, right-1, -1)

    def declaration(self, kind, args):
        rnge = None
        if isinstance(args[0], range):
            rnge = args[0]
            args = args[1:]
        for sd in [SignalDeclaration(kind, signal, rnge) for signal in args]:
            if kind != 'wire' or sd.basename not in self._signal_declarations:
                self._signal_declarations[sd.basename] = sd

    def input(self, args): self.declaration("input", args)
    def output(self, args): self.declaration("output", args)
    def inout(self, args): self.declaration("input", args)  # just treat as input
    def wire(self, args): self.declaration("wire", args)

    def module(self, args):
        c = Circuit(args[0])
        positions = {}
        pos = 0
        const_count = 0
        for intf_sig in args[1].children:
            for name in self._signal_declarations[intf_sig].names:
                positions[name] = pos
                pos += 1
        assignments = []
        for stmt in args[2:]:  # pass 1: instantiate cells and driven signals
            if isinstance(stmt, Instantiation):
                n = Node(c, stmt.name, kind=stmt.type)
                for p, s in stmt.pins.items():
                    if self.tlib.pin_is_output(n.kind, p):
                        Line(c, (n, self.tlib.pin_index(stmt.type, p)), Node(c, s))
            elif stmt is not None and stmt.data == 'assign':
                assignments.append((stmt.children[0], stmt.children[1]))
        for sd in self._signal_declarations.values():
            if sd.kind == 'output' or sd.kind == 'input':
                for name in sd.names:
                    n = Node(c, name, kind=sd.kind)
                    if name in positions:
                        c.interface[positions[name]] = n
                    if sd.kind == 'input':
                        Line(c, n, Node(c, name))
        for s1, s2 in assignments:  # pass 1.5: process signal assignments
            if s1 in c.forks:
                assert s2 not in c.forks, 'assignment between two driven signals'
                Line(c, c.forks[s1], Node(c, s2))
            elif s2 in c.forks:
                assert s1 not in c.forks, 'assignment between two driven signals'
                Line(c, c.forks[s2], Node(c, s1))
            elif s2.startswith("1'b"):
                cnode = Node(c, f'__const{s2[3]}_{const_count}__', f'__const{s2[3]}__')
                const_count += 1
                Line(c, cnode, Node(c, s1))
        for stmt in args[2:]:  # pass 2: connect signals to readers
            if isinstance(stmt, Instantiation):
                for p, s in stmt.pins.items():
                    n = c.cells[stmt.name]
                    if self.tlib.pin_is_output(n.kind, p): continue
                    if s.startswith("1'b"):
                        cname = f'__const{s[3]}_{const_count}__'
                        cnode = Node(c, cname, f'__const{s[3]}__')
                        const_count += 1
                        s = cname
                        Line(c, cnode, Node(c, s))
                    if s not in c.forks:
                        log.warn(f'Signal not driven: {s}')
                        Node(c, s)  # generate fork here
                    fork = c.forks[s]
                    if self.branchforks:
                        branchfork = Node(c, fork.name + "~" + n.name + "/" + p)
                        Line(c, fork, branchfork)
                        fork = branchfork
                    Line(c, fork, (n, self.tlib.pin_index(stmt.type, p)))
        for sd in self._signal_declarations.values():
            if sd.kind == 'output':
                for name in sd.names:
                    if name not in c.forks:
                        log.warn(f'Output not driven: {name}')
                    else:
                        Line(c, c.forks[name], c.cells[name])
        return c

    @staticmethod
    def start(args): return args[0] if len(args) == 1 else args


GRAMMAR = """
    start: (module)*
    module: "module" name parameters ";" (_statement)* "endmodule"
    parameters: "(" [ _namelist ] ")"
    _statement: input | output | inout | tri | wire | assign | instantiation
    input: "input" range? _namelist ";"
    output: "output" range? _namelist ";"
    inout: "inout" range? _namelist ";"
    tri: "tri" range? _namelist ";"
    wire: "wire" range? _namelist ";"
    assign: "assign" name "=" name ";"
    instantiation: name name "(" [ pin ( "," pin )* ] ")" ";"
    pin: "." name "(" name? ")"
    range: "[" /[0-9]+/ ":" /[0-9]+/ "]"

    _namelist: name ( "," name )*
    name: ( /[a-z_][a-z0-9_\\[\\]]*/i | /\\\\[^\\t \\r\\n]+[\\t \\r\\n](\\[[0-9]+\\])?/i | /1'b0/i | /1'b1/i )
    COMMENT: "//" /[^\\n]*/
    %ignore ( /\\r?\\n/ | COMMENT )+
    %ignore /[\\t \\f]+/
    """


def parse(text, *, branchforks=False, tlib=TechLib()):
    """Parses the given ``text`` as Verilog code.

    :param text: A string with Verilog code.
    :param branchforks: If set to ``True``, the returned circuit will include additional `forks` on each fanout branch.
        These forks are needed to correctly annotate interconnect delays
        (see :py:func:`kyupy.sdf.DelayFile.annotation`).
    :param tlib: A technology library object that provides pin name mappings.
    :type tlib: :py:class:`~kyupy.techlib.TechLib`
    :return: A :class:`~kyupy.circuit.Circuit` object.
    """
    return Lark(GRAMMAR, parser="lalr", transformer=VerilogTransformer(branchforks, tlib)).parse(text)


def load(file, *args, **kwargs):
    """Parses the contents of ``file`` as Verilog code.

    The given file may be gzip compressed. Takes the same keyword arguments as :py:func:`parse`.
    """
    return parse(readtext(file), *args, **kwargs)
