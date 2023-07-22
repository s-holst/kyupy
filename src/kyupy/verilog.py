"""A simple and incomplete parser for Verilog files.

The main purpose of this parser is to load synthesized, non-hierarchical (flat) gate-level netlists.
It supports only a subset of Verilog.
"""

from collections import namedtuple

from lark import Lark, Transformer, Tree

from . import log, readtext
from .circuit import Circuit, Node, Line
from .techlib import NANGATE

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
    def __init__(self, branchforks=False, tlib=NANGATE):
        super().__init__()
        self.branchforks = branchforks
        self.tlib = tlib

    @staticmethod
    def name(args):
        s = args[0].value
        return s[1:-1] if s[0] == '\\' else s

    @staticmethod
    def namedpin(args):
        return tuple(args) if len(args) > 1 else (args[0], None)

    @staticmethod
    def instantiation(args):
        pinmap = {}
        for idx, pin in enumerate(args[2:]):
            p = pin.children[0]
            if isinstance(p, tuple):  # named pin
                if p[1] is not None:
                    pinmap[p[0]] = p[1]
            else:  # unnamed pin
                pinmap[idx] = p
        return Instantiation(args[0], args[1], pinmap)

    def range(self, args):
        left = int(args[0].value)
        right = int(args[1].value) if len(args) > 1 else left
        return range(left, right+1) if left <= right else range(left, right-1, -1)

    def sigsel(self, args):
        if len(args) > 1 and isinstance(args[1], range):
            l = [f'{args[0]}[{i}]' for i in args[1]]
            return l if len(l) > 1 else l[0]
        elif "'" in args[0]:
            width, rest = args[0].split("'")
            width = int(width)
            base, const = rest[0], rest[1:]
            const = int(const, {'b': 2, 'd':10, 'h':16}[base.lower()])
            l = []
            for _ in range(width):
                l.insert(0, "1'b1" if (const & 1) else "1'b0")
                const >>= 1
            return l if len(l) > 1 else l[0]
        else:
            return args[0]

    def concat(self, args):
        sigs = []
        for a in args:
            if isinstance(a, list):
                sigs += a
            else:
                sigs.append(a)
        return sigs

    def declaration(self, kind, args):
        rnge = None
        if isinstance(args[0], range):
            rnge = args[0]
            args = args[1:]
        return [SignalDeclaration(kind, signal, rnge) for signal in args]

    def input(self, args): return self.declaration("input", args)
    def output(self, args): return self.declaration("output", args)
    def inout(self, args): return self.declaration("input", args)  # just treat as input
    def wire(self, args): return self.declaration("wire", args)

    def module(self, args):
        c = Circuit(args[0])
        positions = {}
        pos = 0
        const_count = 0
        sig_decls = {}
        for decls in args[2:]:  # pass 0: collect signal declarations
            if isinstance(decls, list):
                if len(decls) > 0 and isinstance(decls[0], SignalDeclaration):
                    for decl in decls:
                        if decl.basename not in sig_decls or sig_decls[decl.basename].kind == 'wire':
                            sig_decls[decl.basename] = decl
        for intf_sig in args[1].children:
            for name in sig_decls[intf_sig].names:
                positions[name] = pos
                pos += 1
        assignments = []
        for stmt in args[2:]:  # pass 1: instantiate cells and driven signals
            if isinstance(stmt, Instantiation):
                n = Node(c, stmt.name, kind=stmt.type)
                for p, s in stmt.pins.items():
                    if self.tlib.pin_is_output(n.kind, p):
                        if s in sig_decls:
                            s = sig_decls[s].names
                            if isinstance(s, list) and len(s) == 1:
                                s = s[0]
                        Line(c, (n, self.tlib.pin_index(stmt.type, p)), Node(c, s))
            elif hasattr(stmt, 'data') and stmt.data == 'assign':
                assignments.append((stmt.children[0], stmt.children[1]))
        for sd in sig_decls.values():
            if sd.kind == 'output' or sd.kind == 'input':
                for name in sd.names:
                    n = Node(c, name, kind=sd.kind)
                    if name in positions:
                        c.io_nodes[positions[name]] = n
                    if sd.kind == 'input':
                        Line(c, n, Node(c, name))
        for target, source in assignments:  # pass 1.5: process signal assignments
            target_sigs = []
            if not isinstance(target, list): target = [target]
            for s in target:
                if s in sig_decls:
                    target_sigs += sig_decls[s].names
                else:
                    target_sigs.append(s)
            source_sigs = []
            if not isinstance(source, list): source = [source]
            for s in source:
                if s in sig_decls:
                    source_sigs += sig_decls[s].names
                else:
                    source_sigs.append(s)
            for t, s in zip(target_sigs, source_sigs):
                if t in c.forks:
                    assert s not in c.forks, 'assignment between two driven signals'
                    Line(c, c.forks[t], Node(c, s))
                elif s in c.forks:
                    assert t not in c.forks, 'assignment between two driven signals'
                    Line(c, c.forks[s], Node(c, t))
                elif s.startswith("1'b"):
                    cnode = Node(c, f'__const{s[3]}_{const_count}__', f'__const{s[3]}__')
                    const_count += 1
                    Line(c, cnode, Node(c, t))
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
                        if f'{s}[0]' in c.forks:  # actually a 1-bit bus?
                            s = f'{s}[0]'
                        else:
                            log.warn(f'Signal not driven: {s}')
                            Node(c, s)  # generate fork here
                    fork = c.forks[s]
                    if self.branchforks:
                        branchfork = Node(c, fork.name + "~" + n.name + "/" + p)
                        Line(c, fork, branchfork)
                        fork = branchfork
                    Line(c, fork, (n, self.tlib.pin_index(stmt.type, p)))
        for sd in sig_decls.values():
            if sd.kind == 'output':
                for name in sd.names:
                    if name not in c.forks:
                        if f'{name}[0]' in c.forks:  # actually a 1-bit bus?
                            name = f'{name}[0]'
                        else:
                            log.warn(f'Output not driven: {name}')
                            continue
                    Line(c, c.forks[name], c.cells[name])
        return c

    @staticmethod
    def start(args): return args[0] if len(args) == 1 else args


GRAMMAR = r"""
    start: (module)*
    module: "module" name parameters ";" (_statement)* "endmodule"
    parameters: "(" [ _namelist ] ")"
    _statement: input | output | inout | tri | wire | assign | instantiation
    input: "input" range? _namelist ";"
    output: "output" range? _namelist ";"
    inout: "inout" range? _namelist ";"
    tri: "tri" range? _namelist ";"
    wire: "wire" range? _namelist ";"
    assign: "assign" sigsel "=" sigsel ";"
    instantiation: name name "(" [ pin ( "," pin )* ] ")" ";"
    pin: namedpin | sigsel
    namedpin: "." name "(" sigsel? ")"
    range: "[" /[0-9]+/ (":" /[0-9]+/)? "]"
    sigsel: name range? | concat
    concat: "{" sigsel ( "," sigsel )*  "}"
    _namelist: name ( "," name )*
    name: ( /[a-z_][a-z0-9_]*/i | /\\[^\t \r\n]+[\t \r\n]/i | /[0-9]+'[bdh][0-9a-f]+/i )
    %import common.NEWLINE
    COMMENT: /\/\*(\*(?!\/)|[^*])*\*\// | /\(\*(\*(?!\))|[^*])*\*\)/ |  "//" /(.)*/ NEWLINE
    %ignore ( /\r?\n/ | COMMENT )+
    %ignore /[\t \f]+/
    """


def parse(text, tlib=NANGATE, branchforks=False):
    """Parses the given ``text`` as Verilog code.

    :param text: A string with Verilog code.
    :param tlib: A technology library object that defines all known cells.
    :type tlib: :py:class:`~kyupy.techlib.TechLib`
    :param branchforks: If set to ``True``, the returned circuit will include additional `forks` on each fanout branch.
        These forks are needed to correctly annotate interconnect delays
        (see :py:func:`~kyupy.sdf.DelayFile.interconnects()`).
    :return: A :py:class:`~kyupy.circuit.Circuit` object.
    """
    return Lark(GRAMMAR, parser="lalr", transformer=VerilogTransformer(branchforks, tlib)).parse(text)


def load(file, tlib=NANGATE, branchforks=False):
    """Parses the contents of ``file`` as Verilog code.

    :param file: A file name or a file handle. Files with `.gz`-suffix are decompressed on-the-fly.
    :param tlib: A technology library object that defines all known cells.
    :type tlib: :py:class:`~kyupy.techlib.TechLib`
    :param branchforks: If set to ``True``, the returned circuit will include additional `forks` on each fanout branch.
        These forks are needed to correctly annotate interconnect delays
        (see :py:func:`~kyupy.sdf.DelayFile.interconnects()`).
    :return: A :py:class:`~kyupy.circuit.Circuit` object.
    """
    return parse(readtext(file), tlib, branchforks)
