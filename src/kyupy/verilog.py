from collections import namedtuple
import gzip

from lark import Lark, Transformer

from .circuit import Circuit, Node, Line
from .saed import pin_index, pin_is_output

Instantiation = namedtuple('Instantiation', ['type', 'name', 'pins'])


class SignalDeclaration:
    
    def __init__(self, kind, tokens):
        self.left = None
        self.right = None
        self.kind = kind
        if len(tokens.children) == 1:
            self.basename = tokens.children[0]
        else:
            self.basename = tokens.children[2]
            self.left = int(tokens.children[0].value)
            self.right = int(tokens.children[1].value)
    
    @property
    def names(self):
        if self.left is None:
            return [self.basename]
        if self.left <= self.right:
            return [f'{self.basename}[{i}]' for i in range(self.left, self.right + 1)]
        else:
            return [f'{self.basename}[{i}]' for i in range(self.left, self.right - 1, -1)]
        
    def __repr__(self):
        return f"{self.kind}:{self.basename}[{self.left}:{self.right}]"


class VerilogTransformer(Transformer):
    def __init__(self, branchforks=False):
        super().__init__()
        self._signal_declarations = {}
        self.branchforks = branchforks

    @staticmethod
    def name(args):
        s = args[0].value
        if s[0] == '\\':
            s = s[1:-1]
        return s

    @staticmethod
    def instantiation(args):
        return Instantiation(args[0], args[1],
                             dict([(pin.children[0], pin.children[1]) for pin in args[2:]]))
       
    def input(self, args):
        for sd in [SignalDeclaration('input', signal) for signal in args]:
            self._signal_declarations[sd.basename] = sd
    
    def inout(self, args):
        for sd in [SignalDeclaration('input', signal) for signal in args]:  # just treat as input
            self._signal_declarations[sd.basename] = sd
    
    def output(self, args):
        for sd in [SignalDeclaration('output', signal) for signal in args]:
            self._signal_declarations[sd.basename] = sd
            
    def wire(self, args):
        for sd in [SignalDeclaration('wire', signal) for signal in args]:
            self._signal_declarations[sd.basename] = sd
                
    def module(self, args):
        c = Circuit(args[0])
        positions = {}
        pos = 0
        for intf_sig in args[1].children:
            for name in self._signal_declarations[intf_sig].names:
                positions[name] = pos
                pos += 1
        assignments = []
        for stmt in args[2:]:  # pass 1: instantiate cells and driven signals
            if type(stmt) is Instantiation:
                n = Node(c, stmt.name, kind=stmt.type)
                for p, s in stmt.pins.items():
                    if pin_is_output(n.kind, p):
                        Line(c, (n, pin_index(stmt.type, p)), Node(c, s))
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
        for stmt in args[2:]:  # pass 2: connect signals to readers
            if type(stmt) is Instantiation:
                for p, s in stmt.pins.items():
                    n = c.cells[stmt.name]
                    if pin_is_output(n.kind, p): continue
                    if s.startswith("1'b"):
                        const = f'__const{s[3]}__'
                        if const not in c.cells:
                            Line(c, Node(c, const, const), Node(c, s))
                    fork = c.forks[s]
                    if self.branchforks:
                        branchfork = Node(c, fork.name + "~" + n.name + "/" + p)
                        Line(c, fork, branchfork)
                        fork = branchfork
                    Line(c, fork, (n, pin_index(stmt.type, p)))
        for sd in self._signal_declarations.values():
            if sd.kind == 'output':
                for name in sd.names:
                    Line(c, c.forks[name], c.cells[name])
        return c

    @staticmethod
    def start(args):
        if len(args) == 1:
            return args[0]
        else:
            return args


grammar = """
    start: (module)*
    module: "module" name parameters ";" (_statement)* "endmodule"
    parameters: "(" [ name ( "," name )* ] ")"
    _statement: input | output | inout | tri | wire | assign | instantiation
    input: "input" signal ( "," signal )* ";"
    output: "output" signal ( "," signal )* ";"
    inout: "inout" signal ( "," signal )* ";"
    tri: "tri" name ";"
    wire: "wire" signal ( "," signal )* ";"
    assign: "assign" name "=" name ";"
    instantiation: name name "(" [ pin ( "," pin )* ] ")" ";"
    pin: "." name "(" name ")"
    signal: ( name | "[" /[0-9]+/ ":" /[0-9]+/ "]" name )

    name: ( /[a-z_][a-z0-9_\\[\\]]*/i | /\\\\[^\\t \\r\\n]+[\\t \\r\\n](\\[[0-9]+\\])?/i | /1'b0/i | /1'b1/i )
    COMMENT: "//" /[^\\n]*/
    %ignore ( /\\r?\\n/ | COMMENT )+
    %ignore /[\\t \\f]+/
    """


def loads(s, *, branchforks=False):
    return Lark(grammar, parser="lalr", transformer=VerilogTransformer(branchforks)).parse(s)


def load(fp, *, branchforks=False):
    return loads(fp.read(), branchforks=branchforks)


def parse(verilog, branchforks=False):
    if '\n' not in str(verilog):  # One line?: Assuming it is a file name.
        if str(verilog).endswith('.gz'):
            with gzip.open(verilog, 'rt') as f:
                text = f.read()
        else:
            with open(verilog, 'r') as f:
                text = f.read()
    else:
        text = str(verilog)
    return loads(text, branchforks=branchforks)
