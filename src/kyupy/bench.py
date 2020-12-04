from lark import Lark, Transformer
from .circuit import Circuit, Node, Line


class BenchTransformer(Transformer):
    
    def __init__(self, name):
        super().__init__()
        self.c = Circuit(name)
    
    def start(self, _): return self.c
        
    def parameters(self, args): return [self.c.get_or_add_fork(name) for name in args]
        
    def interface(self, args): self.c.interface.extend(args[0])

    def assignment(self, args):
        name, cell_type, drivers = args
        cell = Node(self.c, str(name), str(cell_type))
        Line(self.c, cell, self.c.get_or_add_fork(str(name)))
        [Line(self.c, d, cell) for d in drivers]
        

def parse(bench):
    grammar = r"""
    start: (statement)*
    statement: input | output | assignment
    input: ("INPUT" | "input") parameters -> interface
    output: ("OUTPUT" | "output") parameters -> interface
    assignment: NAME "=" NAME parameters
    parameters: "(" [ NAME ( "," NAME )* ] ")"
    NAME: /[-_a-z0-9]+/i
    %ignore ( /\r?\n/ | "#" /[^\n]*/ | /[\t\f ]/ )+
    """
    name = None
    if '(' not in str(bench):  # No parentheses?: Assuming it is a file name.
        name = str(bench).replace('.bench', '')
        with open(bench, 'r') as f:
            text = f.read()
    else:
        text = bench
    return Lark(grammar, parser="lalr", transformer=BenchTransformer(name)).parse(text)

