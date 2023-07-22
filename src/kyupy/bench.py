"""A parser for the ISCAS89 benchmark format.

The ISCAS89 benchmark format (`.bench`-suffix) is a very simple textual description of gate-level netlists.
Historically it was first used in the
`ISCAS89 benchmark set <https://people.engr.ncsu.edu/brglez/CBL/benchmarks/ISCAS89/>`_.
Besides loading these benchmarks, this module is also useful for easily constructing simple circuits:
``c = bench.parse('input(x, y) output(a, o, n) a=and(x,y) o=or(x,y) n=not(x)')``.
"""

from lark import Lark, Transformer

from .circuit import Circuit, Node, Line
from . import readtext


class BenchTransformer(Transformer):

    def __init__(self, name):
        super().__init__()
        self.c = Circuit(name)

    def start(self, _): return self.c

    def parameters(self, args): return [self.c.get_or_add_fork(str(name)) for name in args]

    def interface(self, args): self.c.io_nodes.extend(args[0])

    def assignment(self, args):
        name, cell_type, drivers = args
        cell = Node(self.c, str(name), str(cell_type))
        Line(self.c, cell, self.c.get_or_add_fork(str(name)))
        for d in drivers: Line(self.c, d, cell)


GRAMMAR = r"""
    start: (statement)*
    statement: input | output | assignment
    input: ("INPUT" | "input") parameters -> interface
    output: ("OUTPUT" | "output") parameters -> interface
    assignment: NAME "=" NAME parameters
    parameters: "(" [ NAME ( "," NAME )* ] ")"
    NAME: /[-_a-z0-9]+/i
    %ignore ( /\r?\n/ | "#" /[^\n]*/ | /[\t\f ]/ )+
    """


def parse(text, name=None):
    """Parses the given ``text`` as ISCAS89 bench code.

    :param text: A string with bench code.
    :param name: The name of the circuit. Circuit names are not included in bench descriptions.
    :return: A :class:`Circuit` object.
    """
    return Lark(GRAMMAR, parser="lalr", transformer=BenchTransformer(name)).parse(text)


def load(file, name=None):
    """Parses the contents of ``file`` as ISCAS89 bench code.

    :param file: The file to be loaded. Files with `.gz`-suffix are decompressed on-the-fly.
    :param name: The name of the circuit. If None, the file name is used as circuit name.
    :return: A :class:`Circuit` object.
    """
    return parse(readtext(file), name=name or str(file))
