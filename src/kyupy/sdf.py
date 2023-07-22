"""A simple and incomplete parser for the Standard Delay Format (SDF).

This parser extracts pin-to-pin delay and interconnect delay information from SDF files.
Sophisticated timing specifications (timing checks, conditional delays, etc.) are ignored.

The functions :py:func:`parse` and :py:func:`load` return an intermediate representation (:class:`DelayFile` object).
Call :py:func:`DelayFile.iopaths` and :py:func:`DelayFile.interconnects` to generate delay information for a given circuit.
"""

from collections import namedtuple
import re

import numpy as np
from lark import Lark, Transformer

from . import log, readtext
from .circuit import Circuit
from .techlib import TechLib


Interconnect = namedtuple('Interconnect', ['orig', 'dest', 'r', 'f'])
IOPath = namedtuple('IOPath', ['ipin', 'opin', 'r', 'f'])


class DelayFile:
    """An intermediate representation of an SDF file.
    """
    def __init__(self, name, cells):
        self.name = name
        self._interconnects = cells.get(None, None)
        self.cells = dict((n, l) for n, l in cells.items() if n)

    def __repr__(self):
        return '\n'.join(f'{n}: {l}' for n, l in self.cells.items()) + '\n' + \
               '\n'.join(str(i) for i in self._interconnects)

    def iopaths(self, circuit:Circuit, tlib:TechLib):
        """Constructs an ndarray containing all IOPATH delays.

        All IOPATH delays for a node ``n`` are annotated to the line connected to the input pin specified in the IOPATH.

        Limited support of SDF spec:

        * Only ABSOLUTE delay values are supported.
        * Only two delvals per delval_list is supported. First is rising/posedge, second is falling/negedge
          transition at the output of the IOPATH (SDF spec, pp. 3-17).
        * PATHPULSE declarations are ignored.

        The axes convention of KyuPy's delay data arrays is as follows:

        * Axis 0: dataset (usually 3 datasets per SDF-file)
        * Axis 1: line index (e.g. ``n.ins[0]``, ``n.ins[1]``)
        * Axis 2: polarity of the transition at the IOPATH-input (e.g. at ``n.ins[0]`` or ``n.ins[1]``), 0='rising/posedge', 1='falling/negedge'
        * Axis 3: polarity of the transition at the IOPATH-output (at ``n.outs[0]``), 0='rising/posedge', 1='falling/negedge'
        """

        def find_cell(name:str):
            if name not in circuit.cells: name = name.replace('\\', '')
            if name not in circuit.cells: name = name.replace('[', '_').replace(']', '_')
            return circuit.cells.get(name, None)

        delays = np.zeros((len(circuit.lines), 2, 2, 3))  # dataset last during construction.

        for name, iopaths in self.cells.items():
            name = name.replace('\\', '')
            if cell := circuit.cells.get(name, None):
                for i_pin_spec, o_pin_spec, *dels in iopaths:
                    if i_pin_spec.startswith('(posedge '): i_pol_idxs = [0]
                    elif i_pin_spec.startswith('(negedge '): i_pol_idxs = [1]
                    else: i_pol_idxs = [0, 1]
                    i_pin_spec = re.sub(r'\((neg|pos)edge ([^)]+)\)', r'\2', i_pin_spec)
                    if line := cell.ins[tlib.pin_index(cell.kind, i_pin_spec)]:
                        delays[line, i_pol_idxs] = [d if len(d) > 0 else [0, 0, 0] for d in dels]
                    else:
                        log.warn(f'No line to annotate in circuit: {i_pin_spec} for {cell}')
            else:
                log.warn(f'Name from SDF not found in circuit: {name}')

        return np.moveaxis(delays, -1, 0)

    def interconnects(self, circuit:Circuit, tlib:TechLib):
        """Constructs an ndarray containing all INTERCONNECT delays.

        To properly annotate interconnect delays, the circuit model has to include a '__fork__' node on
        every signal and every fanout-branch. The Verilog parser aids in this by setting the parameter
        `branchforks=True` in :py:func:`~kyupy.verilog.parse` or :py:func:`~kyupy.verilog.load`.

        Limited support of SDF spec:

        * Only ABSOLUTE delay values are supported.
        * Only two delvals per delval_list is supported. First is rising/posedge, second is falling/negedge
          transition.
        * PATHPULSE declarations are ignored.

        The axes convention of KyuPy's delay data arrays is as follows:

        * Axis 0: dataset (usually 3 datasets per SDF-file)
        * Axis 1: line index. Usually input line of a __fork__.
        * Axis 2: (axis of size 2 for compatability to IOPATH results. Values are broadcast along this axis.)
        * Axis 3: polarity of the transition, 0='rising/posedge', 1='falling/negedge'
        """

        delays = np.zeros((len(circuit.lines), 2, 2, 3))  # dataset last during construction.

        for n1, n2, *delvals in self._interconnects:
            delvals = [d if len(d) > 0 else [0, 0, 0] for d in delvals]
            if max(max(delvals)) == 0: continue
            cn1, pn1 = n1.split('/') if '/' in n1 else (n1, None)
            cn2, pn2 = n2.split('/') if '/' in n2 else (n2, None)
            cn1 = cn1.replace('\\','')
            cn2 = cn2.replace('\\','')
            c1, c2 = circuit.cells[cn1], circuit.cells[cn2]
            p1 = tlib.pin_index(c1.kind, pn1) if pn1 is not None else 0
            p2 = tlib.pin_index(c2.kind, pn2) if pn2 is not None else 0
            if len(c1.outs) <= p1 or c1.outs[p1] is None:
                log.warn(f'No line to annotate pin {pn1} of {c1}')
                continue
            if len(c2.ins) <= p2 or c2.ins[p2] is None:
                log.warn(f'No line to annotate pin {pn2} of {c2}')
                continue
            f1, f2 = c1.outs[p1].reader, c2.ins[p2].driver  # find the forks between cells.
            assert f1.kind == '__fork__'
            assert f2.kind == '__fork__'
            if f1 != f2:  # at least two forks, make sure f2 is a branchfork connected to f1
                assert len(f2.outs) == 1
                assert f1.outs[f2.ins[0].driver_pin] == f2.ins[0]
                line = f2.ins[0]
            elif len(f2.outs) == 1:  # f1==f2, only OK when there is no fanout.
                line = f2.ins[0]
            else:
                log.warn(f'No branchfork to annotate interconnect delay {c1.name}/{p1}->{c2.name}/{p2}')
                continue
            delays[line, :] = delvals

        return np.moveaxis(delays, -1, 0)


def sanitize(args):
    if len(args) == 3: args.append(args[2])
    return [str(args[0]), str(args[1])] + args[2:]


class SdfTransformer(Transformer):
    @staticmethod
    def triple(args): return [float(a.value[:-1]) if len(a.value) > 1 else 0.0 for a in args]

    @staticmethod
    def interconnect(args): return Interconnect(*sanitize(args))

    @staticmethod
    def iopath(args): return IOPath(*sanitize(args))

    @staticmethod
    def cell(args):
        name = next((a for a in args if isinstance(a, str)), None)
        entries = [e for a in args if hasattr(a, 'children') for e in a.children]
        return name, entries

    @staticmethod
    def start(args):
        name = next((a for a in args if isinstance(a, str)), None)
        cells = dict(t for t in args if isinstance(t, tuple))
        return DelayFile(name, cells)


GRAMMAR = r"""
    start: "(DELAYFILE" ( "(SDFVERSION" _NOB ")"
        | "(DESIGN" "\"" NAME "\"" ")"
        | "(DATE" _NOB ")"
        | "(VENDOR" _NOB ")"
        | "(PROGRAM" _NOB ")"
        | "(VERSION" _NOB ")"
        | "(DIVIDER" _NOB ")"
        | "(VOLTAGE" _NOB ")"
        | "(PROCESS" _NOB? ")"
        | "(TEMPERATURE" _NOB ")"
        | "(TIMESCALE" _NOB ")"
        | cell )* ")"
    cell: "(CELL" ( "(CELLTYPE" _NOB ")"
        | "(INSTANCE" ID? ")"
        | "(TIMINGCHECK" _ignore* ")"
        | delay )* ")"
    delay: "(DELAY" "(ABSOLUTE" (interconnect | iopath)* ")" ")"
    interconnect: "(INTERCONNECT" ID ID triple* ")"
    iopath: "(IOPATH" ID_OR_EDGE ID_OR_EDGE triple* ")"
    NAME: /[^"]+/
    ID_OR_EDGE: ( /[^() ]+/ | "(" /[^)]+/ ")" )
    ID: ( /[^"() ]+/ | "\"" /[^"]+/ "\"" )
    triple: "(" ( /[-.0-9]*:/ /[-.0-9]*:/ /[-.0-9]*\)/ | ")" )
    _ignore: "(" _NOB? _ignore* ")" _NOB?
    _NOB: /[^()]+/
    COMMENT: "//" /[^\n]*/
    %ignore ( /\r?\n/ | COMMENT )+
    %ignore /[\t\f ]+/
    """


def parse(text):
    """Parses the given ``text`` and returns a :class:`DelayFile` object."""
    return Lark(GRAMMAR, parser="lalr", transformer=SdfTransformer()).parse(text)


def load(file):
    """Parses the contents of ``file`` and returns a :class:`DelayFile` object.

    Files with `.gz`-suffix are decompressed on-the-fly.
    """
    return parse(readtext(file))
