"""A simple and incomplete parser for the Standard Delay Format (SDF).

The main purpose of this parser is to extract pin-to-pin delay and interconnect delay information from SDF files.
Sophisticated timing specifications (timing checks, conditional delays, etc.) are currently not supported.

The functions :py:func:`load` and :py:func:`read` return an intermediate representation (:class:`DelayFile` object).
Call :py:func:`DelayFile.annotation` to match the intermediate representation to a given circuit.

"""

from collections import namedtuple

import numpy as np
from lark import Lark, Transformer

from . import log, readtext
from .techlib import TechLib


Interconnect = namedtuple('Interconnect', ['orig', 'dest', 'r', 'f'])
IOPath = namedtuple('IOPath', ['ipin', 'opin', 'r', 'f'])


class DelayFile:
    """An intermediate representation of an SDF file.
    """
    def __init__(self, name, cells):
        self.name = name
        if None in cells:
            self.interconnects = cells[None]
        else:
            self.interconnects = None
        self.cells = dict((n, l) for n, l in cells.items() if n)

    def __repr__(self):
        return '\n'.join(f'{n}: {l}' for n, l in self.cells.items()) + '\n' + \
               '\n'.join(str(i) for i in self.interconnects)

    def annotation(self, circuit, tlib=TechLib(), dataset=1, interconnect=True, ffdelays=True):
        """Constructs an 3-dimensional ndarray with timing data for each line in ``circuit``.

        An IOPATH delay for a node is annotated to the line connected to the input pin specified in the IOPATH.

        Currently, only ABSOLUTE IOPATH and INTERCONNECT delays are supported.
        Pulse rejection limits are derived from absolute delays, explicit declarations (PATHPULSE etc.) are ignored.

        :param circuit: The circuit to annotate. Names from the STIL file are matched to the node names.
        :type circuit: :class:`~kyupy.circuit.Circuit`
        :param tlib: A technology library object that provides pin name mappings.
        :type tlib: :py:class:`~kyupy.techlib.TechLib`
        :param dataset: SDFs store multiple values for each delay (e.g. minimum, typical, maximum).
            An integer selects the dataset to use (default is 1 for 'typical').
            If a tuple is given, the annotator will calculate the average of multiple datasets.
        :type dataset: ``int`` or ``tuple``
        :param interconnect: Whether or not to include the delays of interconnects in the annotation.
            To properly annotate interconnect delays, the circuit model has to include a '__fork__' node on
            every signal and every fanout-branch. The Verilog parser aids in this by setting the parameter
            `branchforks=True` in :py:func:`kyupy.verilog.parse`.
        :type interconnect: ``bool``
        :param ffdelays: Whether or not to include the delays of flip-flops in the annotation.
        :type ffdelays: ``bool``
        :return: A 3-dimensional ndarray with timing data.

            * Axis 0: line index.
            * Axis 1: type of timing data: 0='delay', 1='pulse rejection limit'.
            * Axis 2: The polarity of the output transition of the reading node: 0='rising', 1='falling'.

            The polarity for pulse rejection is determined by the latter transition of the pulse.
            E.g., ``timing[42, 1, 0]`` is the rejection limit of a negative pulse at the output
            of the reader of line 42.
        """
        def select_del(_delvals, idx):
            if isinstance(dataset, tuple):
                return sum(_delvals[idx][d] for d in dataset) / len(dataset)
            return _delvals[idx][dataset]

        def find_cell(name):
            if name not in circuit.cells:
                name = name.replace('\\', '')
            if name not in circuit.cells:
                name = name.replace('[', '_').replace(']', '_')
            if name not in circuit.cells:
                return None
            return circuit.cells[name]

        timing = np.zeros((len(circuit.lines), 2, 2))
        for cn, iopaths in self.cells.items():
            for ipn, opn, *delvals in iopaths:
                delvals = [d if len(d) > 0 else [0, 0, 0] for d in delvals]
                if max(max(delvals)) == 0:
                    continue
                cell = find_cell(cn)
                if cell is None:
                    log.warn(f'Cell from SDF not found in circuit: {cn}')
                    continue
                ipin = tlib.pin_index(cell.kind, ipn)
                opin = tlib.pin_index(cell.kind, opn)
                kind = cell.kind.lower()

                ipn2 = ipn.replace('(posedge A1)', 'A1').replace('(negedge A1)', 'A1')\
                    .replace('(posedge A2)', 'A2').replace('(negedge A2)', 'A2')

                def add_delays(_line):
                    if _line is not None:
                        timing[_line, :, 0] += select_del(delvals, 0)
                        timing[_line, :, 1] += select_del(delvals, 1)

                take_avg = False
                if kind.startswith('sdff'):
                    if not ipn.startswith('(posedge CLK'):
                        continue
                    if ffdelays and (len(cell.outs) > opin):
                        add_delays(cell.outs[opin])
                else:
                    if kind.startswith(('xor', 'xnor')):
                        ipin = tlib.pin_index(cell.kind, ipn2)
                        # print(ipn, ipin, times[cell.i_lines[ipin], 0, 0])
                        take_avg = timing[cell.ins[ipin]].sum() > 0
                    add_delays(cell.ins[ipin])
                    if take_avg:
                        timing[cell.ins[ipin]] /= 2

        if not interconnect or self.interconnects is None:
            return timing

        for n1, n2, *delvals in self.interconnects:
            delvals = [d if len(d) > 0 else [0, 0, 0] for d in delvals]
            if max(max(delvals)) == 0:
                continue
            if '/' in n1:
                i = n1.rfind('/')
                cn1 = n1[0:i]
                pn1 = n1[i+1:]
            else:
                cn1, pn1 = (n1, 'Z')
            if '/' in n2:
                i = n2.rfind('/')
                cn2 = n2[0:i]
                pn2 = n2[i+1:]
            else:
                cn2, pn2 = (n2, 'IN')
            c1 = find_cell(cn1)
            if c1 is None:
                log.warn(f'Cell from SDF not found in circuit: {cn1}')
                continue
            c2 = find_cell(cn2)
            if c2 is None:
                log.warn(f'Cell from SDF not found in circuit: {cn2}')
                continue
            p1, p2 = tlib.pin_index(c1.kind, pn1), tlib.pin_index(c2.kind, pn2)
            line = None
            f1, f2 = c1.outs[p1].reader, c2.ins[p2].driver
            if f1 != f2:  # possible branchfork
                assert len(f2.ins) == 1
                line = f2.ins[0]
                assert f1.outs[f2.ins[0].driver_pin] == line
            elif len(f2.outs) == 1:  # no fanout?
                line = f2.ins[0]
            if line is not None:
                timing[line, :, 0] += select_del(delvals, 0)
                timing[line, :, 1] += select_del(delvals, 1)
            else:
                log.warn(f'No branchfork for annotating interconnect delay {c1.name}/{p1}->{c2.name}/{p2}')
        return timing


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

    The given file may be gzip compressed.
    """
    return parse(readtext(file))
