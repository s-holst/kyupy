"""A simple and incomplete parser for the Standard Test Interface Language (STIL).

The main purpose of this parser is to load scan pattern sets from STIL files.
It supports only a subset of STIL.

The functions :py:func:`parse` and :py:func:`load` return an intermediate representation (:py:class:`StilFile` object).
Call :py:func:`StilFile.tests()`, :py:func:`StilFile.tests_loc()`, or :py:func:`StilFile.responses()` to
obtain the appropriate vector sets.
"""

import re
from collections import namedtuple

import numpy as np
from lark import Lark, Transformer

from . import readtext, logic
from .logic_sim import LogicSim


Call = namedtuple('Call', ['name', 'parameters'])
ScanPattern = namedtuple('ScanPattern', ['load', 'launch', 'capture', 'unload'])


class StilFile:
    """An intermediate representation of a STIL file.
    """
    def __init__(self, version, signal_groups, scan_chains, calls):
        self.version = version
        self.signal_groups = signal_groups
        self.scan_chains = scan_chains
        self.si_ports = dict((v[0], k) for k, v in scan_chains.items())
        self.so_ports = dict((v[-1], k) for k, v in scan_chains.items())
        self.calls = calls
        self.patterns = []
        launch = {}
        capture = {}
        sload = {}
        for call in self.calls:
            if call.name == 'load_unload':
                unload = {}
                for so_port in self.so_ports:
                    if so_port in call.parameters:
                        unload[so_port] = call.parameters[so_port].replace('\n', '').replace('N', '-')
                if len(capture) > 0:
                    self.patterns.append(ScanPattern(sload, launch, capture, unload))
                    capture = {}
                    launch = {}
                sload = {}
                for si_port in self.si_ports:
                    if si_port in call.parameters:
                        sload[si_port] = call.parameters[si_port].replace('\n', '').replace('N', '-')
            if call.name.endswith('_launch'):
                launch = dict((k, v.replace('\n', '').replace('N', '-')) for k, v in call.parameters.items())
            if call.name.endswith('_capture'):
                capture = dict((k, v.replace('\n', '').replace('N', '-')) for k, v in call.parameters.items())

    def _maps(self, c):
        interface = list(c.io_nodes) + [n for n in c.nodes if 'DFF' in n.kind]
        intf_pos = dict((n.name, i) for i, n in enumerate(interface))
        pi_map = [intf_pos[n] for n in self.signal_groups['_pi']]
        po_map = [intf_pos[n] for n in self.signal_groups['_po']]
        scan_maps = {}
        scan_inversions = {}
        for chain in self.scan_chains.values():
            scan_map = []
            scan_in_inversion = []
            scan_out_inversion = []
            inversion = False
            for n in chain[1:-1]:
                if n == '!':
                    inversion = not inversion
                else:
                    scan_in_inversion.append(inversion)
            scan_in_inversion = list(reversed(scan_in_inversion))
            inversion = False
            for n in reversed(chain[1:-1]):
                if n == '!':
                    inversion = not inversion
                else:
                    scan_map.append(intf_pos[n])
                    scan_out_inversion.append(inversion)
            scan_maps[chain[0]] = scan_map
            scan_maps[chain[-1]] = scan_map
            scan_inversions[chain[0]] = logic.mvarray(scan_in_inversion)[0]
            scan_inversions[chain[-1]] = logic.mvarray(scan_out_inversion)[0]
        return interface, pi_map, po_map, scan_maps, scan_inversions

    def tests(self, circuit):
        """Assembles and returns a scan test pattern set for given circuit.

        This function assumes a static (stuck-at fault) test.

        :param circuit: The circuit to assemble the patterns for. The patterns will follow the
            :py:attr:`~kyupy.circuit.Circuit.s_nodes` ordering of the this circuit.
        :return: A 4-valued multi-valued (mv) logic array (see :py:mod:`~kyupy.logic`).
            The values for primary inputs and sequential elements are filled, the primary outputs are left unassigned.
        """
        interface, pi_map, _, scan_maps, scan_inversions = self._maps(circuit)
        tests = np.full((len(interface), len(self.patterns)), logic.UNASSIGNED)
        for i, p in enumerate(self.patterns):
            for si_port in self.si_ports.keys():
                pattern = logic.mvarray(p.load[si_port])
                inversions = np.choose((pattern == logic.UNASSIGNED) | (pattern == logic.UNKNOWN),
                                       [scan_inversions[si_port], logic.ZERO]).astype(np.uint8)
                np.bitwise_xor(pattern, inversions, out=pattern)
                tests[scan_maps[si_port], i] = pattern
            tests[pi_map, i] = logic.mvarray(p.capture['_pi'])
        return tests

    def tests_loc(self, circuit, init_filter=None, launch_filter=None):
        """Assembles and returns a LoC scan test pattern set for given circuit.

        This function assumes a launch-on-capture (LoC) delay test.
        It performs a logic simulation to obtain the first capture pattern (the one that launches the delay
        test) and assembles the test pattern set from from pairs for initialization- and launch-patterns.

        :param circuit: The circuit to assemble the patterns for. The patterns will follow the
            :py:attr:`~kyupy.circuit.Circuit.s_nodes` ordering of the this circuit.
        :param init_filter: A function for filtering the initialization patterns. This function is called
            with the initialization patterns from the STIL file as mvarray before logic simulation.
            It shall return an mvarray with the same shape. This function can be used, for example, to fill
            patterns.
        :param launch_filter: A function for filtering the launch patterns. This function is called
            with the launch patterns generated by logic simulation before they are combined with
            the initialization patterns to form the final 8-valued test patterns.
            The function shall return an mvarray with the same shape. This function can be used, for example, to fill
            patterns.
        :return: An 8-valued multi-valued (mv) logic array (see :py:mod:`~kyupy.logic`). The values for primary
            inputs and sequential elements are filled, the primary outputs are left unassigned.
        """
        interface, pi_map, po_map, scan_maps, scan_inversions = self._maps(circuit)
        init = np.full((len(interface), len(self.patterns)), logic.UNASSIGNED)
        for i, p in enumerate(self.patterns):
            # init.set_values(i, '0' * len(interface))
            for si_port in self.si_ports.keys():
                pattern = logic.mvarray(p.load[si_port])
                inversions = np.choose((pattern == logic.UNASSIGNED) | (pattern == logic.UNKNOWN),
                                       [scan_inversions[si_port], logic.ZERO]).astype(np.uint8)
                np.bitwise_xor(pattern, inversions, out=pattern)
                init[scan_maps[si_port], i] = pattern
            init[pi_map, i] = logic.mvarray(p.launch['_pi'] if '_pi' in p.launch else p.capture['_pi'])
        if init_filter: init = init_filter(init)
        sim8v = LogicSim(circuit, init.shape[-1], m=8)
        sim8v.s[0] = logic.mv_to_bp(init)
        sim8v.s_to_c()
        sim8v.c_prop()
        sim8v.c_to_s()
        launch = logic.bp_to_mv(sim8v.s[1])[..., :init.shape[-1]]
        for i, p in enumerate(self.patterns):
            # if there was no launch cycle or launch clock, then init = launch
            if '_pi' not in p.launch or 'P' not in p.launch['_pi'] or 'P' not in p.capture['_pi']:
                for si_port in self.si_ports.keys():
                    pattern = logic.mv_xor(logic.mvarray(p.load[si_port]), scan_inversions[si_port])
                    launch[scan_maps[si_port], i] = pattern
            if '_pi' in p.capture and 'P' in p.capture['_pi']:
                launch[pi_map, i] = logic.mvarray(p.capture['_pi'])
            launch[po_map, i] = logic.UNASSIGNED
        if launch_filter: launch = launch_filter(launch)

        return logic.mv_transition(init, launch)

    def responses(self, circuit):
        """Assembles and returns a scan test response pattern set for given circuit.

        :param circuit: The circuit to assemble the patterns for. The patterns will follow the
            :py:attr:`~kyupy.circuit.Circuit.s_nodes` ordering of the this circuit.
        :return: A 4-valued multi-valued (mv) logic array (see :py:mod:`~kyupy.logic`).
            The values for primary outputs and sequential elements are filled, the primary inputs are left unassigned.
        """
        interface, _, po_map, scan_maps, scan_inversions = self._maps(circuit)
        resp = np.full((len(interface), len(self.patterns)), logic.UNASSIGNED)
        for i, p in enumerate(self.patterns):
            resp[po_map, i] = logic.mvarray(p.capture['_po'] if len(p.capture) > 0 else p.launch['_po'])
            for so_port in self.so_ports.keys():
                pattern = logic.mv_xor(logic.mvarray(p.unload[so_port]), scan_inversions[so_port])
                resp[scan_maps[so_port], i] = pattern
        return resp


class StilTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self._signal_groups = None
        self._calls = None
        self._scan_chains = None

    @staticmethod
    def quoted(args): return args[0][1:-1]

    @staticmethod
    def call(args): return Call(args[0], dict(args[1:]))

    @staticmethod
    def call_parameter(args): return args[0], args[1].value

    @staticmethod
    def signal_group(args): return args[0], args[1:]

    @staticmethod
    def scan_chain(args):
        scan_in = None
        scan_cells = None
        scan_out = None
        for t in args[1:]:
            if t.data == 'scan_in':
                scan_in = t.children[0]
            elif t.data == 'scan_out':
                scan_out = t.children[0]
            if t.data == 'scan_cells':
                scan_cells = [n.replace('.SI', '') for n in t.children]
                scan_cells = [re.sub(r'.*\.', '', s) if '.' in s else s for s in scan_cells]
        return args[0], ([scan_in] + scan_cells + [scan_out])

    def signal_groups(self, args): self._signal_groups = dict(args)

    def pattern(self, args): self._calls = [c for c in args if isinstance(c, Call)]

    def scan_structures(self, args): self._scan_chains = dict(args)

    def start(self, args):
        return StilFile(float(args[0]), self._signal_groups, self._scan_chains, self._calls)


GRAMMAR = r"""
    start: "STIL" FLOAT ( _ignore | ";" ) _block*
    _block: signal_groups | scan_structures | pattern
        | "Header" _ignore
        | "Signals" _ignore
        | "Timing" _ignore
        | "PatternBurst" quoted _ignore
        | "PatternExec" _ignore
        | "Procedures" _ignore
        | "MacroDefs" _ignore
        | "UserKeywords" /[a-zA-Z]*;/

    signal_groups: "SignalGroups" "{" signal_group* "}"
    signal_group: quoted "=" "'" quoted ( "+" quoted)* "'" _ignore? ";"?

    scan_structures: "ScanStructures" "{" scan_chain* "}"
    scan_chain: "ScanChain" quoted "{" ( scan_length
        | scan_in | scan_out | scan_inversion | scan_cells | scan_master_clock )* "}"
    scan_length: "ScanLength" /[0-9]+/ ";"
    scan_in: "ScanIn" quoted ";"
    scan_out: "ScanOut" quoted ";"
    scan_inversion: "ScanInversion" /[0-9]+/ ";"
    scan_cells: "ScanCells" (quoted | /!/)* ";"
    scan_master_clock: "ScanMasterClock" quoted ";"

    pattern: "Pattern" quoted "{" ( label | w | c | macro | ann | call )* "}"
    label: quoted ":"
    w: "W" quoted ";"
    c: "C" _ignore
    macro: "Macro" quoted ";"
    ann: "Ann" _ignore
    call: "Call" quoted "{" call_parameter* "}"
    call_parameter: quoted "=" /[^;]+/ ";"

    quoted: /"[^"]*"/
    FLOAT: /[-0-9.]+/
    _ignore: "{" _NOB? _ignore_inner* "}"
    _ignore_inner: "{" _NOB? _ignore_inner* "}" _NOB?
    _NOB: /[^{}]+/
    %ignore ( /\r?\n/ | "//" /[^\n]*/ | /[\t\f ]/ )+
    """


def parse(text):
    """Parses the given ``text`` and returns a :class:`StilFile` object."""
    return Lark(GRAMMAR, parser="lalr", transformer=StilTransformer()).parse(text)


def load(file):
    """Parses the contents of ``file`` and returns a :class:`StilFile` object.

    Files with `.gz`-suffix are decompressed on-the-fly.
    """
    return parse(readtext(file))
