"""A simple and incomplete parser for the Standard Test Interface Language (STIL).

The main purpose of this parser is to load scan pattern sets from STIL files.
It supports only a very limited subset of STIL.

The functions :py:func:`load` and :py:func:`read` return an intermediate representation (:class:`StilFile` object).
Call :py:func:`StilFile.tests`, :py:func:`StilFile.tests_loc`, or :py:func:`StilFile.responses` to
obtain the appropriate vector sets.
"""

import re
from collections import namedtuple

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
                        unload[so_port] = call.parameters[so_port].replace('\n', '')
                if len(launch) > 0:
                    self.patterns.append(ScanPattern(sload, launch, capture, unload))
                    capture = {}
                    launch = {}
                sload = {}
                for si_port in self.si_ports:
                    if si_port in call.parameters:
                        sload[si_port] = call.parameters[si_port].replace('\n', '')
            if call.name.endswith('_launch') or call.name.endswith('_capture'):
                if len(launch) == 0:
                    launch = dict((k, v.replace('\n', '')) for k, v in call.parameters.items())
                else:
                    capture = dict((k, v.replace('\n', '')) for k, v in call.parameters.items())

    def _maps(self, c):
        interface = list(c.interface) + [n for n in c.nodes if 'DFF' in n.kind]
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
            scan_inversions[chain[0]] = scan_in_inversion
            scan_inversions[chain[-1]] = scan_out_inversion
        return interface, pi_map, po_map, scan_maps, scan_inversions

    def tests(self, circuit):
        """Assembles and returns a scan test pattern set for given circuit.

        This function assumes a static (stuck-at fault) test.
        """
        interface, pi_map, _, scan_maps, scan_inversions = self._maps(circuit)
        tests = logic.MVArray((len(interface), len(self.patterns)))
        for i, p in enumerate(self.patterns):
            for si_port in self.si_ports.keys():
                pattern = logic.mv_xor(p.load[si_port], scan_inversions[si_port])
                tests.data[scan_maps[si_port], i] = pattern.data[:, 0]
            tests.data[pi_map, i] = logic.MVArray(p.launch['_pi']).data[:, 0]
        return tests

    def tests_loc(self, circuit):
        """Assembles and returns a LoC scan test pattern set for given circuit.

        This function assumes a launch-on-capture (LoC) delay test.
        It performs a logic simulation to obtain the first capture pattern (the one that launches the
        delay test) and assembles the test pattern set from from pairs for initialization- and launch-patterns.
        """
        interface, pi_map, po_map, scan_maps, scan_inversions = self._maps(circuit)
        init = logic.MVArray((len(interface), len(self.patterns)), m=4)
        # init = PackedVectors(len(self.patterns), len(interface), 2)
        for i, p in enumerate(self.patterns):
            # init.set_values(i, '0' * len(interface))
            for si_port in self.si_ports.keys():
                pattern = logic.mv_xor(p.load[si_port], scan_inversions[si_port])
                init.data[scan_maps[si_port], i] = pattern.data[:, 0]
            init.data[pi_map, i] = logic.MVArray(p.launch['_pi']).data[:, 0]
        launch_bp = logic.BPArray(init)
        sim4v = LogicSim(circuit, len(init), m=4)
        sim4v.assign(launch_bp)
        sim4v.propagate()
        sim4v.capture(launch_bp)
        launch = logic.MVArray(launch_bp)
        for i, p in enumerate(self.patterns):
            # if there was no launch clock, then init = launch
            if ('P' not in p.launch['_pi']) or ('P' not in p.capture['_pi']):
                for si_port in self.si_ports.keys():
                    pattern = logic.mv_xor(p.load[si_port], scan_inversions[si_port])
                    launch.data[scan_maps[si_port], i] = pattern.data[:, 0]
            if '_pi' in p.capture and 'P' in p.capture['_pi']:
                launch.data[pi_map, i] = logic.MVArray(p.capture['_pi']).data[:, 0]
            launch.data[po_map, i] = logic.UNASSIGNED

        return logic.mv_transition(init, launch)

    def responses(self, circuit):
        """Assembles and returns a scan test response pattern set for given circuit."""
        interface, _, po_map, scan_maps, scan_inversions = self._maps(circuit)
        resp = logic.MVArray((len(interface), len(self.patterns)))
        # resp = PackedVectors(len(self.patterns), len(interface), 2)
        for i, p in enumerate(self.patterns):
            resp.data[po_map, i] = logic.MVArray(p.capture['_po'] if len(p.capture) > 0 else p.launch['_po']).data[:, 0]
            # if len(p.capture) > 0:
            #    resp.set_values(i, p.capture['_po'], po_map)
            # else:
            #    resp.set_values(i, p.launch['_po'], po_map)
            for so_port in self.so_ports.keys():
                pattern = logic.mv_xor(p.unload[so_port], scan_inversions[so_port])
                resp.data[scan_maps[so_port], i] = pattern.data[:, 0]
                # resp.set_values(i, p.unload[so_port], scan_maps[so_port], scan_inversions[so_port])
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
    start: "STIL" FLOAT _ignore _block*
    _block: signal_groups | scan_structures | pattern
        | "Header" _ignore
        | "Signals" _ignore
        | "Timing" _ignore
        | "PatternBurst" quoted _ignore
        | "PatternExec" _ignore
        | "Procedures" _ignore
        | "MacroDefs" _ignore

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

    The given file may be gzip compressed.
    """
    return parse(readtext(file))
