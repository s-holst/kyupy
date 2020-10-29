from lark import Lark, Transformer
from collections import namedtuple
import re
import gzip
from .packed_vectors import PackedVectors
from .logic_sim import LogicSim


Call = namedtuple('Call', ['name', 'parameters'])
ScanPattern = namedtuple('ScanPattern', ['load', 'launch', 'capture', 'unload'])


class StilFile:
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
        load = {}
        for call in self.calls:
            if call.name == 'load_unload':
                unload = {}
                for so_port in self.so_ports:
                    if so_port in call.parameters:
                        unload[so_port] = call.parameters[so_port].replace('\n', '')
                if len(capture) > 0:
                    self.patterns.append(ScanPattern(load, launch, capture, unload))
                    capture = {}
                    launch = {}
                load = {}
                for si_port in self.si_ports:
                    if si_port in call.parameters:
                        load[si_port] = call.parameters[si_port].replace('\n', '')
            if call.name.endswith('_launch') or call.name.endswith('_capture'):
                if len(launch) == 0:
                    launch = dict((k, v.replace('\n', '')) for k, v in call.parameters.items())
                else:
                    capture = dict((k, v.replace('\n', '')) for k, v in call.parameters.items())
    
    def _maps(self, c):
        interface = list(c.interface) + [n for n in c.nodes if 'DFF' in n.kind]
        intf_pos = dict([(n.name, i) for i, n in enumerate(interface)])
        pi_map = [intf_pos[n] for n in self.signal_groups['_pi']]
        po_map = [intf_pos[n] for n in self.signal_groups['_po']]
        scan_maps = {}
        scan_inversions = {}
        for chain_name, chain in self.scan_chains.items():
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
        
    def tests(self, c):
        interface, pi_map, po_map, scan_maps, scan_inversions = self._maps(c)
        tests = PackedVectors(len(self.patterns), len(interface), 2)
        for i, p in enumerate(self.patterns):
            for si_port in self.si_ports.keys():
                tests.set_values(i, p.load[si_port], scan_maps[si_port], scan_inversions[si_port])
            tests.set_values(i, p.launch['_pi'], pi_map)
        return tests

    def tests8v(self, c):
        interface, pi_map, po_map, scan_maps, scan_inversions = self._maps(c)
        init = PackedVectors(len(self.patterns), len(interface), 2)
        for i, p in enumerate(self.patterns):
            # init.set_values(i, '0' * len(interface))
            for si_port in self.si_ports.keys():
                init.set_values(i, p.load[si_port], scan_maps[si_port], scan_inversions[si_port])
            init.set_values(i, p.launch['_pi'], pi_map)
        sim4v = LogicSim(c, len(init), 2)
        sim4v.assign(init)
        sim4v.propagate()
        launch = init.copy()
        sim4v.capture(launch)
        for i, p in enumerate(self.patterns):
            # if there was no launch clock, then init = launch
            if ('P' not in p.launch['_pi']) or ('P' not in p.capture['_pi']):
                for si_port in self.si_ports.keys():
                    launch.set_values(i, p.load[si_port], scan_maps[si_port], scan_inversions[si_port])
            if 'P' in p.capture['_pi']:
                launch.set_values(i, p.capture['_pi'], pi_map)
        
        return PackedVectors.from_pair(init, launch)
                
    def responses(self, c):
        interface, pi_map, po_map, scan_maps, scan_inversions = self._maps(c)
        resp = PackedVectors(len(self.patterns), len(interface), 2)
        for i, p in enumerate(self.patterns):
            resp.set_values(i, p.capture['_po'], po_map)
            for so_port in self.so_ports.keys():
                resp.set_values(i, p.unload[so_port], scan_maps[so_port], scan_inversions[so_port])
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
        

def parse(stil):
    grammar = r"""
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
    if '\n' not in str(stil):  # One line?: Assuming it is a file name.
        if str(stil).endswith('.gz'):
            with gzip.open(stil, 'rt') as f:
                text = f.read()
        else:
            with open(stil, 'r') as f:
                text = f.read()
    else:
        text = str(stil)
    return Lark(grammar, parser="lalr", transformer=StilTransformer()).parse(text)


def extract_scan_pattens(stil_calls):
    pats = []
    pi = None
    scan_in = None
    for call in stil_calls:
        if call.name == 'load_unload':
            scan_out = call.parameters.get('Scan_Out')
            if scan_out is not None:
                scan_out = scan_out.replace('\n', '')
            if pi: pats.append(ScanPattern(scan_in, pi, None, scan_out))
            scan_in = call.parameters.get('Scan_In')
            if scan_in is not None:
                scan_in = scan_in.replace('\n', '')
        if call.name == 'allclock_capture':
            pi = call.parameters['_pi'].replace('\n', '')
    return pats


def match_patterns(stil_file, pats, interface):    
    intf_pos = dict([(n.name, i) for i, n in enumerate(interface)])
    pi_map = [intf_pos[n] for n in stil_file.signal_groups['_pi']]
    scan_map = [intf_pos[re.sub(r'b..\.', '', n)] for n in reversed(stil_file.scan_chains['1'])]
    # print(scan_map)
    tests = PackedVectors(len(pats), len(interface), 2)
    for i, p in enumerate(pats):
        tests.set_values(i, p.scan_in, scan_map)
        tests.set_values(i, p.pi, pi_map)

    resp = PackedVectors(len(pats), len(interface), 2)
    for i, p in enumerate(pats):
        resp.set_values(i, p.pi, pi_map)
        resp.set_values(i, p.scan_out, scan_map)

    return tests, resp

