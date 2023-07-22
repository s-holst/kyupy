"""A simple and incomplete parser for the Design Exchange Format (DEF).

This parser extracts information on components and nets from DEF files and make them available
as an intermediate representation (:class:`DefFile` object).
"""

from collections import defaultdict

from lark import Lark, Transformer, Tree

from kyupy import readtext


class DefNet:
    def __init__(self, name):
        self.name = name
        self.pins = []

    @property
    def wires(self):
        ww = defaultdict(list)
        [ww[dw.layer].append((int(dw.width), dw.wire_points)) for dw in self.routed if len(dw.wire_points) > 0]
        return ww

    @property
    def vias(self):
        vv = defaultdict(list)
        [vv[vtype].extend(locs) for dw in self.routed for vtype, locs in dw.vias.items()]
        return vv


class DefWire:
    def __init__(self):
        self.layer = None
        self.width = None
        self.points = []

    @property
    def wire_points(self):
        start = [self.points[0]]
        rest = [p for p in self.points[1:] if not isinstance(p[0], str)]  # skip over vias
        return start + rest if len(rest) > 0 else []

    @property
    def vias(self):
        vv = defaultdict(list)
        loc = self.points[0]
        for p in self.points[1:]:
            if not isinstance(p[0], str):  # new location
                loc = (loc[0] if p[0] is None else p[0], loc[1] if p[1] is None else p[1])  # if None, keep previous value
                continue
            vtype, param = p
            if isinstance(param, tuple):  # expand "DO x BY y STEP xs ys"
                x_cnt, y_cnt, x_sp, y_sp = param
                [vv[vtype].append((loc[0] + x*x_sp, loc[1] + y*y_sp, 'N')) for x in range(x_cnt) for y in range(y_cnt)]
            else:
                vv[vtype].append((loc[0], loc[1], param or 'N'))
        return vv

    def __repr__(self):
        return f'<DefWire {self.layer} {self.width} {self.points}>'


class DefVia:
    def __init__(self, name):
        self.name = name
        self.rowcol = [1, 1]
        self.cutspacing = [0, 0]


class DefPin:
    def __init__(self, name):
        self.name = name
        self.points = []


class DefFile:
    """Intermediate representation of a DEF file."""
    def __init__(self):
        self.rows = []
        self.tracks = []
        self.units = []
        self.vias = {}
        self.components = {}
        self.pins = {}
        self.specialnets = {}
        self.nets = {}


class DefTransformer(Transformer):
    def __init__(self): self.def_file = DefFile()
    def start(self, args): return self.def_file
    def design(self, args): self.def_file.design = args[0].value
    def point(self, args): return tuple(int(arg.value) if arg != '*' else None for arg in args)
    def do_step(self, args): return tuple(map(int, args))
    def spnet_wires(self, args): return args[0].lower(), args[1:]
    def net_wires(self, args): return args[0].lower(), args[1:]
    def sppoints(self, args): return args
    def points(self, args): return args
    def net_pin(self, args): return '__pin__', (args[0].value, args[1].value)
    def net_opt(self, args): return args[0].lower(), args[1].value

    def file_stmt(self, args):
        value = args[1].value
        value = value[1:-1] if value[0] == '"' else value
        setattr(self.def_file, args[0].lower(), value)

    def design_stmt(self, args):
        stmt = args[0].lower()
        if stmt == 'units': self.def_file.units.append((args[1].value, args[2].value, int(args[3])))
        elif stmt == 'diearea': self.def_file.diearea = args[1:]
        elif stmt == 'row':
            self.def_file.rows.append((args[1].value,  # rowName
                                       args[2].value,  # siteName
                                       (int(args[3]), int(args[4])),  # origin x/y
                                       args[5].value,  # orientation
                                       max(args[6][0], args[6][1]),  # number of sites
                                       max(args[6][2], args[6][3])  # site width
                                      ))
        elif stmt == 'tracks':
            self.def_file.tracks.append((args[1].value,  # orientation
                                         int(args[2]),  # start
                                         int(args[3]),  # number of tracks
                                         int(args[4]),  # spacing
                                         args[5].value  # layer
                                        ))

    def vias_stmt(self, args):
        via = DefVia(args[0].value)
        [setattr(via, opt, val) for opt, val in args[1:]]
        self.def_file.vias[via.name] = via

    def vias_opt(self, args):
        opt = args[0].lower()
        if opt in ['viarule', 'pattern']: val = args[1].value
        elif opt in ['layers']: val = [arg.value for arg in args[1:]]
        else: val = [int(arg) for arg in args[1:]]
        return opt, val

    def comp_stmt(self, args):
        name = args[0].value
        kind = args[1].value
        point = args[2]
        orientation = args[3].value
        self.def_file.components[name] = (kind, point, orientation)

    def pins_stmt(self, args):
        pin = DefPin(args[0].value)
        [pin.points.append(val) if opt == 'placed' else setattr(pin, opt, val) for opt, val in args[1:]]
        self.def_file.pins[pin.name] = pin

    def pins_opt(self, args):
        opt = args[0].lower()
        if opt in ['net', 'direction', 'use']: val = args[1].value
        elif opt in ['layer']: val = [args[1].value] + args[2:]
        elif opt in ['placed']: val = (args[1][0], args[1][1], args[2].value)
        else: val = []
        return opt, val

    def spnets_stmt(self, args):
        dnet = DefNet(args[0].value)
        for arg in args[1:]:
            if arg[0] == '__pin__': dnet.pins.append(arg[1])
            else: setattr(dnet, arg[0], arg[1])
        self.def_file.specialnets[dnet.name] = dnet

    def nets_stmt(self, args):
        dnet = DefNet(args[0].value)
        for arg in args[1:]:
            if arg[0] == '__pin__': dnet.pins.append(arg[1])
            else: setattr(dnet, arg[0], arg[1])
        self.def_file.nets[dnet.name] = dnet

    def spwire(self, args):
        wire = DefWire()
        wire.layer = args[0].value
        wire.width = args[1].value
        wire.points = args[-1]
        return wire

    def wire(self, args):
        wire = DefWire()
        wire.layer = args[0].value
        wire.points = args[-1]
        return wire

    def sppoints_via(self, args):
        if len(args) == 1: return args[0].value, None
        else: return args[0].value, args[1]

    def points_via(self, args):
        if len(args) == 1: return args[0].value, 'N'
        else: return args[0].value, args[1].value.strip()


GRAMMAR = r"""
    start: /#[^\n]*/? file_stmt*

    ?file_stmt: /VERSION/ ID ";"
              | /DIVIDERCHAR/ STRING ";"
              | /BUSBITCHARS/ STRING ";"
              | design

    design: "DESIGN" ID ";" design_stmt* "END" "DESIGN"

    ?design_stmt: /UNITS/ ID ID NUMBER ";"
                | /DIEAREA/ point+ ";"
                | /ROW/ ID ID NUMBER NUMBER ID do_step ";"
                | /TRACKS/ /[XY]/ NUMBER "DO" NUMBER "STEP" NUMBER "LAYER" ID ";"
                | propdef | vias | nondef | comp | pins | pinprop | spnets | nets

    propdef: "PROPERTYDEFINITIONS" propdef_stmt* "END" "PROPERTYDEFINITIONS"
    propdef_stmt: /COMPONENTPIN/ ID ID ";"

    vias: "VIAS" NUMBER ";" vias_stmt* "END" "VIAS"
    vias_stmt: "-" ID vias_opt* ";"
    vias_opt: "+" /VIARULE/ ID
            | "+" /CUTSIZE/ NUMBER NUMBER
            | "+" /LAYERS/ ID ID ID
            | "+" /CUTSPACING/ NUMBER NUMBER
            | "+" /ENCLOSURE/ NUMBER NUMBER NUMBER NUMBER
            | "+" /ROWCOL/ NUMBER NUMBER
            | "+" /PATTERN/ ID

    nondef: "NONDEFAULTRULES" NUMBER ";" nondef_stmt+ "END" "NONDEFAULTRULES"
    nondef_stmt: "-" ID ( "+" /HARDSPACING/
                        | "+" /LAYER/ ID "WIDTH" NUMBER "SPACING" NUMBER
                        | "+" /VIA/ ID )* ";"

    comp: "COMPONENTS" NUMBER ";" comp_stmt* "END" "COMPONENTS"
    comp_stmt: "-" ID ID "+" "PLACED" point ID ";"

    pins: "PINS" NUMBER ";" pins_stmt* "END" "PINS"
    pins_stmt: "-" ID pins_opt* ";"
    pins_opt: "+" /NET/ ID
            | "+" /SPECIAL/
            | "+" /DIRECTION/ ID
            | "+" /USE/ ID
            | "+" /PORT/
            | "+" /LAYER/ ID point point
            | "+" /PLACED/ point ID

    pinprop: "PINPROPERTIES" NUMBER ";" pinprop_stmt* "END" "PINPROPERTIES"
    pinprop_stmt: "-" "PIN" ID "+" "PROPERTY" ID STRING ";"

    spnets: "SPECIALNETS" NUMBER ";" spnets_stmt* "END" "SPECIALNETS"
    spnets_stmt: "-" ID ( net_pin | net_opt | spnet_wires )* ";"

    spnet_wires: "+" ( /COVER/ | /FIXED/ | /ROUTED/ ) spwire ( "NEW" spwire )*

    spwire: ID NUMBER spwire_opt* sppoints
    spwire_opt: "+" /SHAPE/ ID
              | "+" /STYLE/ ID

    sppoints: point ( point | sppoints_via )+
    sppoints_via: ID do_step?

    nets: "NETS" NUMBER ";" nets_stmt* "END" "NETS"
    nets_stmt: "-" ID ( net_pin | net_opt | net_wires )* ";"

    net_pin: "(" ID ID ")"
    net_opt: "+" /USE/ ID
           | "+" /NONDEFAULTRULE/ ID
    net_wires: "+" ( /COVER/ | /FIXED/ | /ROUTED/ | /NOSHIELD/ ) wire ( "NEW" wire )*

    wire: ID wire_opt points
    wire_opt: ( "TAPER" | "TAPERRULE" ID )? ("STYLE" ID)?

    points: point ( point | points_via )+
    points_via: ID ORIENTATION?

    point: "(" (NUMBER|/\*/) (NUMBER|/\*/) NUMBER? ")"

    do_step: "DO" NUMBER "BY" NUMBER "STEP" (NUMBER|SIGNED_NUMBER) (NUMBER|SIGNED_NUMBER)

    ORIENTATION.2: /F?[NWES]/ WS
    ID: /[^ \t\f\r\n+][^ \t\f\r\n]*/
    STRING : "\"" /.*?/s /(?<!\\)(\\\\)*?/ "\""
    WS: /[ \t\f\r\n]/

    %import common.NUMBER
    %import common.SIGNED_NUMBER
    %ignore WS (/#[^\n]*/)?
    """


def parse(text):
    """Parses the given ``text`` and returns a :class:`DefFile` object."""
    return Lark(GRAMMAR, parser="lalr", transformer=DefTransformer()).parse(text)


def load(file):
    """Parses the contents of ``file`` and returns a :class:`DefFile` object.

    Files with `.gz`-suffix are decompressed on-the-fly.
    """
    return parse(readtext(file))