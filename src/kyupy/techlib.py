"""KyuPy's Built-In Technology Libraries

Technology libraries provide cell definitions and their implementation with simulation primitives.
A couple of common standard cell libraries are built-in.
Others can be easily added by providing a bench-like description of the cells.
"""

import re
from itertools import product

from . import bench


class TechLibOld:
    @staticmethod
    def pin_index(kind, pin):
        if isinstance(pin, int):
            return max(0, pin-1)
        if kind[:3] in ('OAI', 'AOI'):
            if pin[0] == 'A': return int(pin[1]) - 1
            if pin == 'B': return int(kind[3])
            if pin[0] == 'B': return int(pin[1]) - 1 + int(kind[3])
        for prefix, pins, index in [('HADD', ('B0', 'SO'), 1),
                                    ('HADD', ('A0', 'C1'), 0),
                                    ('MUX21', ('S', 'S0'), 2),
                                    ('MX2', ('S0',), 2),
                                    ('TBUF', ('OE',), 1),
                                    ('TINV', ('OE',), 1),
                                    ('LATCH', ('D',), 0),
                                    ('LATCH', ('QN',), 1),
                                    ('DFF', ('D',), 0),
                                    ('DFF', ('QN',), 1),
                                    ('SDFF', ('D',), 0),
                                    ('SDFF', ('QN',), 1),
                                    ('SDFF', ('CLK',), 3),
                                    ('SDFF', ('RSTB', 'RN'), 4),
                                    ('SDFF', ('SETB',), 5),
                                    ('ISOL', ('ISO',), 0),
                                    ('ISOL', ('D',), 1)]:
            if kind.startswith(prefix) and pin in pins: return index
        for index, pins in enumerate([('A1', 'IN1', 'A', 'S', 'INP', 'I', 'Q', 'QN', 'Y', 'Z', 'ZN'),
                                      ('A2', 'IN2', 'B', 'CK', 'CLK', 'CO', 'SE'),
                                      ('A3', 'IN3', 'C', 'RN', 'RSTB', 'CI', 'SI'),
                                      ('A4', 'IN4', 'D', 'SN', 'SETB'),
                                      ('A5', 'IN5', 'E'),
                                      ('A6', 'IN6', 'F')]):
            if pin in pins: return index
        raise ValueError(f'Unknown pin index for {kind}.{pin}')

    @staticmethod
    def pin_is_output(kind, pin):
        if isinstance(pin, int):
            return pin == 0
        if 'MUX' in kind and pin == 'S': return False
        return pin in ('Q', 'QN', 'Z', 'ZN', 'Y', 'CO', 'S', 'SO', 'C1')


class TechLib:
    """Class for standard cell library definitions.

    :py:class:`~kyupy.circuit.Node` objects do not have pin names.
    This class maps pin names to pin directions and defined positions in the ``node.ins`` and ``node.outs`` lists.
    Furthermore, it gives access to implementations of complex cells. See also :py:func:`~kyupy.circuit.substitute` and
    :py:func:`~kyupy.circuit.resolve_tlib_cells`.
    """
    def __init__(self, lib_src):
        self.cells = dict()
        """A dictionary with pin definitions and circuits for each cell kind (type).
        """
        for c_str in re.split(r';\s+', lib_src):
            c_str = re.sub(r'^\s+', '', c_str)
            name_len = c_str.find(' ')
            if name_len <= 0: continue
            c = bench.parse(c_str[name_len:])
            c.name = c_str[:name_len]
            c.eliminate_1to1_forks()
            i_idx, o_idx = 0, 0
            pin_dict = dict()
            for n in c.io_nodes:
                if len(n.ins) == 0:
                    pin_dict[n.name] = (i_idx, False)
                    i_idx += 1
                else:
                    pin_dict[n.name] = (o_idx, True)
                    o_idx += 1
            parts = [s[1:-1].split(',') if s[0] == '{' else [s] for s in re.split(r'({[^}]+})', c.name) if len(s) > 0]
            for name in [''.join(item) for item in product(*parts)]:
                self.cells[name] = (c, pin_dict)

    def pin_index(self, kind, pin):
        """Returns a pin list position for a given node kind and pin name."""
        assert kind in self.cells, f'Unknown cell: {kind}'
        assert pin in self.cells[kind][1], f'Unknown pin: {pin} for cell {kind}'
        return self.cells[kind][1][pin][0]

    def pin_is_output(self, kind, pin):
        """Returns True, if given pin name of a node kind is an output."""
        assert kind in self.cells, f'Unknown cell: {kind}'
        assert pin in self.cells[kind][1], f'Unknown pin: {pin} for cell {kind}'
        return self.cells[kind][1][pin][1]


GSC180 = TechLib(r"""
BUFX{1,3}      input(A)    output(Y) Y=BUF1(A)    ;
CLKBUFX{1,2,3} input(A)    output(Y) Y=BUF1(A)    ;
INVX{1,2,4,8}  input(A)    output(Y) Y=INV1(A)    ;
TBUFX{1,2,4,8} input(A,OE) output(Y) Y=AND2(A,OE) ;
TINVX1         input(A,OE) output(Y) AB=INV1(A) Y=AND2(AB,OE) ;

AND2X1      input(A,B)     output(Y) Y=AND2(A,B)      ;
NAND2X{1,2} input(A,B)     output(Y) Y=NAND2(A,B)     ;
NAND3X1     input(A,B,C)   output(Y) Y=NAND3(A,B,C)   ;
NAND4X1     input(A,B,C,D) output(Y) Y=NAND4(A,B,C,D) ;
OR2X1       input(A,B)     output(Y) Y=OR2(A,B)       ;
OR4X1       input(A,B,C,D) output(Y) Y=OR4(A,B,C,D)   ;
NOR2X1      input(A,B)     output(Y) Y=NOR2(A,B)      ;
NOR3X1      input(A,B,C)   output(Y) Y=NOR3(A,B,C)    ;
NOR4X1      input(A,B,C,D) output(Y) Y=NOR4(A,B,C,D)  ;
XOR2X1      input(A,B)     output(Y) Y=XOR2(A,B)      ;

MX2X1   input(A,B,S0)            output(Y)    Y=MUX21(A,B,S0)      ;
AOI21X1 input(A0,A1,B0)          output(Y)    Y=AOI21(A0,A1,B0)    ;
AOI22X1 input(A0,A1,B0,B1)       output(Y)    Y=AOI22(A0,A1,B0,B1) ;
OAI21X1 input(A0,A1,B0)          output(Y)    Y=OAI21(A0,A1,B0)    ;
OAI22X1 input(A0,A1,B0,B1)       output(Y)    Y=OAI22(A0,A1,B0,B1) ;
OAI33X1 input(A0,A1,A2,B0,B1,B2) output(Y)    AA=OR2(A0,A1) BB=OR2(B0,B1) Y=OAI22(AA,A2,BB,B2) ;
ADDFX1  input(A,B,CI)            output(CO,S) AB=XOR2(A,B) CO=XOR2(AB,CI) S=AO22(AB,CI,A,B)    ;
ADDHX1  input(A,B)               output(CO,S) CO=XOR2(A,B) S=AND2(A,B)                         ;

DFFX1    input(CK,D)             output(Q,QN) Q=DFF(D,CK) QN=INV1(Q) ;
DFFSRX1  input(CK,D,RN,SN)       output(Q,QN) DR=AND2(D,RN) SET=INV1(SN) DRS=OR2(DR,SET) Q=DFF(DRS,CK) QN=INV1(Q) ;
SDFFSRX1 input(CK,D,RN,SE,SI,SN) output(Q,QN) DR=AND2(D,RN) SET=INV1(SN) DRS=OR2(DR,SET) DI=MUX21(DRS,SI,SE) Q=DFF(DI,CK) QN=INV1(Q) ;

TLATSRX1 input(D,G,RN,SN) output(Q,QN) DR=AND2(D,RN) SET=INV1(SN) DRS=OR2(DR,SET) Q=LATCH(DRS,G) QN=INV1(Q) ;
TLATX1   input(C,D)       output(Q,QN) Q=LATCH(D,C) QN=INV1(Q) ;
""")
"""The GSC 180nm generic standard cell library.
"""


_nangate_common = r"""
FILLCELL_X{1,2,4,8,16,32} ;

LOGIC0_X1 output(Z) Z=__const0__() ;
LOGIC1_X1 output(Z) Z=__const1__() ;

BUF_X{1,2,4,8,16,32}  input(A) output(Z)  Z=BUF1(A)  ;
CLKBUF_X{1,2,3}       input(A) output(Z)  Z=BUF1(A)  ;

NAND2_X{1,2,4} input(A1,A2)       output(ZN) ZN=NAND2(A1,A2)       ;
NAND3_X{1,2,4} input(A1,A2,A3)    output(ZN) ZN=NAND3(A1,A2,A3)    ;
NAND4_X{1,2,4} input(A1,A2,A3,A4) output(ZN) ZN=NAND4(A1,A2,A3,A4) ;
NOR2_X{1,2,4}  input(A1,A2)       output(ZN) ZN=NOR2(A1,A2)        ;
NOR3_X{1,2,4}  input(A1,A2,A3)    output(ZN) ZN=NOR3(A1,A2,A3)     ;
NOR4_X{1,2,4}  input(A1,A2,A3,A4) output(ZN) ZN=NOR4(A1,A2,A3,A4)  ;

AOI21_X{1,2,4} input(A,B1,B2)     output(ZN) ZN=AOI21(B1,B2,A)     ;
OAI21_X{1,2,4} input(A,B1,B2)     output(ZN) ZN=OAI21(B1,B2,A)     ;
AOI22_X{1,2,4} input(A1,A2,B1,B2) output(ZN) ZN=AOI22(A1,A2,B1,B2) ;
OAI22_X{1,2,4} input(A1,A2,B1,B2) output(ZN) ZN=OAI22(A1,A2,B1,B2) ;

OAI211_X{1,2,4} input(A,B,C1,C2) output(ZN) ZN=OAI211(C1,C2,A,B)   ;
AOI211_X{1,2,4} input(A,B,C1,C2) output(ZN) ZN=AOI211(C1,C2,A,B)   ;

MUX2_X{1,2} input(A,B,S) output(Z) Z=MUX21(A,B,S) ;

AOI221_X{1,2,4} input(A,B1,B2,C1,C2) output(ZN) BC=AO22(B1,B2,C1,C2) ZN=NOR2(BC,A)  ;
OAI221_X{1,2,4} input(A,B1,B2,C1,C2) output(ZN) BC=OA22(B1,B2,C1,C2) ZN=NAND2(BC,A) ;

AOI222_X{1,2,4} input(A1,A2,B1,B2,C1,C2) output(ZN) BC=AO22(B1,B2,C1,C2) ZN=AOI21(A1,A2,BC) ;
OAI222_X{1,2,4} input(A1,A2,B1,B2,C1,C2) output(ZN) BC=OA22(B1,B2,C1,C2) ZN=OAI21(A1,A2,BC) ;

OAI33_X1 input(A1,A2,A3,B1,B2,B3) output(ZN) AA=OR2(A1,A2) BB=OR2(B1,B2) ZN=OAI22(AA,A3,BB,B3) ;

HA_X1 input(A,B) output(CO,S) CO=XOR2(A,B) S=AND2(A,B) ;

FA_X1 input(A,B,CI) output(CO,S) AB=XOR2(A,B) CO=XOR2(AB,CI) S=AO22(CI,A,B) ;

CLKGATE_X{1,2,4,8} input(CK,E) output(GCK) GCK=AND2(CK,E) ;

CLKGATETST_X{1,2,4,8} input(CK,E,SE) output(GCK) GCK=OA21(CK,E,SE) ;

DFF_X{1,2}   input(D,CK)       output(Q,QN)  Q=DFF(D,CK) QN=INV1(Q) ;
DFFR_X{1,2}  input(D,RN,CK)    output(Q,QN)  DR=AND2(D,RN) Q=DFF(DR,CK) QN=INV1(Q) ;
DFFS_X{1,2}  input(D,SN,CK)    output(Q,QN)  S=INV1(SN) DS=OR2(D,S) Q=DFF(DS,CK) QN=INV1(Q) ;
DFFRS_X{1,2} input(D,RN,SN,CK) output(Q,QN)  S=INV1(SN) DS=OR2(D,S) DRS=AND2(DS,RN) Q=DFF(DRS,CK) QN=INV1(Q) ;

SDFF_X{1,2}   input(D,SE,SI,CK)       output(Q,QN)  DI=MUX21(D,SI,SE) Q=DFF(DI,CK) QN=INV1(Q) ;
SDFFR_X{1,2}  input(D,RN,SE,SI,CK)    output(Q,QN)  DR=AND2(D,RN) DI=MUX21(DR,SI,SE) Q=DFF(DI,CK) QN=INV1(Q) ;
SDFFS_X{1,2}  input(D,SE,SI,SN,CK)    output(Q,QN)  S=INV1(SN) DS=OR2(D,S) DI=MUX21(DS,SI,SE) Q=DFF(DI,CK) QN=INV1(Q) ;
SDFFRS_X{1,2} input(D,RN,SE,SI,SN,CK) output(Q,QN)  S=INV1(SN) DS=OR2(D,S) DRS=AND2(DS,RN) DI=MUX21(DRS,SI,SE) Q=DFF(DI,CK) QN=INV1(Q) ;

TBUF_X{1,2,4,8,16} input(A,EN)   output(Z)  Z=BUF1(A)    ;
TINV_X1            input(I,EN)   output(ZN) ZN=INV1(I)   ;
TLAT_X1            input(D,G,OE) output(Q)  Q=LATCH(D,G) ;

DLH_X{1,2} input(D,G) output(Q)  Q=LATCH(D,G)            ;
DLL_X{1,2} input(D,GN) output(Q) G=INV1(GN) Q=LATCH(D,G) ;
"""


NANGATE = TechLib(_nangate_common + r"""
INV_X{1,2,4,8,16,32}  input(I) output(ZN) ZN=INV1(I) ;

AND2_X{1,2,4}  input(A1,A2)       output(Z)  Z=AND2(A1,A2)        ;
AND3_X{1,2,4}  input(A1,A2,A3)    output(Z)  Z=AND3(A1,A2,A3)     ;
AND4_X{1,2,4}  input(A1,A2,A3,A4) output(Z)  Z=AND4(A1,A2,A3,A4)  ;
OR2_X{1,2,4}   input(A1,A2)       output(Z)  Z=OR2(A1,A2)         ;
OR3_X{1,2,4}   input(A1,A2,A3)    output(Z)  Z=OR3(A1,A2,A3)      ;
OR4_X{1,2,4}   input(A1,A2,A3,A4) output(Z)  Z=OR4(A1,A2,A3,A4)   ;
XOR2_X{1,2}    input(A1,A2)       output(Z)  Z=XOR2(A1,A2)        ;
XNOR2_X{1,2}   input(A1,A2)       output(ZN) ZN=XNOR2(A1,A2)      ;
""")
"""An newer NANGATE-variant that uses 'Z' as output pin names for AND and OR gates.
"""


NANGATE_ZN = TechLib(_nangate_common + r"""
INV_X{1,2,4,8,16,32}  input(A) output(ZN) ZN=INV1(A) ;

AND2_X{1,2,4}  input(A1,A2)       output(ZN) ZN=AND2(A1,A2)        ;
AND3_X{1,2,4}  input(A1,A2,A3)    output(ZN) ZN=AND3(A1,A2,A3)     ;
AND4_X{1,2,4}  input(A1,A2,A3,A4) output(ZN) ZN=AND4(A1,A2,A3,A4)  ;
OR2_X{1,2,4}   input(A1,A2)       output(ZN) ZN=OR2(A1,A2)         ;
OR3_X{1,2,4}   input(A1,A2,A3)    output(ZN) ZN=OR3(A1,A2,A3)      ;
OR4_X{1,2,4}   input(A1,A2,A3,A4) output(ZN) ZN=OR4(A1,A2,A3,A4)   ;
XOR2_X{1,2}    input(A,B)         output(Z)  Z=XOR2(A,B)           ;
XNOR2_X{1,2}   input(A,B)         output(ZN) ZN=XNOR2(A,B)         ;
""")
"""An older NANGATE-variant that uses 'ZN' as output pin names for AND and OR gates.
"""


SAED32 = TechLib(r"""
NBUFFX{2,4,8,16,32}$ input(A) output(Y) Y=BUF1(A) ;
AOBUFX{1,2,4}$       input(A) output(Y) Y=BUF1(A) ;
DELLN{1,2,3}X2$      input(A) output(Y) Y=BUF1(A) ;

INVX{0,1,2,4,8,16,32}$ input(A) output(Y) Y=INV1(A) ;
AOINVX{1,2,4}$         input(A) output(Y) Y=INV1(A) ;
IBUFFX{2,4,8,16,32}$   input(A) output(Y) Y=INV1(A) ;

TIEH$ output(Y) Y=__const1__() ;
TIEL$ output(Y) Y=__const0__() ;

HEAD2X{2,4,8,16,32}$ input(SLEEP) output(SLEEPOUT) SLEEPOUT=BUF1(SLEEP) ;
HEADX{2,4,8,16,32}$  input(SLEEP) ;

FOOT2X{2,4,8,16,32}$ input(SLEEP) output(SLEEPOUT) SLEEPOUT=BUF1(SLEEP) ;
FOOTX{2,4,8,16,32}$  input(SLEEP) ;

ANTENNA$ input(INP)   ;
CLOAD1$  input(A)     ;
DCAP$                 ;
DHFILLH2$             ;
DHFILLHL2$            ;
DHFILLHLHLS11$        ;
SHFILL{1,2,3,64,128}$ ;

AND2X{1,2,4}$    input(A1,A2)       output(Y) Y=AND2(A1,A2)        ;
AND3X{1,2,4}$    input(A1,A2,A3)    output(Y) Y=AND3(A1,A2,A3)     ;
AND4X{1,2,4}$    input(A1,A2,A3,A4) output(Y) Y=AND4(A1,A2,A3,A4)  ;
OR2X{1,2,4}$     input(A1,A2)       output(Y) Y=OR2(A1,A2)         ;
OR3X{1,2,4}$     input(A1,A2,A3)    output(Y) Y=OR3(A1,A2,A3)      ;
OR4X{1,2,4}$     input(A1,A2,A3,A4) output(Y) Y=OR4(A1,A2,A3,A4)   ;
XOR2X{1,2}$      input(A1,A2)       output(Y) Y=XOR2(A1,A2)        ;
XOR3X{1,2}$      input(A1,A2,A3)    output(Y) Y=XOR3(A1,A2,A3)     ;
NAND2X{0,1,2,4}$ input(A1,A2)       output(Y) Y=NAND2(A1,A2)       ;
NAND3X{0,1,2,4}$ input(A1,A2,A3)    output(Y) Y=NAND3(A1,A2,A3)    ;
NAND4X{0,1}$     input(A1,A2,A3,A4) output(Y) Y=NAND4(A1,A2,A3,A4) ;
NOR2X{0,1,2,4}$  input(A1,A2)       output(Y) Y=NOR2(A1,A2)        ;
NOR3X{0,1,2,4}$  input(A1,A2,A3)    output(Y) Y=NOR3(A1,A2,A3)     ;
NOR4X{0,1}$      input(A1,A2,A3,A4) output(Y) Y=NOR4(A1,A2,A3,A4)  ;
XNOR2X{1,2}$     input(A1,A2)       output(Y) Y=XNOR2(A1,A2)       ;
XNOR3X{1,2}$     input(A1,A2,A3)    output(Y) Y=XNOR3(A1,A2,A3)    ;

ISOLAND{,AO}X{1,2,4,8}$ input(ISO,D) output(Q) ISOB=NOT1(ISO) Q=AND2(ISOB,D) ;
ISOLOR{,AO}X{1,2,4,8}$  input(ISO,D) output(Q) Q=OR2(ISO,D)  ;

AO21X{1,2}$  input(A1,A2,A3) output(Y) Y=AO21(A1,A2,A3)  ;
OA21X{1,2}$  input(A1,A2,A3) output(Y) Y=OA21(A1,A2,A3)  ;
AOI21X{1,2}$ input(A1,A2,A3) output(Y) Y=AOI21(A1,A2,A3) ;
OAI21X{1,2}$ input(A1,A2,A3) output(Y) Y=OAI21(A1,A2,A3) ;

AO22X{1,2}$  input(A1,A2,A3,A4) output(Y) Y=AO22(A1,A2,A3,A4)  ;
OA22X{1,2}$  input(A1,A2,A3,A4) output(Y) Y=OA22(A1,A2,A3,A4)  ;
AOI22X{1,2}$ input(A1,A2,A3,A4) output(Y) Y=AOI22(A1,A2,A3,A4) ;
OAI22X{1,2}$ input(A1,A2,A3,A4) output(Y) Y=OAI22(A1,A2,A3,A4) ;

MUX21X{1,2}$ input(A1,A2,S0) output(Y) Y=MUX21(A1,A2,S0) ;

AO221X{1,2}$  input(A1,A2,A3,A4,A5) output(Y) A=AO22(A1,A2,A3,A4) Y=OR2(A5,A)   ;
OA221X{1,2}$  input(A1,A2,A3,A4,A5) output(Y) A=OA22(A1,A2,A3,A4) Y=AND2(A5,A)  ;
AOI221X{1,2}$ input(A1,A2,A3,A4,A5) output(Y) A=AO22(A1,A2,A3,A4) Y=NOR2(A5,A)  ;
OAI221X{1,2}$ input(A1,A2,A3,A4,A5) output(Y) A=OA22(A1,A2,A3,A4) Y=NAND2(A5,A) ;

AO222X{1,2}$ input(A1,A2,A3,A4,A5,A6)  output(Y) A=AO22(A1,A2,A3,A4) Y=AO21(A5,A6,A)  ;
OA222X{1,2}$ input(A1,A2,A3,A4,A5,A6)  output(Y) A=OA22(A1,A2,A3,A4) Y=OA21(A5,A6,A)  ;
AOI222X{1,2}$ input(A1,A2,A3,A4,A5,A6) output(Y) A=AO22(A1,A2,A3,A4) Y=AOI21(A5,A6,A) ;
OAI222X{1,2}$ input(A1,A2,A3,A4,A5,A6) output(Y) A=OA22(A1,A2,A3,A4) Y=OAI21(A5,A6,A) ;

MUX41X{1,2}$ input(A1,A2,A3,A4,S0,S1) output(Y) A=MUX21(A1,A2,S0) B=MUX21(A3,A4,S0) Y=MUX21(A,B,S1) ;

DEC24X{1,2}$ input(A0,A1) output(Y0,Y1,Y2,Y3) A0B=INV1(A0) A1B=INV1(A1) Y0=NOR2(A0,A1) Y1=AND(A0,A1B) Y2=AND(A0B,A1) Y3=AND(A0,A1) ;
FADDX{1,2}$ input(A,B,CI) output(S,CO) AB=XOR2(A,B) CO=XOR2(AB,CI) S=AO22(AB,CI,A,B) ;
HADDX{1,2}$ input(A0,B0) output(SO,C1) C1=XOR2(A0,B0) SO=AND2(A0,B0) ;

{,AO}DFFARX{1,2}$ input(D,CLK,RSTB)      output(Q,QN) DR=AND2(D,RSTB) Q=DFF(DR,CLK) QN=INV1(Q) ;
DFFASRX{1,2}$     input(D,CLK,RSTB,SETB) output(Q,QN) DR=AND2(D,RSTB) SET=INV1(SETB) DRS=OR2(DR,SET) Q=DFF(DRS,CLK) QN=INV1(Q) ;
DFFASX{1,2}$      input(D,CLK,SETB)      output(Q,QN) SET=INV1(SETB) DS=OR2(D,SET) Q=DFF(DS,CLK) QN=INV1(Q) ;
DFFSSRX{1,2}$     input(CLK,D,RSTB,SETB) output(Q,QN) DR=AND2(D,RSTB) SET=INV1(SETB) DRS=OR2(DR,SET) Q=DFF(DRS,CLK) QN=INV1(Q) ;
DFFX{1,2}$        input(D,CLK)           output(Q,QN) Q=DFF(D,CLK) QN=INV1(Q) ;

SDFFARX{1,2}$   input(D,CLK,RSTB,SE,SI)      output(Q,QN) DR=AND2(D,RSTB) DI=MUX21(DR,SI,SE) Q=DFF(DI,CLK) QN=INV1(Q) ;
SDFFASRSX{1,2}$ input(D,CLK,RSTB,SETB,SE,SI) output(Q,QN,SO) DR=AND2(D,RSTB) SET=INV1(SETB) DRS=OR2(DR,SET) DI=MUX21(DRS,SI,SE) Q=DFF(DI,CLK) QN=INV1(Q) SO=BUF1(Q) ;
SDFFASRX{1,2}$  input(D,CLK,RSTB,SETB,SE,SI) output(Q,QN) DR=AND2(D,RSTB) SET=INV1(SETB) DRS=OR2(DR,SET) DI=MUX21(DRS,SI,SE) Q=DFF(DI,CLK) QN=INV1(Q) ;
SDFFASX{1,2}$   input(D,CLK,SETB,SE,SI)      output(Q,QN) SET=INV1(SETB) DS=OR2(D,SET) DI=MUX21(DS,SI,SE) Q=DFF(DI,CLK) QN=INV1(Q) ;
SDFFSSRX{1,2}$  input(CLK,D,RSTB,SETB,SI,SE) output(Q,QN) DR=AND2(D,RSTB) SET=INV1(SETB) DRS=OR2(DR,SET) DI=MUX21(DRS,SI,SE) Q=DFF(DI,CLK) QN=INV1(Q) ;
SDFFX{1,2}$     input(D,CLK,SE,SI)           output(Q,QN) DI=MUX21(D,SI,SE) Q=DFF(DI,CLK) QN=INV1(Q) ;

LATCHX{1,2}$ input(D,CLK) output(Q,QN) Q=LATCH(D,CLK) QN=INV1(Q) ;
""".replace('$','_RVT'))
"""The SAED 32nm educational technology library.
It defines all cells except: negative-edge flip-flops, tri-state, latches, clock gating, level shifters
"""


SAED90 = TechLib(r"""
NBUFFX{2,4,8,16,32}$ input(INP) output(Z) Z=BUF1(INP) ;
AOBUFX{1,2,4}$       input(INP) output(Z) Z=BUF1(INP) ;
DELLN{1,2,3}X2$      input(INP) output(Z)Z=BUF1(INP) ;

INVX{0,1,2,4,8,16,32}$ input(INP) output(ZN) ZN=INV1(INP) ;
AOINVX{1,2,4}$         input(INP) output(ZN) ZN=INV1(INP) ;
IBUFFX{2,4,8,16,32}$   input(INP) output(ZN) ZN=INV1(INP) ;

TIEH$ output(Z)   Z=__const1__() ;
TIEL$ output(ZN) ZN=__const0__() ;

HEAD2X{2,4,8,16,32}$ input(SLEEP) output(SLEEPOUT) SLEEPOUT=BUF1(SLEEP) ;
HEADX{2,4,8,16,32}$  input(SLEEP) ;

ANTENNA$ input(INP)   ;
CLOAD1$  input(INP)   ;
DCAP$                 ;
DHFILL{HLH,LHL}2      ;
DHFILLHLHLS11$        ;
SHFILL{1,2,3,64,128}$ ;

AND2X{1,2,4}$    input(IN1,IN2)         output(Q)   Q=AND2(IN1,IN2)          ;
AND3X{1,2,4}$    input(IN1,IN2,IN3)     output(Q)   Q=AND3(IN1,IN2,IN3)      ;
AND4X{1,2,4}$    input(IN1,IN2,IN3,IN4) output(Q)   Q=AND4(IN1,IN2,IN3,IN4)  ;
OR2X{1,2,4}$     input(IN1,IN2)         output(Q)   Q=OR2(IN1,IN2)           ;
OR3X{1,2,4}$     input(IN1,IN2,IN3)     output(Q)   Q=OR3(IN1,IN2,IN3)       ;
OR4X{1,2,4}$     input(IN1,IN2,IN3,IN4) output(Q)   Q=OR4(IN1,IN2,IN3,IN4)   ;
XOR2X{1,2}$      input(IN1,IN2)         output(Q)   Q=XOR2(IN1,IN2)          ;
XOR3X{1,2}$      input(IN1,IN2,IN3)     output(Q)   Q=XOR3(IN1,IN2,IN3)      ;
NAND2X{0,1,2,4}$ input(IN1,IN2)         output(QN) QN=NAND2(IN1,IN2)         ;
NAND3X{0,1,2,4}$ input(IN1,IN2,IN3)     output(QN) QN=NAND3(IN1,IN2,IN3)     ;
NAND4X{0,1}$     input(IN1,IN2,IN3,IN4) output(QN) QN=NAND4(IN1,IN2,IN3,IN4) ;
NOR2X{0,1,2,4}$  input(IN1,IN2)         output(QN) QN=NOR2(IN1,IN2)          ;
NOR3X{0,1,2,4}$  input(IN1,IN2,IN3)     output(QN) QN=NOR3(IN1,IN2,IN3)      ;
NOR4X{0,1}$      input(IN1,IN2,IN3,IN4) output(QN) QN=NOR4(IN1,IN2,IN3,IN4)  ;
XNOR2X{1,2}$     input(IN1,IN2)         output(Q)   Q=XNOR2(IN1,IN2)         ;
XNOR3X{1,2}$     input(IN1,IN2,IN3)     output(Q)   Q=XNOR3(IN1,IN2,IN3)     ;

ISOLAND{,AO}X{1,2,4,8}$ input(ISO,D) output(Q) ISOB=NOT1(ISO) Q=AND2(ISOB,D) ;
ISOLOR{,AO}X{1,2,4,8}$  input(ISO,D) output(Q) Q=OR2(ISO,D)  ;

AO21X{1,2}$  input(IN1,IN2,IN3) output(Q)   Q=AO21(IN1,IN2,IN3)  ;
OA21X{1,2}$  input(IN1,IN2,IN3) output(Q)   Q=OA21(IN1,IN2,IN3)  ;
AOI21X{1,2}$ input(IN1,IN2,IN3) output(QN) QN=AOI21(IN1,IN2,IN3) ;
OAI21X{1,2}$ input(IN1,IN2,IN3) output(QN) QN=OAI21(IN1,IN2,IN3) ;

AO22X{1,2}$  input(IN1,IN2,IN3,IN4) output(Q)   Q=AO22(IN1,IN2,IN3,IN4)  ;
OA22X{1,2}$  input(IN1,IN2,IN3,IN4) output(Q)   Q=OA22(IN1,IN2,IN3,IN4)  ;
AOI22X{1,2}$ input(IN1,IN2,IN3,IN4) output(QN) QN=AOI22(IN1,IN2,IN3,IN4) ;
OAI22X{1,2}$ input(IN1,IN2,IN3,IN4) output(QN) QN=OAI22(IN1,IN2,IN3,IN4) ;

MUX21X{1,2}$ input(IN1,IN2,S) output(Q) Q=MUX21(IN1,IN2,S) ;

AO221X{1,2}$  input(IN1,IN2,IN3,IN4,IN5) output(Q)  A=AO22(IN1,IN2,IN3,IN4)  Q=OR2(IN5,A)   ;
OA221X{1,2}$  input(IN1,IN2,IN3,IN4,IN5) output(Q)  A=OA22(IN1,IN2,IN3,IN4)  Q=AND2(IN5,A)  ;
AOI221X{1,2}$ input(IN1,IN2,IN3,IN4,IN5) output(QN) A=AO22(IN1,IN2,IN3,IN4) QN=NOR2(IN5,A)  ;
OAI221X{1,2}$ input(IN1,IN2,IN3,IN4,IN5) output(QN) A=OA22(IN1,IN2,IN3,IN4) QN=NAND2(IN5,A) ;

AO222X{1,2}$ input(IN1,IN2,IN3,IN4,IN5,IN6)  output(Q)  A=AO22(IN1,IN2,IN3,IN4)  Q=AO21(IN5,IN6,A)  ;
OA222X{1,2}$ input(IN1,IN2,IN3,IN4,IN5,IN6)  output(Q)  A=OA22(IN1,IN2,IN3,IN4)  Q=OA21(IN5,IN6,A)  ;
AOI222X{1,2}$ input(IN1,IN2,IN3,IN4,IN5,IN6) output(QN) A=AO22(IN1,IN2,IN3,IN4) QN=AOI21(IN5,IN6,A) ;
OAI222X{1,2}$ input(IN1,IN2,IN3,IN4,IN5,IN6) output(QN) A=OA22(IN1,IN2,IN3,IN4) QN=OAI21(IN5,IN6,A) ;

MUX41X{1,2}$ input(IN1,IN2,IN3,IN4,S0,S1) output(Q) A=MUX21(IN1,IN2,S0) B=MUX21(IN3,IN4,S0) Q=MUX21(A,B,S1) ;

DEC24X{1,2}$ input(IN1,IN2) output(Q0,Q1,Q2,Q3) IN1B=INV1(IN1) IN2B=INV1(IN2) Q0=NOR2(IN1,IN2) Q1=AND(IN1,IN2B) Q2=AND(IN1B,IN2) Q3=AND(IN1,IN2) ;
FADDX{1,2}$ input(A,B,CI) output(S,CO) AB=XOR2(A,B) CO=XOR2(AB,CI) S=AO22(AB,CI,A,B) ;
HADDX{1,2}$ input(A0,B0) output(SO,C1) C1=XOR2(A0,B0) SO=AND2(A0,B0) ;

{,AO}DFFARX{1,2}$ input(D,CLK,RSTB)      output(Q,QN) DR=AND2(D,RSTB) Q=DFF(DR,CLK) QN=INV1(Q) ;
DFFASRX{1,2}$     input(D,CLK,RSTB,SETB) output(Q,QN) DR=AND2(D,RSTB) SET=INV1(SETB) DRS=OR2(DR,SET) Q=DFF(DRS,CLK) QN=INV1(Q) ;
DFFASX{1,2}$      input(D,CLK,SETB)      output(Q,QN) SET=INV1(SETB) DS=OR2(D,SET) Q=DFF(DS,CLK) QN=INV1(Q) ;
DFFSSRX{1,2}$     input(CLK,D,RSTB,SETB) output(Q,QN) DR=AND2(D,RSTB) SET=INV1(SETB) DRS=OR2(DR,SET) Q=DFF(DRS,CLK) QN=INV1(Q) ;
DFFX{1,2}$        input(D,CLK)           output(Q,QN) Q=DFF(D,CLK) QN=INV1(Q) ;

SDFFARX{1,2}$   input(D,CLK,RSTB,SE,SI)      output(Q,QN) DR=AND2(D,RSTB) DI=MUX21(DR,SI,SE) Q=DFF(DI,CLK) QN=INV1(Q) ;
SDFFASRSX{1,2}$ input(D,CLK,RSTB,SETB,SE,SI) output(Q,QN,S0) DR=AND2(D,RSTB) SET=INV1(SETB) DRS=OR2(DR,SET) DI=MUX21(DRS,SI,SE) Q=DFF(DI,CLK) QN=INV1(Q) S0=BUF1(Q) ;
SDFFASRX{1,2}$  input(D,CLK,RSTB,SETB,SE,SI) output(Q,QN) DR=AND2(D,RSTB) SET=INV1(SETB) DRS=OR2(DR,SET) DI=MUX21(DRS,SI,SE) Q=DFF(DI,CLK) QN=INV1(Q) ;
SDFFASX{1,2}$   input(D,CLK,SETB,SE,SI)      output(Q,QN) SET=INV1(SETB) DS=OR2(D,SET) DI=MUX21(DS,SI,SE) Q=DFF(DI,CLK) QN=INV1(Q) ;
SDFFSSRX{1,2}$  input(CLK,D,RSTB,SETB,SI,SE) output(Q,QN) DR=AND2(D,RSTB) SET=INV1(SETB) DRS=OR2(DR,SET) DI=MUX21(DRS,SI,SE) Q=DFF(DI,CLK) QN=INV1(Q) ;
SDFFX{1,2}$     input(D,CLK,SE,SI)           output(Q,QN) DI=MUX21(D,SI,SE) Q=DFF(DI,CLK) QN=INV1(Q) ;

LATCHX{1,2}$ input(D,CLK) output(Q,QN) Q=LATCH(D,CLK) QN=INV1(Q) ;
""".replace('$','{,_LVT,_HVT}'))
"""The SAED 90nm educational technology library.
It defines all cells except: negative-edge flip-flops, tri-state, latches, clock gating, level shifters
"""
