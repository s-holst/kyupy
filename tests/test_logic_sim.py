from kyupy.logic_sim import LogicSim
from kyupy import bench
from kyupy.packed_vectors import PackedVectors


def test_vd1():
    c = bench.parse('input(x, y) output(a, o, n) a=and(x,y) o=or(x,y) n=not(x)')
    s = LogicSim(c, 4)
    assert len(s.interface) == 5
    p = PackedVectors(4, len(s.interface))
    p[0] = '00000'
    p[1] = '01000'
    p[2] = '10000'
    p[3] = '11000'
    s.assign(p)
    s.propagate()
    s.capture(p)
    assert p[0] == '00001'
    assert p[1] == '01011'
    assert p[2] == '10010'
    assert p[3] == '11110'


def test_vd2():
    c = bench.parse('input(x, y) output(a, o, n) a=and(x,y) o=or(x,y) n=not(x)')
    s = LogicSim(c, 16, 2)
    assert len(s.interface) == 5
    p = PackedVectors(16, len(s.interface), 2)
    p[0] = '00000'
    p[1] = '01000'
    p[2] = '0-000'
    p[3] = '0X000'
    p[4] = '10000'
    p[5] = '11000'
    p[6] = '1-000'
    p[7] = '1X000'
    p[8] = '-0000'
    p[9] = '-1000'
    p[10] = '--000'
    p[11] = '-X000'
    p[12] = 'X0000'
    p[13] = 'X1000'
    p[14] = 'X-000'
    p[15] = 'XX000'
    s.assign(p)
    s.propagate()
    s.capture(p)
    assert p[0] == '00001'
    assert p[1] == '01011'
    assert p[2] == '0-0X1'
    assert p[3] == '0X0X1'
    assert p[4] == '10010'
    assert p[5] == '11110'
    assert p[6] == '1-X10'
    assert p[7] == '1XX10'
    assert p[8] == '-00XX'
    assert p[9] == '-1X1X'
    assert p[10] == '--XXX'
    assert p[11] == '-XXXX'
    assert p[12] == 'X00XX'
    assert p[13] == 'X1X1X'
    assert p[14] == 'X-XXX'
    assert p[15] == 'XXXXX'

    
def test_vd3():
    c = bench.parse('input(x, y) output(a, o, n, xo) a=and(x,y) o=or(x,y) n=not(x) xo=xor(x,y)')
    s = LogicSim(c, 64, 3)
    assert len(s.interface) == 6
    p = PackedVectors(64, len(s.interface), 3)
    p[0] = '000010'
    p[1] = '010111'
    p[2] = '0-0X1X'
    p[3] = '0X0X1X'
    p[4] = '0R0R1R'
    p[5] = '0F0F1F'
    p[6] = '0P0P1P'
    p[7] = '0N0N1N'
    p[8] = '100101'
    p[9] = '111100'
    p[10] = '1-X10X'
    p[11] = '1XX10X'
    p[12] = '1RR10F'
    p[13] = '1FF10R'
    p[14] = '1PP10N'
    p[15] = '1NN10P'
    p[16] = '-00XXX'
    p[17] = '-1X1XX'
    p[18] = '--XXXX'
    p[19] = '-XXXXX'
    p[20] = '-RXXXX'
    p[21] = '-FXXXX'
    p[22] = '-PXXXX'
    p[23] = '-NXXXX'
    p[24] = 'X00XXX'
    p[25] = 'X1X1XX'
    p[26] = 'X-XXXX'
    p[27] = 'XXXXXX'
    p[28] = 'XRXXXX'
    p[29] = 'XFXXXX'
    p[30] = 'XPXXXX'
    p[31] = 'XNXXXX'
    p[32] = 'R00RFR'
    p[33] = 'R1R1FF'
    p[34] = 'R-XXFX'
    p[35] = 'RXXXFX'
    p[36] = 'RRRRFP'
    p[37] = 'RFPNFN'
    p[38] = 'RPPRFR'
    p[39] = 'RNRNFF'
    p[40] = 'F00FRF'
    p[41] = 'F1F1RR'
    p[42] = 'F-XXRX'
    p[43] = 'FXXXRX'
    p[44] = 'FRPNRN'
    p[45] = 'FFFFRP'
    p[46] = 'FPPFRF'
    p[47] = 'FNFNRR'
    p[48] = 'P00PNP'
    p[49] = 'P1P1NN'
    p[50] = 'P-XXNX'
    p[51] = 'PXXXNX'
    p[52] = 'PRPRNR'
    p[53] = 'PFPFNF'
    p[54] = 'PPPPNP'
    p[55] = 'PNPNNN'
    p[56] = 'N00NPN'
    p[57] = 'N1N1PP'
    p[58] = 'N-XXPX'
    p[59] = 'NXXXPX'
    p[60] = 'NRRNPF'
    p[61] = 'NFFNPR'
    p[62] = 'NPPNPN'
    p[63] = 'NNNNPP'
    expect = p.copy()
    s.assign(p)
    s.propagate()
    s.capture(p)
    for i in range(64):
        assert p[i] == expect[i]
        

def test_b01(mydir):
    c = bench.parse(mydir / 'b01.bench')

    # 2-valued
    s = LogicSim(c, 8)
    assert len(s.interface) == 9
    t = PackedVectors(8, len(s.interface))
    t.randomize()
    s.assign(t)
    s.propagate()
    s.capture(t)

    # 8-valued
    s = LogicSim(c, 8, 3)
    t = PackedVectors(8, len(s.interface), 3)
    t.randomize()
    s.assign(t)
    s.propagate()
    s.capture(t)
