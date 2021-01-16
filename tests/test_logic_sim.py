from kyupy.logic_sim import LogicSim
from kyupy import bench
from kyupy.logic import MVArray, BPArray


def test_2v():
    c = bench.parse('input(x, y) output(a, o, n) a=and(x,y) o=or(x,y) n=not(x)')
    s = LogicSim(c, 4, m=2)
    assert len(s.interface) == 5
    mva = MVArray(['00000', '01000', '10000', '11000'], m=2)
    bpa = BPArray(mva)
    s.assign(bpa)
    s.propagate()
    s.capture(bpa)
    mva = MVArray(bpa)
    assert mva[0] == '00001'
    assert mva[1] == '01011'
    assert mva[2] == '10010'
    assert mva[3] == '11110'


def test_4v():
    c = bench.parse('input(x, y) output(a, o, n) a=and(x,y) o=or(x,y) n=not(x)')
    s = LogicSim(c, 16, m=4)
    assert len(s.interface) == 5
    mva = MVArray(['00000', '01000', '0-000', '0X000',
                   '10000', '11000', '1-000', '1X000',
                   '-0000', '-1000', '--000', '-X000',
                   'X0000', 'X1000', 'X-000', 'XX000'], m=4)
    bpa = BPArray(mva)
    s.assign(bpa)
    s.propagate()
    s.capture(bpa)
    mva = MVArray(bpa)
    assert mva[0] == '00001'
    assert mva[1] == '01011'
    assert mva[2] == '0-0X1'
    assert mva[3] == '0X0X1'
    assert mva[4] == '10010'
    assert mva[5] == '11110'
    assert mva[6] == '1-X10'
    assert mva[7] == '1XX10'
    assert mva[8] == '-00XX'
    assert mva[9] == '-1X1X'
    assert mva[10] == '--XXX'
    assert mva[11] == '-XXXX'
    assert mva[12] == 'X00XX'
    assert mva[13] == 'X1X1X'
    assert mva[14] == 'X-XXX'
    assert mva[15] == 'XXXXX'


def test_8v():
    c = bench.parse('input(x, y) output(a, o, n, xo) a=and(x,y) o=or(x,y) n=not(x) xo=xor(x,y)')
    s = LogicSim(c, 64, m=8)
    assert len(s.interface) == 6
    mva = MVArray(['000010', '010111', '0-0X1X', '0X0X1X', '0R0R1R', '0F0F1F', '0P0P1P', '0N0N1N',
                   '100101', '111100', '1-X10X', '1XX10X', '1RR10F', '1FF10R', '1PP10N', '1NN10P',
                   '-00XXX', '-1X1XX', '--XXXX', '-XXXXX', '-RXXXX', '-FXXXX', '-PXXXX', '-NXXXX',
                   'X00XXX', 'X1X1XX', 'X-XXXX', 'XXXXXX', 'XRXXXX', 'XFXXXX', 'XPXXXX', 'XNXXXX',
                   'R00RFR', 'R1R1FF', 'R-XXFX', 'RXXXFX', 'RRRRFP', 'RFPNFN', 'RPPRFR', 'RNRNFF',
                   'F00FRF', 'F1F1RR', 'F-XXRX', 'FXXXRX', 'FRPNRN', 'FFFFRP', 'FPPFRF', 'FNFNRR',
                   'P00PNP', 'P1P1NN', 'P-XXNX', 'PXXXNX', 'PRPRNR', 'PFPFNF', 'PPPPNP', 'PNPNNN',
                   'N00NPN', 'N1N1PP', 'N-XXPX', 'NXXXPX', 'NRRNPF', 'NFFNPR', 'NPPNPN', 'NNNNPP'], m=8)
    bpa = BPArray(mva)
    s.assign(bpa)
    s.propagate()
    resp_bp = BPArray(bpa)
    s.capture(resp_bp)
    resp = MVArray(resp_bp)

    for i in range(64):
        assert resp[i] == mva[i]


def test_b01(mydir):
    c = bench.load(mydir / 'b01.bench')

    # 2-valued
    s = LogicSim(c, 8, m=2)
    assert len(s.interface) == 9
    mva = MVArray((len(s.interface), 8), m=2)
    # mva.randomize()
    bpa = BPArray(mva)
    s.assign(bpa)
    s.propagate()
    s.capture(bpa)

    # 8-valued
    s = LogicSim(c, 8, m=8)
    mva = MVArray((len(s.interface), 8), m=8)
    # mva.randomize()
    bpa = BPArray(mva)
    s.assign(bpa)
    s.propagate()
    s.capture(bpa)
