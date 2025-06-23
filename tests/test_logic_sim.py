import numpy as np

from kyupy.logic_sim import LogicSim, LogicSim6V
from kyupy import bench, logic, sim
from kyupy.logic import mvarray, bparray, bp_to_mv, mv_to_bp

def assert_equal_shape_and_contents(actual, desired):
    desired = np.array(desired, dtype=np.uint8)
    assert actual.shape == desired.shape
    np.testing.assert_allclose(actual, desired)

def test_2v():
    c = bench.parse(f'''
        input(i3, i2, i1, i0)
        output({",".join([f"o{i:02d}" for i in range(33)])})
        o00=BUF1(i0)
        o01=INV1(i0)
        o02=AND2(i0,i1)
        o03=AND3(i0,i1,i2)
        o04=AND4(i0,i1,i2,i3)
        o05=NAND2(i0,i1)
        o06=NAND3(i0,i1,i2)
        o07=NAND4(i0,i1,i2,i3)
        o08=OR2(i0,i1)
        o09=OR3(i0,i1,i2)
        o10=OR4(i0,i1,i2,i3)
        o11=NOR2(i0,i1)
        o12=NOR3(i0,i1,i2)
        o13=NOR4(i0,i1,i2,i3)
        o14=XOR2(i0,i1)
        o15=XOR3(i0,i1,i2)
        o16=XOR4(i0,i1,i2,i3)
        o17=XNOR2(i0,i1)
        o18=XNOR3(i0,i1,i2)
        o19=XNOR4(i0,i1,i2,i3)
        o20=AO21(i0,i1,i2)
        o21=OA21(i0,i1,i2)
        o22=AO22(i0,i1,i2,i3)
        o23=OA22(i0,i1,i2,i3)
        o24=AOI21(i0,i1,i2)
        o25=OAI21(i0,i1,i2)
        o26=AOI22(i0,i1,i2,i3)
        o27=OAI22(i0,i1,i2,i3)
        o28=AO211(i0,i1,i2,i3)
        o29=OA211(i0,i1,i2,i3)
        o30=AOI211(i0,i1,i2,i3)
        o31=OAI211(i0,i1,i2,i3)
        o32=MUX21(i0,i1,i2)
    ''')
    s = LogicSim(c, 16, m=2)
    bpa = logic.bparray([f'{i:04b}'+('-'*(s.s_len-4)) for i in range(16)])
    s.s[0] = bpa
    s.s_to_c()
    s.c_prop()
    s.c_to_s()
    mva = logic.bp_to_mv(s.s[1])
    for res, exp in zip(logic.packbits(mva[4:], dtype=np.uint32), [
            sim.BUF1, sim.INV1,
            sim.AND2, sim.AND3, sim.AND4,
            sim.NAND2, sim.NAND3, sim.NAND4,
            sim.OR2, sim.OR3, sim.OR4,
            sim.NOR2, sim.NOR3, sim.NOR4,
            sim.XOR2, sim.XOR3, sim.XOR4,
            sim.XNOR2, sim.XNOR3, sim.XNOR4,
            sim.AO21, sim.OA21,
            sim.AO22, sim.OA22,
            sim.AOI21, sim.OAI21,
            sim.AOI22, sim.OAI22,
            sim.AO211, sim.OA211,
            sim.AOI211, sim.OAI211,
            sim.MUX21
        ]):
        assert res == exp, f'Mismatch for SimPrim {sim.names[exp]} res={bin(res)} exp={bin(exp)}'


def test_4v():
    c = bench.parse('input(x, y) output(a, o, n) a=and(x,y) o=or(x,y) n=not(x)')
    s = LogicSim(c, 16, m=8)  # FIXME: m=4
    assert s.s_len == 5
    bpa = bparray(
        '00---', '01---', '0----', '0X---',
        '10---', '11---', '1----', '1X---',
        '-0---', '-1---', '-----', '-X---',
        'X0---', 'X1---', 'X----', 'XX---')
    s.s[0] = bpa
    s.s_to_c()
    s.c_prop()
    s.c_to_s()
    mva = bp_to_mv(s.s[1])
    assert_equal_shape_and_contents(mva, mvarray(
        '--001', '--011', '--0X1', '--0X1',
        '--010', '--110', '--X10', '--X10',
        '--0XX', '--X1X', '--XXX', '--XXX',
        '--0XX', '--X1X', '--XXX', '--XXX'))


def test_6v():
    c = bench.parse('input(x, y) output(a, o, n, xo, no) a=AND2(x,y) o=OR2(x,y) n=INV1(x) xo=XOR2(x,y) no=NOR2(x,y)')
    s = LogicSim6V(c, 36)
    assert s.s_len == 7
    mva = mvarray(
        '0000101', '0101110', '0R0R1RF', '0F0F1FR', '0P0P1PN', '0N0N1NP',
        '1001010', '1111000', '1RR10F0', '1FF10R0', '1PP10N0', '1NN10P0',
        'R00RFRF', 'R1R1FF0', 'RRRRFPF', 'RFPNFNP', 'RPPRFRF', 'RNRNFFP',
        'F00FRFR', 'F1F1RR0', 'FRPNRNP', 'FFFFRPR', 'FPPFRFR', 'FNFNRRP',
        'P00PNPN', 'P1P1NN0', 'PRPRNRF', 'PFPFNFR', 'PPPPNPN', 'PNPNNNP',
        'N00NPNP', 'N1N1PP0', 'NRRNPFP', 'NFFNPRP', 'NPPNPNP', 'NNNNPPP')
    tests = np.copy(mva)
    tests[2:] = logic.ZERO
    s.s[0] = tests
    s.s_to_c()
    s.c_prop()
    s.c_to_s()
    resp = s.s[1].copy()

    exp_resp = np.copy(mva)
    exp_resp[:2] = logic.ZERO
    np.testing.assert_allclose(resp, exp_resp)


def test_8v():
    c = bench.parse('input(x, y) output(a, o, n, xo) a=and(x,y) o=or(x,y) n=not(x) xo=xor(x,y)')
    s = LogicSim(c, 64, m=8)
    assert s.s_len == 6
    mva = mvarray(
        '000010', '010111', '0-0X1X', '0X0X1X', '0R0R1R', '0F0F1F', '0P0P1P', '0N0N1N',
        '100101', '111100', '1-X10X', '1XX10X', '1RR10F', '1FF10R', '1PP10N', '1NN10P',
        '-00XXX', '-1X1XX', '--XXXX', '-XXXXX', '-RXXXX', '-FXXXX', '-PXXXX', '-NXXXX',
        'X00XXX', 'X1X1XX', 'X-XXXX', 'XXXXXX', 'XRXXXX', 'XFXXXX', 'XPXXXX', 'XNXXXX',
        'R00RFR', 'R1R1FF', 'R-XXFX', 'RXXXFX', 'RRRRFP', 'RFPNFN', 'RPPRFR', 'RNRNFF',
        'F00FRF', 'F1F1RR', 'F-XXRX', 'FXXXRX', 'FRPNRN', 'FFFFRP', 'FPPFRF', 'FNFNRR',
        'P00PNP', 'P1P1NN', 'P-XXNX', 'PXXXNX', 'PRPRNR', 'PFPFNF', 'PPPPNP', 'PNPNNN',
        'N00NPN', 'N1N1PP', 'N-XXPX', 'NXXXPX', 'NRRNPF', 'NFFNPR', 'NPPNPN', 'NNNNPP')
    tests = np.copy(mva)
    tests[2:] = logic.UNASSIGNED
    bpa = mv_to_bp(tests)
    s.s[0] = bpa
    s.s_to_c()
    s.c_prop()
    s.c_to_s()
    resp = bp_to_mv(s.s[1])

    exp_resp = np.copy(mva)
    exp_resp[:2] = logic.UNASSIGNED
    np.testing.assert_allclose(resp, exp_resp)


def test_loop():
    c = bench.parse('q=dff(d) d=not(q)')
    s = LogicSim(c, 4, m=8)
    assert s.s_len == 1
    mva = mvarray([['0'], ['1'], ['R'], ['F']])

    # TODO
    # s.assign(BPArray(mva))
    # s.propagate()
    # resp_bp = BPArray((len(s.interface), s.sims))
    # s.capture(resp_bp)
    # resp = MVArray(resp_bp)

    # assert resp[0] == '1'
    # assert resp[1] == '0'
    # assert resp[2] == 'F'
    # assert resp[3] == 'R'

    # resp_bp = s.cycle(resp_bp)
    # resp = MVArray(resp_bp)

    # assert resp[0] == '0'
    # assert resp[1] == '1'
    # assert resp[2] == 'R'
    # assert resp[3] == 'F'


def test_latch():
    c = bench.parse('input(d, t) output(q) q=latch(d, t)')
    s = LogicSim(c, 8, m=8)
    assert s.s_len == 4
    mva = mvarray('00-0', '00-1', '01-0', '01-1', '10-0', '10-1', '11-0', '11-1')
    exp = mvarray('0000', '0011', '0100', '0100', '1000', '1011', '1111', '1111')

    # TODO
    # resp = MVArray(s.cycle(BPArray(mva)))

    # for i in range(len(mva)):
    #     assert resp[i] == exp[i]


def test_b01(mydir):
    c = bench.load(mydir / 'b01.bench')

    # 8-valued
    s = LogicSim(c, 8, m=8)
    mva = np.zeros((s.s_len, 8), dtype=np.uint8)
    s.s[0] = mv_to_bp(mva)
    s.s_to_c()
    s.c_prop()
    s.c_to_s()
    bp_to_mv(s.s[1])


def sim_and_compare(c, test_resp, m=8):
    tests, resp = test_resp
    lsim = LogicSim(c, m=m, sims=tests.shape[1])
    lsim.s[0] = logic.mv_to_bp(tests)
    lsim.s_to_c()
    lsim.c_prop()
    lsim.c_to_s()
    resp_sim = logic.bp_to_mv(lsim.s[1])[:,:tests.shape[1]]
    idxs, pats = np.nonzero(((resp == logic.ONE) & (resp_sim != logic.ONE)) | ((resp == logic.ZERO) & (resp_sim != logic.ZERO)))
    for i, (idx, pat) in enumerate(zip(idxs, pats)):
        if i >= 10:
            print(f'...')
            break
        print(f'mismatch pattern:{pat} ppio:{idx} exp:{logic.mv_str(resp[idx,pat])} act:{logic.mv_str(resp_sim[idx,pat])}')
    assert len(idxs) == 0

def sim_and_compare_6v(c, test_resp):
    tests, resp = test_resp
    lsim = LogicSim6V(c, sims=tests.shape[1])
    lsim.s[0] = tests
    lsim.s_to_c()
    lsim.c_prop()
    lsim.c_to_s()
    resp_sim = lsim.s[1]
    idxs, pats = np.nonzero(((resp == logic.ONE) & (resp_sim != logic.ONE)) | ((resp == logic.ZERO) & (resp_sim != logic.ZERO)))
    for i, (idx, pat) in enumerate(zip(idxs, pats)):
        if i >= 10:
            print(f'...')
            break
        print(f'mismatch pattern:{pat} ppio:{idx} exp:{logic.mv_str(resp[idx,pat])} act:{logic.mv_str(resp_sim[idx,pat])}')
    assert len(idxs) == 0


def test_b15_2ig_sa_2v(b15_2ig_circuit_resolved, b15_2ig_sa_nf_test_resp):
    sim_and_compare(b15_2ig_circuit_resolved, b15_2ig_sa_nf_test_resp, m=2)


def test_b15_2ig_sa_4v(b15_2ig_circuit_resolved, b15_2ig_sa_nf_test_resp):
    sim_and_compare(b15_2ig_circuit_resolved, b15_2ig_sa_nf_test_resp, m=4)


def test_b15_2ig_sa_6v(b15_2ig_circuit_resolved, b15_2ig_sa_nf_test_resp):
    sim_and_compare_6v(b15_2ig_circuit_resolved, b15_2ig_sa_nf_test_resp)


def test_b15_2ig_sa_8v(b15_2ig_circuit_resolved, b15_2ig_sa_nf_test_resp):
    sim_and_compare(b15_2ig_circuit_resolved, b15_2ig_sa_nf_test_resp, m=8)


def test_b15_4ig_sa_2v(b15_4ig_circuit_resolved, b15_4ig_sa_rf_test_resp):
    sim_and_compare(b15_4ig_circuit_resolved, b15_4ig_sa_rf_test_resp, m=2)


def test_b15_4ig_sa_4v(b15_4ig_circuit_resolved, b15_4ig_sa_rf_test_resp):
    sim_and_compare(b15_4ig_circuit_resolved, b15_4ig_sa_rf_test_resp, m=4)


def test_b15_4ig_sa_8v(b15_4ig_circuit_resolved, b15_4ig_sa_rf_test_resp):
    sim_and_compare(b15_4ig_circuit_resolved, b15_4ig_sa_rf_test_resp, m=8)
