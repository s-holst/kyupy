import numpy as np

from kyupy.wave_sim import WaveSim, WaveSimCuda, wave_eval_cpu, TMIN, TMAX
from kyupy.logic_sim import LogicSim
from kyupy import logic, bench, sim
from kyupy.logic import mvarray

def test_xnor2_delays():
    op = (sim.XNOR2, 2, 0, 1, 3, 3, -1, 0, 0)
    #op = (0b0111, 4, 0, 1)
    c = np.full((4*16, 1), TMAX, dtype=np.float32)  # 4 waveforms of capacity 16
    c_locs = np.zeros((4,), dtype='int')
    c_caps = np.zeros((4,), dtype='int')
    ebuf = np.zeros((4, 1, 2), dtype=np.int32)

    for i in range(4): c_locs[i], c_caps[i] = i*16, 16  # 1:1 mapping
    delays = np.zeros((1, 4, 2, 2))
    delays[0, 0, 0, 0] = 0.031  # A rise -> Z rise
    delays[0, 0, 0, 1] = 0.027  # A rise -> Z fall
    delays[0, 0, 1, 0] = 0.033  # A fall -> Z rise
    delays[0, 0, 1, 1] = 0.037  # A fall -> Z fall
    delays[0, 1, 0, 0] = 0.032  # B rise -> Z rise
    delays[0, 1, 0, 1] = 0.030  # B rise -> Z fall
    delays[0, 1, 1, 0] = 0.038  # B fall -> Z rise
    delays[0, 1, 1, 1] = 0.036  # B fall -> Z fall

    simctl_int = np.asarray([0], dtype=np.int32)

    def wave_assert(inputs, output):
        for i, a in zip(inputs, c.reshape(-1,16)): a[:len(i)] = i
        wave_eval_cpu(op, c, c_locs, c_caps, ebuf, 0, delays, simctl_int, 0, 0)
        for i, v in enumerate(output): np.testing.assert_allclose(c.reshape(-1,16)[2,i], v)

    wave_assert([[TMIN,TMAX],[TMIN,TMAX]], [TMIN,TMAX])      # XNOR(1,1) => 1
    wave_assert([[TMAX,TMAX],[TMIN,TMAX]], [TMAX])           # XNOR(0,1) => 0
    # using Afall/Zfall for pulse length, bug: was using Arise/Zfall
    #wave_assert([[0.07, 0.10, TMAX], [0.0, TMAX]], [TMIN, 0.03, 0.101, 0.137, TMAX])
    wave_assert([[0.07, 0.10, TMAX], [0.0, TMAX]], [TMIN, 0.03, TMAX])
    wave_assert([[0.06, 0.10, TMAX], [0.0, TMAX]], [TMIN, 0.03, 0.091, 0.137, TMAX])

def test_nand_delays():
    op = (sim.NAND4, 4, 0, 1, 2, 3, -1, 0, 0)
    #op = (0b0111, 4, 0, 1)
    c = np.full((5*16, 1), TMAX, dtype=np.float32)  # 5 waveforms of capacity 16
    c_locs = np.zeros((5,), dtype='int')
    c_caps = np.zeros((5,), dtype='int')
    ebuf = np.zeros((4, 1, 2), dtype=np.int32)

    for i in range(5): c_locs[i], c_caps[i] = i*16, 16  # 1:1 mapping

    # SDF specifies IOPATH delays with respect to output polarity
    # SDF pulse rejection value is determined by IOPATH causing last transition and polarity of last transition
    delays = np.zeros((1, 5, 2, 2))
    delays[0, 0, 0, 0] = 0.1  # A rise -> Z rise
    delays[0, 0, 0, 1] = 0.2  # A rise -> Z fall
    delays[0, 0, 1, 0] = 0.1  # A fall -> Z rise
    delays[0, 0, 1, 1] = 0.2  # A fall -> Z fall
    delays[0, 1, :, 0] = 0.3  # as above for B -> Z
    delays[0, 1, :, 1] = 0.4
    delays[0, 2, :, 0] = 0.5  # as above for C -> Z
    delays[0, 2, :, 1] = 0.6
    delays[0, 3, :, 0] = 0.7  # as above for D -> Z
    delays[0, 3, :, 1] = 0.8

    simctl_int = np.asarray([0], dtype=np.int32)

    def wave_assert(inputs, output):
        for i, a in zip(inputs, c.reshape(-1,16)): a[:len(i)] = i
        wave_eval_cpu(op, c, c_locs, c_caps, ebuf, 0, delays, simctl_int, 0, 0)
        for i, v in enumerate(output): np.testing.assert_allclose(c.reshape(-1,16)[4,i], v)

    wave_assert([[TMAX,TMAX],[TMAX,TMAX],[TMIN,TMAX],[TMIN,TMAX]], [TMIN,TMAX]) # NAND(0,0,1,1) => 1
    wave_assert([[TMIN,TMAX],[TMAX,TMAX],[TMIN,TMAX],[TMIN,TMAX]], [TMIN,TMAX]) # NAND(1,0,1,1) => 1
    wave_assert([[TMIN,TMAX],[TMIN,TMAX],[TMIN,TMAX],[TMIN,TMAX]], [TMAX])      # NAND(1,1,1,1) => 0

    # Keep inputs C=1 and D=1.
    wave_assert([[1,TMAX],[2,TMAX]], [TMIN,2.4,TMAX])              # _/⎺⎺⎺ NAND __/⎺⎺ => ⎺⎺⎺\___ (B->Z fall delay)
    wave_assert([[TMIN,TMAX],[TMIN,2,TMAX]],  [2.3,TMAX])          # ⎺⎺⎺⎺⎺ NAND ⎺⎺\__ => ___/⎺⎺⎺ (B->Z rise delay)
    wave_assert([[TMIN,TMAX],[TMIN,2,2.35,TMAX]], [2.3,2.75,TMAX]) # ⎺⎺⎺⎺⎺ NAND ⎺\_/⎺ => __/⎺⎺\_ (pos pulse, .35@B -> .45@Z)
    wave_assert([[TMIN,TMAX],[TMIN,2,2.25,TMAX]], [TMAX])          # ⎺⎺⎺⎺⎺ NAND ⎺\_/⎺ => _______ (pos pulse, .25@B -> .35@Z, filtered)
    wave_assert([[TMIN,TMAX],[2,2.45,TMAX]], [TMIN,2.4,2.75,TMAX]) # ⎺⎺⎺⎺⎺ NAND _/⎺\_ => ⎺⎺\_/⎺⎺ (neg pulse, .45@B -> .35@Z)
    wave_assert([[TMIN,TMAX],[2,2.35,TMAX]], [TMIN,TMAX])          # ⎺⎺⎺⎺⎺ NAND _/⎺\_ => ⎺⎺⎺⎺⎺⎺⎺ (neg pulse, .35@B -> .25@Z, filtered)


def test_tiny_circuit():
    c = bench.parse('input(x, y) output(a, o, n) a=and(x,y) o=or(x,y) n=not(x)')
    delays = np.full((1, len(c.lines), 2, 2), 1.0)  # unit delay for all lines
    wsim = WaveSim(c, delays)
    assert wsim.s.shape[1] == 5

    # values for x
    wsim.s[:3,0,0] = 0, 10, 0
    wsim.s[:3,0,1] = 0, 20, 1
    wsim.s[:3,0,2] = 1, 30, 0
    wsim.s[:3,0,3] = 1, 40, 1

    # values for y
    wsim.s[:3,1,0] = 1, 50, 0
    wsim.s[:3,1,1] = 1, 60, 0
    wsim.s[:3,1,2] = 1, 70, 0
    wsim.s[:3,1,3] = 0, 80, 1

    wsim.s_to_c()

    x_c_loc = wsim.c_locs[wsim.ppi_offset+0] # check x waveforms
    np.testing.assert_allclose(wsim.c[x_c_loc:x_c_loc+3, 0], [TMAX, TMAX, TMAX])
    np.testing.assert_allclose(wsim.c[x_c_loc:x_c_loc+3, 1], [20, TMAX, TMAX])
    np.testing.assert_allclose(wsim.c[x_c_loc:x_c_loc+3, 2], [TMIN, 30, TMAX])
    np.testing.assert_allclose(wsim.c[x_c_loc:x_c_loc+3, 3], [TMIN, TMAX, TMAX])

    y_c_loc = wsim.c_locs[wsim.ppi_offset+1] # check y waveforms
    np.testing.assert_allclose(wsim.c[y_c_loc:y_c_loc+3, 0], [TMIN, 50, TMAX])
    np.testing.assert_allclose(wsim.c[y_c_loc:y_c_loc+3, 1], [TMIN, 60, TMAX])
    np.testing.assert_allclose(wsim.c[y_c_loc:y_c_loc+3, 2], [TMIN, 70, TMAX])
    np.testing.assert_allclose(wsim.c[y_c_loc:y_c_loc+3, 3], [80, TMAX, TMAX])

    wsim.c_prop()

    a_c_loc = wsim.c_locs[wsim.ppo_offset+2] # check a waveforms
    np.testing.assert_allclose(wsim.c[a_c_loc:a_c_loc+3, 0], [TMAX, TMAX, TMAX])
    np.testing.assert_allclose(wsim.c[a_c_loc:a_c_loc+3, 1], [21, 61, TMAX])
    np.testing.assert_allclose(wsim.c[a_c_loc:a_c_loc+3, 2], [TMIN, 31, TMAX])
    np.testing.assert_allclose(wsim.c[a_c_loc:a_c_loc+3, 3], [81, TMAX, TMAX])

    o_c_loc = wsim.c_locs[wsim.ppo_offset+3] # check o waveforms
    np.testing.assert_allclose(wsim.c[o_c_loc:o_c_loc+3, 0], [TMIN, 51, TMAX])
    np.testing.assert_allclose(wsim.c[o_c_loc:o_c_loc+3, 1], [TMIN, TMAX, TMAX])
    np.testing.assert_allclose(wsim.c[o_c_loc:o_c_loc+3, 2], [TMIN, 71, TMAX])
    np.testing.assert_allclose(wsim.c[o_c_loc:o_c_loc+3, 3], [TMIN, TMAX, TMAX])

    n_c_loc = wsim.c_locs[wsim.ppo_offset+4] # check n waveforms
    np.testing.assert_allclose(wsim.c[n_c_loc:n_c_loc+3, 0], [TMIN, TMAX, TMAX])
    np.testing.assert_allclose(wsim.c[n_c_loc:n_c_loc+3, 1], [TMIN, 21, TMAX])
    np.testing.assert_allclose(wsim.c[n_c_loc:n_c_loc+3, 2], [31, TMAX, TMAX])
    np.testing.assert_allclose(wsim.c[n_c_loc:n_c_loc+3, 3], [TMAX, TMAX, TMAX])

    wsim.c_to_s()

    # check a captures
    np.testing.assert_allclose(wsim.s[3:7, 2, 0], [0, TMAX, TMIN, 0])
    np.testing.assert_allclose(wsim.s[3:7, 2, 1], [0, 21, 61, 0])
    np.testing.assert_allclose(wsim.s[3:7, 2, 2], [1, 31, 31, 0])
    np.testing.assert_allclose(wsim.s[3:7, 2, 3], [0, 81, 81, 1])

    # check o captures
    np.testing.assert_allclose(wsim.s[3:7, 3, 0], [1, 51, 51, 0])
    np.testing.assert_allclose(wsim.s[3:7, 3, 1], [1, TMAX, TMIN, 1])
    np.testing.assert_allclose(wsim.s[3:7, 3, 2], [1, 71, 71, 0])
    np.testing.assert_allclose(wsim.s[3:7, 3, 3], [1, TMAX, TMIN, 1])

    # check o captures
    np.testing.assert_allclose(wsim.s[3:7, 4, 0], [1, TMAX, TMIN, 1])
    np.testing.assert_allclose(wsim.s[3:7, 4, 1], [1, 21, 21, 0])
    np.testing.assert_allclose(wsim.s[3:7, 4, 2], [0, 31, 31, 1])
    np.testing.assert_allclose(wsim.s[3:7, 4, 3], [0, TMAX, TMIN, 0])


def compare_to_logic_sim(wsim: WaveSim):
    choices = np.asarray([logic.ZERO, logic.ONE, logic.RISE, logic.FALL], dtype=np.uint8)
    rng = np.random.default_rng(10)
    tests = rng.choice(choices, (wsim.s_len, wsim.sims))

    wsim.s[0] = (tests & 2) >> 1
    wsim.s[3] = (tests & 2) >> 1
    wsim.s[1] = 0.0
    wsim.s[2] = tests & 1
    wsim.s[6] = tests & 1

    wsim.s_to_c()
    wsim.c_prop()
    wsim.c_to_s()

    resp = np.array(wsim.s[6], dtype=np.uint8) | (np.array(wsim.s[3], dtype=np.uint8)<<1)
    resp |= ((resp ^ (resp >> 1)) & 1) << 2  # transitions
    resp[wsim.pi_s_locs] = logic.UNASSIGNED

    lsim = LogicSim(wsim.circuit, tests.shape[-1])
    lsim.s[0] = logic.mv_to_bp(tests)
    lsim.s_to_c()
    lsim.c_prop()
    lsim.c_to_s()
    exp = logic.bp_to_mv(lsim.s[1])[:,:tests.shape[-1]]

    resp[resp == logic.PPULSE] = logic.ZERO
    resp[resp == logic.NPULSE] = logic.ONE

    exp[exp == logic.PPULSE] = logic.ZERO
    exp[exp == logic.NPULSE] = logic.ONE

    np.testing.assert_allclose(resp, exp)


def test_b15(b15_2ig_circuit_resolved, b15_2ig_delays):
    compare_to_logic_sim(WaveSim(b15_2ig_circuit_resolved, b15_2ig_delays, 8))


def test_b15_strip_forks(b15_2ig_circuit_resolved, b15_2ig_delays):
    compare_to_logic_sim(WaveSim(b15_2ig_circuit_resolved, b15_2ig_delays, 8, strip_forks=True))


def test_b15_cuda(b15_2ig_circuit_resolved, b15_2ig_delays):
    compare_to_logic_sim(WaveSimCuda(b15_2ig_circuit_resolved, b15_2ig_delays, 8, strip_forks=True))
