import numpy as np

from kyupy.wave_sim import WaveSim, WaveSimCuda, wave_eval, TMIN, TMAX
from kyupy.logic_sim import LogicSim
from kyupy import verilog, sdf, logic
from kyupy.logic import MVArray, BPArray


def test_wave_eval():
    # SDF specifies IOPATH delays with respect to output polarity
    # SDF pulse rejection value is determined by IOPATH causing last transition and polarity of last transition
    line_times = np.zeros((3, 2, 2))
    line_times[0, 0, 0] = 0.1  # A -> Z rise delay
    line_times[0, 0, 1] = 0.2  # A -> Z fall delay
    line_times[0, 1, 0] = 0.1  # A -> Z negative pulse limit (terminate in rising Z)
    line_times[0, 1, 1] = 0.2  # A -> Z positive pulse limit
    line_times[1, 0, 0] = 0.3  # as above for B -> Z
    line_times[1, 0, 1] = 0.4
    line_times[1, 1, 0] = 0.3
    line_times[1, 1, 1] = 0.4

    state = np.zeros((3*16, 1)) + TMAX  # 3 waveforms of capacity 16
    state[::16, 0] = 16  # first entry is capacity
    a = state[0:16, 0]
    b = state[16:32, 0]
    z = state[32:, 0]
    sat = np.zeros((3, 3), dtype='int')
    sat[0] = 0, 16, 0
    sat[1] = 16, 16, 0
    sat[2] = 32, 16, 0

    wave_eval((0b0111, 2, 0, 1), state, sat, 0, line_times)
    assert z[0] == TMIN

    a[0] = TMIN
    wave_eval((0b0111, 2, 0, 1), state, sat, 0, line_times)
    assert z[0] == TMIN

    b[0] = TMIN
    wave_eval((0b0111, 2, 0, 1), state, sat, 0, line_times)
    assert z[0] == TMAX

    a[0] = 1  # A _/^^^
    b[0] = 2  # B __/^^
    wave_eval((0b0111, 2, 0, 1), state, sat, 0, line_times)
    assert z[0] == TMIN  # ^^^\___ B -> Z fall delay
    assert z[1] == 2.4
    assert z[2] == TMAX

    a[0] = TMIN  # A ^^^^^^
    b[0] = TMIN  # B ^^^\__
    b[1] = 2
    wave_eval((0b0111, 2, 0, 1), state, sat, 0, line_times)
    assert z[0] == 2.3  # ___/^^^ B -> Z rise delay
    assert z[1] == TMAX

    # pos pulse of 0.35 at B -> 0.45 after delays
    a[0] = TMIN  # A ^^^^^^^^
    b[0] = TMIN
    b[1] = 2     # B ^^\__/^^
    b[2] = 2.35
    wave_eval((0b0111, 2, 0, 1), state, sat, 0, line_times)
    assert z[0] == 2.3  # __/^^\__
    assert z[1] == 2.75
    assert z[2] == TMAX

    # neg pulse of 0.45 at B -> 0.35 after delays
    a[0] = TMIN  # A ^^^^^^^^
    b[0] = 2  # B __/^^\__
    b[1] = 2.45
    b[2] = TMAX
    wave_eval((0b0111, 2, 0, 1), state, sat, 0, line_times)
    assert z[0] == TMIN  # ^^\__/^^
    assert z[1] == 2.4
    assert z[2] == 2.75
    assert z[3] == TMAX

    # neg pulse of 0.35 at B -> 0.25 after delays (filtered)
    a[0] = TMIN  # A ^^^^^^^^
    b[0] = 2  # B __/^^\__
    b[1] = 2.35
    b[2] = TMAX
    wave_eval((0b0111, 2, 0, 1), state, sat, 0, line_times)
    assert z[0] == TMIN  # ^^^^^^
    assert z[1] == TMAX

    # pos pulse of 0.25 at B -> 0.35 after delays (filtered)
    a[0] = TMIN  # A ^^^^^^^^
    b[0] = TMIN
    b[1] = 2  # B ^^\__/^^
    b[2] = 2.25
    wave_eval((0b0111, 2, 0, 1), state, sat, 0, line_times)
    assert z[0] == TMAX  # ______


def compare_to_logic_sim(wsim):
    tests = MVArray((len(wsim.interface), wsim.sims))
    choices = np.asarray([logic.ZERO, logic.ONE, logic.RISE, logic.FALL], dtype=np.uint8)
    rng = np.random.default_rng(10)
    tests.data[...] = rng.choice(choices, tests.data.shape)
    tests_bp = BPArray(tests)
    wsim.assign(tests_bp)
    wsim.propagate()
    cdata = wsim.capture()

    resp = MVArray(tests)

    for iidx, inode in enumerate(wsim.interface):
        if len(inode.ins) > 0:
            for vidx in range(wsim.sims):
                resp.data[iidx, vidx] = logic.ZERO if cdata[iidx, vidx, 0] < 0.5 else logic.ONE
                # resp.set_value(vidx, iidx, 0 if cdata[iidx, vidx, 0] < 0.5 else 1)

    lsim = LogicSim(wsim.circuit, len(tests_bp))
    lsim.assign(tests_bp)
    lsim.propagate()
    exp_bp = BPArray(tests_bp)
    lsim.capture(exp_bp)
    exp = MVArray(exp_bp)

    for i in range(8):
        exp_str = exp[i].replace('R', '1').replace('F', '0').replace('P', '0').replace('N', '1')
        res_str = resp[i].replace('R', '1').replace('F', '0').replace('P', '0').replace('N', '1')
        assert res_str == exp_str


def test_b14(mydir):
    c = verilog.load(mydir / 'b14.v.gz', branchforks=True)
    df = sdf.load(mydir / 'b14.sdf.gz')
    lt = df.annotation(c)
    wsim = WaveSim(c, lt, 8)
    compare_to_logic_sim(wsim)


def test_b14_strip_forks(mydir):
    c = verilog.load(mydir / 'b14.v.gz', branchforks=True)
    df = sdf.load(mydir / 'b14.sdf.gz')
    lt = df.annotation(c)
    wsim = WaveSim(c, lt, 8, strip_forks=True)
    compare_to_logic_sim(wsim)


def test_b14_cuda(mydir):
    c = verilog.load(mydir / 'b14.v.gz', branchforks=True)
    df = sdf.load(mydir / 'b14.sdf.gz')
    lt = df.annotation(c)
    wsim = WaveSimCuda(c, lt, 8)
    compare_to_logic_sim(wsim)
