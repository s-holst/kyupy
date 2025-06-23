import numpy as np

from kyupy import sdf, verilog, bench
from kyupy.wave_sim import WaveSim, TMAX, TMIN
from kyupy.techlib import SAED32, NANGATE45

def test_parse():
    test = '''
    (DELAYFILE
    (SDFVERSION "OVI 2.1")
    (DESIGN "test")
    (DATE "Wed May 31 14:46:06 2017")
    (VENDOR "saed90nm_max")
    (PROGRAM "Synopsys Design Compiler cmos-annotated")
    (VERSION "I-2013.12-ICC-SP3")
    (DIVIDER /)
    (VOLTAGE 1.20:1.20:1.20)
    (PROCESS "TYPICAL")
    (TEMPERATURE 25.00:25.00:25.00)
    (TIMESCALE 1ns)
    (CELL
        (CELLTYPE "b14")
        (INSTANCE)
        (DELAY
            (ABSOLUTE
                (INTERCONNECT U621/ZN U19246/IN1 (0.000:0.000:0.000))
                (INTERCONNECT U13292/QN U19246/IN2 (0.001:0.001:0.001))
                (INTERCONNECT U15050/QN U19247/IN1 (0.000:0.000:0.000))
                (INTERCONNECT U13293/QN U19247/IN2 (0.000:0.000:0.000) (0.000:0.000:0.000))
            )
        )
    )
    (CELL
        (CELLTYPE "INVX2")
        (INSTANCE U78)
        (DELAY
            (ABSOLUTE
                (IOPATH INP ZN (0.201:0.227:0.227) (0.250:0.271:0.271))
            )
        )
    )
    (CELL
        (CELLTYPE "SDFFARX1")
        (INSTANCE reg3_reg_1_0)
        (DELAY
            (ABSOLUTE
                (IOPATH (posedge CLK) Q (0.707:0.710:0.710) (0.737:0.740:0.740))
                (IOPATH (negedge RSTB) Q () (0.909:0.948:0.948))
                (IOPATH (posedge CLK) QN (0.585:0.589:0.589) (0.545:0.550:0.550))
                (IOPATH (negedge RSTB) QN (1.546:1.593:1.593) ())
            )
        )
        (TIMINGCHECK
            (WIDTH (posedge CLK) (0.284:0.284:0.284))
            (WIDTH (negedge CLK) (0.642:0.642:0.642))
            (SETUP (posedge D) (posedge CLK) (0.544:0.553:0.553))
            (SETUP (negedge D) (posedge CLK) (0.620:0.643:0.643))
            (HOLD (posedge D) (posedge CLK) (-0.321:-0.331:-0.331))
            (HOLD (negedge D) (posedge CLK) (-0.196:-0.219:-0.219))
            (RECOVERY (posedge RSTB) (posedge CLK) (-1.390:-1.455:-1.455))
            (HOLD (posedge RSTB) (posedge CLK) (1.448:1.509:1.509))
            (SETUP (posedge SE) (posedge CLK) (0.662:0.670:0.670))
            (SETUP (negedge SE) (posedge CLK) (0.698:0.702:0.702))
            (HOLD (posedge SE) (posedge CLK) (-0.435:-0.444:-0.444))
            (HOLD (negedge SE) (posedge CLK) (-0.291:-0.295:-0.295))
            (SETUP (posedge SI) (posedge CLK) (0.544:0.544:0.544))
            (SETUP (negedge SI) (posedge CLK) (0.634:0.688:0.688))
            (HOLD (posedge SI) (posedge CLK) (-0.317:-0.318:-0.318))
            (HOLD (negedge SI) (posedge CLK) (-0.198:-0.247:-0.247))
            (WIDTH (negedge RSTB) (0.345:0.345:0.345))
    )))
    '''
    df = sdf.parse(test)
    assert df.name == 'test'


def test_b15(mydir):
    df = sdf.load(mydir / 'b15_2ig.sdf.gz')
    assert df.name == 'b15'


def test_gates(mydir):
    c = verilog.load(mydir / 'gates.v', tlib=NANGATE45)
    df = sdf.load(mydir / 'gates.sdf')
    lt = df.iopaths(c, tlib=NANGATE45)[1]
    nand_a = c.cells['nandgate'].ins[0]
    nand_b = c.cells['nandgate'].ins[1]
    and_a = c.cells['andgate'].ins[0]
    and_b = c.cells['andgate'].ins[1]

    assert lt[nand_a, 0, 0] == 0.103
    assert lt[nand_a, 0, 1] == 0.127

    assert lt[nand_b, 0, 0] == 0.086
    assert lt[nand_b, 0, 1] == 0.104

    assert lt[and_a, 0, 0] == 0.378
    assert lt[and_a, 0, 1] == 0.377

    assert lt[and_b, 0, 0] == 0.375
    assert lt[and_b, 0, 1] == 0.370


def test_nand_xor():
    c = bench.parse("""
        input(A1,A2)
        output(lt_1237_U91,lt_1237_U92)
        lt_1237_U91 = NAND2X0_RVT(A1,A2)
        lt_1237_U92 = XOR2X1_RVT(A1,A2)
        """)
    df = sdf.parse("""
        (DELAYFILE
            (CELL
                (CELLTYPE "NAND2X0_RVT")
                (INSTANCE lt_1237_U91)
                (DELAY
                    (ABSOLUTE
                        (IOPATH A1 Y (0.018:0.022:0.021) (0.017:0.019:0.019))
                        (IOPATH A2 Y (0.021:0.024:0.024) (0.018:0.021:0.021))
                    )
                )
            )
            (CELL
                (CELLTYPE "XOR2X1_RVT")
                (INSTANCE lt_1237_U92)
                (DELAY
                    (ABSOLUTE
                        (IOPATH (posedge A1) Y (0.035:0.038:0.038) (0.037:0.062:0.062))
                        (IOPATH (negedge A1) Y (0.035:0.061:0.061) (0.036:0.040:0.040))
                        (IOPATH (posedge A2) Y (0.042:0.043:0.043) (0.051:0.064:0.064))
                        (IOPATH (negedge A2) Y (0.041:0.066:0.066) (0.051:0.053:0.053))
                    )
                )
            )
        )
        """)
    d = df.iopaths(c, tlib=SAED32)[1]
    c.resolve_tlib_cells(SAED32)
    sim = WaveSim(c, delays=d, sims=16)

    # input A1
    sim.s[0,0] = [0,1,0,1] * 4  # initial values  0101010101010101
    sim.s[1,0] = 0.0            # transition time
    sim.s[2,0] = [0,0,1,1] * 4  # final values    0011001100110011

    # input A2
    sim.s[0,1] = ([0]*4 + [1]*4)*2  # initial values  0000111100001111
    sim.s[1,1] = 0.0                # transition time
    sim.s[2,1] = [0]*8 + [1]*8      # final values    0000000011111111

    # A1:   0FR10FR10FR10FR1
    # A2:   0000FFFFRRRR1111
    # nand: 11111RNR1NFF1RF0
    # xor:  0FR1FPPRRNPF1RF0

    sim.s_to_c()
    sim.c_prop()
    sim.c_to_s()

    eat = sim.s[4,2:]
    lst = sim.s[5,2:]

    # NAND-gate output
    assert np.allclose(eat[0], [
        TMAX, TMAX, TMAX, TMAX, TMAX,
        0.022,  # FF -> rising Y: min(0.022, 0.024)
        TMAX,   # RF: pulse filtered
        0.024,  # falling A2 -> rising Y
        TMAX,
        TMAX,   # FR: pulse filtered
        0.021,  # RR -> falling Y: max(0.019, 0.021)
        0.021,  # rising A2 -> falling Y
        TMAX,
        0.022,  # falling A1 -> rising Y
        0.019,  # rising A1 -> falling Y
        TMAX
    ])

    assert np.allclose(lst[0], [
        TMIN, TMIN, TMIN, TMIN, TMIN,
        0.022,  # FF -> rising Y: min(0.022, 0.024)
        TMIN,   # RF: pulse filtered
        0.024,  # falling A2 -> rising Y
        TMIN,
        TMIN,   # FR: pulse filtered
        0.021,  # RR -> falling Y: max(0.019, 0.021)
        0.021,  # rising A2 -> falling Y
        TMIN,
        0.022,  # falling A1 -> rising Y
        0.019,  # rising A1 -> falling Y
        TMIN
    ])

    #XOR-gate output
    assert np.allclose(eat[1], [
        TMAX,
        0.040,  # A1:F -> Y:F
        0.038,  # A1:R -> Y:R
        TMAX,
        0.053,  # A2:F -> Y:F
        TMAX,   # P filtered
        TMAX,   # P filtered
        0.066,  # A2:F -> Y:R
        0.043,  # A2:R -> Y:R
        TMAX,   # N filtered
        TMAX,   # P filtered
        0.064,  # A2:R -> Y:F
        TMAX,
        0.061,  # A1:F -> Y:R
        0.062,  # A1:R -> Y:F
        TMAX,
    ])

    assert np.allclose(lst[1], [
        TMIN,
        0.040,  # A1:F -> Y:F
        0.038,  # A1:R -> Y:R
        TMIN,
        0.053,  # A2:F -> Y:F
        TMIN,   # P filtered
        TMIN,   # P filtered
        0.066,  # A2:F -> Y:R
        0.043,  # A2:R -> Y:R
        TMIN,   # N filtered
        TMIN,   # P filtered
        0.064,  # A2:R -> Y:F
        TMIN,
        0.061,  # A1:F -> Y:R
        0.062,  # A1:R -> Y:F
        TMIN,
    ])