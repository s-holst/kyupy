from kyupy import sdf, verilog


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
    # print(f'DelayFile(name={df.name}, interconnects={len(df.interconnects)}, iopaths={len(df.iopaths)})')


def test_b14(mydir):
    df = sdf.load(mydir / 'b14.sdf.gz')
    assert df.name == 'b14'


def test_gates(mydir):
    c = verilog.load(mydir / 'gates.v')
    df = sdf.load(mydir / 'gates.sdf')
    lt = df.annotation(c, dataset=1)
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
