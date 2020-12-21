from kyupy import verilog


def test_b01(mydir):
    with open(mydir / 'b01.v', 'r') as f:
        modules = verilog.parse(f.read())
    assert modules is not None
    assert verilog.load(mydir / 'b01.v') is not None
