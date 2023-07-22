from kyupy import verilog
from kyupy.techlib import SAED90, SAED32

def test_b01(mydir):
    with open(mydir / 'b01.v', 'r') as f:
        c = verilog.parse(f.read(), tlib=SAED90)
    assert c is not None
    assert verilog.load(mydir / 'b01.v', tlib=SAED90) is not None

    assert len(c.nodes) == 139
    assert len(c.lines) == 203
    stats = c.stats
    assert stats['input'] == 6
    assert stats['output'] == 3
    assert stats['__seq__'] == 5


def test_b15(mydir):
    c = verilog.load(mydir / 'b15_4ig.v.gz', tlib=SAED32)
    assert len(c.nodes) == 12067
    assert len(c.lines) == 20731
    stats = c.stats
    assert stats['input'] == 40
    assert stats['output'] == 71
    assert stats['__seq__'] == 417


def test_gates(mydir):
    c = verilog.load(mydir / 'gates.v', tlib=SAED90)
    assert len(c.nodes) == 10
    assert len(c.lines) == 10
    stats = c.stats
    assert stats['input'] == 2
    assert stats['output'] == 2
    assert stats['__seq__'] == 0


def test_halton2(mydir):
    c = verilog.load(mydir / 'rng_haltonBase2.synth_yosys.v', tlib=SAED90)
    assert len(c.nodes) == 146
    assert len(c.lines) == 210
    stats = c.stats
    assert stats['input'] == 2
    assert stats['output'] == 12
    assert stats['__seq__'] == 12