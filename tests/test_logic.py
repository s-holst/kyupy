import numpy as np
import kyupy.logic as lg
from kyupy.logic import mvarray, bparray, bp_to_mv, mv_to_bp


def assert_equal_shape_and_contents(actual, desired):
    desired = np.array(desired, dtype=np.uint8)
    assert actual.shape == desired.shape
    np.testing.assert_allclose(actual, desired)


def test_mvarray_single_vector():
    assert_equal_shape_and_contents(mvarray(1, 0, 1), [lg.ONE, lg.ZERO, lg.ONE])
    assert_equal_shape_and_contents(mvarray([1, 0, 1]), [lg.ONE, lg.ZERO, lg.ONE])
    assert_equal_shape_and_contents(mvarray('10X-RFPN'), [lg.ONE, lg.ZERO, lg.UNKNOWN, lg.UNASSIGNED, lg.RISE, lg.FALL, lg.PPULSE, lg.NPULSE])
    assert_equal_shape_and_contents(mvarray(['1']), [lg.ONE])
    assert_equal_shape_and_contents(mvarray('1'), [lg.ONE])


def test_mvarray_multi_vector():
    assert_equal_shape_and_contents(mvarray([0, 0], [0, 1], [1, 0], [1, 1]), [[lg.ZERO, lg.ZERO, lg.ONE, lg.ONE], [lg.ZERO, lg.ONE, lg.ZERO, lg.ONE]])
    assert_equal_shape_and_contents(mvarray('10X', '--1'), [[lg.ONE, lg.UNASSIGNED], [lg.ZERO, lg.UNASSIGNED], [lg.UNKNOWN, lg.ONE]])


def test_mv_ops():
    x1_8v = mvarray('00000000XXXXXXXX--------11111111PPPPPPPPRRRRRRRRFFFFFFFFNNNNNNNN')
    x2_8v = mvarray('0X-1PRFN'*8)

    assert_equal_shape_and_contents(lg.mv_not(x1_8v), mvarray('11111111XXXXXXXXXXXXXXXX00000000NNNNNNNNFFFFFFFFRRRRRRRRPPPPPPPP'))
    assert_equal_shape_and_contents(lg.mv_or(x1_8v, x2_8v), mvarray('0XX1PRFNXXX1XXXXXXX1XXXX11111111PXX1PRFNRXX1RRNNFXX1FNFNNXX1NNNN'))
    assert_equal_shape_and_contents(lg.mv_and(x1_8v, x2_8v), mvarray('000000000XXXXXXX0XXXXXXX0XX1PRFN0XXPPPPP0XXRPRPR0XXFPPFF0XXNPRFN'))
    assert_equal_shape_and_contents(lg.mv_xor(x1_8v, x2_8v), mvarray('0XX1PRFNXXXXXXXXXXXXXXXX1XX0NFRPPXXNPRFNRXXFRPNFFXXRFNPRNXXPNFRP'))

    # TODO
    #assert_equal_shape_and_contents(lg.mv_transition(x1_8v, x2_8v), mvarray('0XXR PRFNXXXXXXXXXXXXXXXX1XX0NFRPPXXNPRFNRXXFRPNFFXXRFNPRNXXPNFRP'))

    x30_8v = mvarray('0000000000000000000000000000000000000000000000000000000000000000')
    x31_8v = mvarray('1111111111111111111111111111111111111111111111111111111111111111')

    assert_equal_shape_and_contents(lg.mv_latch(x1_8v, x2_8v, x30_8v), mvarray('0XX000000XXXXXXX0XXXXXXX0XX10R110XX000000XXR0R0R0XXF001F0XX10R11'))
    assert_equal_shape_and_contents(lg.mv_latch(x1_8v, x2_8v, x31_8v), mvarray('1XX01F001XXXXXXX1XXXXXXX1XX111111XX01F001XXR110R1XXF1F1F1XX11111'))


def test_bparray():

    bpa = bparray('0X-1PRFN')
    assert bpa.shape == (8, 3, 1)

    bpa = bparray('0X-1PRFN-')
    assert bpa.shape == (9, 3, 1)

    bpa = bparray('000', '001', '010', '011', '100', '101', '110', '111')
    assert bpa.shape == (3, 3, 1)

    bpa = bparray('000', '001', '010', '011', '100', '101', '110', '111', 'RFX')
    assert bpa.shape == (3, 3, 2)

    assert_equal_shape_and_contents(bp_to_mv(bparray('0X-1PRFN'))[:,0], mvarray('0X-1PRFN'))
    assert_equal_shape_and_contents(bparray('0X-1PRFN'), mv_to_bp(mvarray('0X-1PRFN')))

    x1_8v = bparray('00000000XXXXXXXX--------11111111PPPPPPPPRRRRRRRRFFFFFFFFNNNNNNNN')
    x2_8v = bparray('0X-1PRFN'*8)

    out_8v = np.empty((64, 3, 1), dtype=np.uint8)

    assert_equal_shape_and_contents(bp_to_mv(lg.bp8v_buf(out_8v, x1_8v))[:,0], mvarray('00000000XXXXXXXXXXXXXXXX11111111PPPPPPPPRRRRRRRRFFFFFFFFNNNNNNNN'))
    assert_equal_shape_and_contents(bp_to_mv(lg.bp8v_or(out_8v, x1_8v, x2_8v))[:,0], mvarray('0XX1PRFNXXX1XXXXXXX1XXXX11111111PXX1PRFNRXX1RRNNFXX1FNFNNXX1NNNN'))
    assert_equal_shape_and_contents(bp_to_mv(lg.bp8v_and(out_8v, x1_8v, x2_8v))[:,0], mvarray('000000000XXXXXXX0XXXXXXX0XX1PRFN0XXPPPPP0XXRPRPR0XXFPPFF0XXNPRFN'))
    assert_equal_shape_and_contents(bp_to_mv(lg.bp8v_xor(out_8v, x1_8v, x2_8v))[:,0], mvarray('0XX1PRFNXXXXXXXXXXXXXXXX1XX0NFRPPXXNPRFNRXXFRPNFFXXRFNPRNXXPNFRP'))

    x30_8v = bparray('0000000000000000000000000000000000000000000000000000000000000000')
    x31_8v = bparray('1111111111111111111111111111111111111111111111111111111111111111')

    assert_equal_shape_and_contents(bp_to_mv(lg.bp8v_latch(out_8v, x1_8v, x2_8v, x30_8v))[:,0], mvarray('0XX000000XXXXXXX0XXXXXXX0XX10R110XX000000XXR0R0R0XXF001F0XX10R11'))
    assert_equal_shape_and_contents(bp_to_mv(lg.bp8v_latch(out_8v, x1_8v, x2_8v, x31_8v))[:,0], mvarray('1XX01F001XXXXXXX1XXXXXXX1XX111111XX01F001XXR110R1XXF1F1F1XX11111'))
