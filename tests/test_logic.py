import kyupy.logic as lg


def test_mvarray():

    # instantiation with shape

    ary = lg.MVArray(4)
    assert ary.length == 1
    assert len(ary) == 1
    assert ary.width == 4

    ary = lg.MVArray((3, 2))
    assert ary.length == 2
    assert len(ary) == 2
    assert ary.width == 3

    # instantiation with single vector

    ary = lg.MVArray([1, 0, 1])
    assert ary.length == 1
    assert ary.width == 3
    assert str(ary) == "['101']"
    assert ary[0] == '101'

    ary = lg.MVArray("10X-")
    assert ary.length == 1
    assert ary.width == 4
    assert str(ary) == "['10X-']"
    assert ary[0] == '10X-'

    ary = lg.MVArray("1")
    assert ary.length == 1
    assert ary.width == 1

    ary = lg.MVArray(["1"])
    assert ary.length == 1
    assert ary.width == 1

    # instantiation with multiple vectors

    ary = lg.MVArray([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert ary.length == 4
    assert ary.width == 2

    ary = lg.MVArray(["000", "001", "110", "---"])
    assert ary.length == 4
    assert ary.width == 3
    assert str(ary) == "['000', '001', '110', '---']"
    assert ary[2] == '110'

    # casting to 2-valued logic

    ary = lg.MVArray([0, 1, 2, None], m=2)
    assert ary.data[0] == lg.ZERO
    assert ary.data[1] == lg.ONE
    assert ary.data[2] == lg.ZERO
    assert ary.data[3] == lg.ZERO

    ary = lg.MVArray("0-X1PRFN", m=2)
    assert ary.data[0] == lg.ZERO
    assert ary.data[1] == lg.ZERO
    assert ary.data[2] == lg.ZERO
    assert ary.data[3] == lg.ONE
    assert ary.data[4] == lg.ZERO
    assert ary.data[5] == lg.ONE
    assert ary.data[6] == lg.ZERO
    assert ary.data[7] == lg.ONE

    # casting to 4-valued logic

    ary = lg.MVArray([0, 1, 2, None, 'F'], m=4)
    assert ary.data[0] == lg.ZERO
    assert ary.data[1] == lg.ONE
    assert ary.data[2] == lg.UNKNOWN
    assert ary.data[3] == lg.UNASSIGNED
    assert ary.data[4] == lg.ZERO

    ary = lg.MVArray("0-X1PRFN", m=4)
    assert ary.data[0] == lg.ZERO
    assert ary.data[1] == lg.UNASSIGNED
    assert ary.data[2] == lg.UNKNOWN
    assert ary.data[3] == lg.ONE
    assert ary.data[4] == lg.ZERO
    assert ary.data[5] == lg.ONE
    assert ary.data[6] == lg.ZERO
    assert ary.data[7] == lg.ONE

    # casting to 8-valued logic

    ary = lg.MVArray([0, 1, 2, None, 'F'], m=8)
    assert ary.data[0] == lg.ZERO
    assert ary.data[1] == lg.ONE
    assert ary.data[2] == lg.UNKNOWN
    assert ary.data[3] == lg.UNASSIGNED
    assert ary.data[4] == lg.FALL

    ary = lg.MVArray("0-X1PRFN", m=8)
    assert ary.data[0] == lg.ZERO
    assert ary.data[1] == lg.UNASSIGNED
    assert ary.data[2] == lg.UNKNOWN
    assert ary.data[3] == lg.ONE
    assert ary.data[4] == lg.PPULSE
    assert ary.data[5] == lg.RISE
    assert ary.data[6] == lg.FALL
    assert ary.data[7] == lg.NPULSE

    # copy constructor and casting

    ary8 = lg.MVArray(ary, m=8)
    assert ary8.length == 1
    assert ary8.width == 8
    assert ary8.data[7] == lg.NPULSE

    ary4 = lg.MVArray(ary, m=4)
    assert ary4.data[1] == lg.UNASSIGNED
    assert ary4.data[7] == lg.ONE

    ary2 = lg.MVArray(ary, m=2)
    assert ary2.data[1] == lg.ZERO
    assert ary2.data[7] == lg.ONE


def test_mv_operations():
    x1_2v = lg.MVArray("0011", m=2)
    x2_2v = lg.MVArray("0101", m=2)
    x1_4v = lg.MVArray("0000XXXX----1111", m=4)
    x2_4v = lg.MVArray("0X-10X-10X-10X-1", m=4)
    x1_8v = lg.MVArray("00000000XXXXXXXX--------11111111PPPPPPPPRRRRRRRRFFFFFFFFNNNNNNNN", m=8)
    x2_8v = lg.MVArray("0X-1PRFN0X-1PRFN0X-1PRFN0X-1PRFN0X-1PRFN0X-1PRFN0X-1PRFN0X-1PRFN", m=8)

    assert lg.mv_not(x1_2v)[0] == '1100'
    assert lg.mv_not(x1_4v)[0] == '1111XXXXXXXX0000'
    assert lg.mv_not(x1_8v)[0] == '11111111XXXXXXXXXXXXXXXX00000000NNNNNNNNFFFFFFFFRRRRRRRRPPPPPPPP'

    assert lg.mv_or(x1_2v, x2_2v)[0] == '0111'
    assert lg.mv_or(x1_4v, x2_4v)[0] == '0XX1XXX1XXX11111'
    assert lg.mv_or(x1_8v, x2_8v)[0] == '0XX1PRFNXXX1XXXXXXX1XXXX11111111PXX1PRFNRXX1RRNNFXX1FNFNNXX1NNNN'

    assert lg.mv_and(x1_2v, x2_2v)[0] == '0001'
    assert lg.mv_and(x1_4v, x2_4v)[0] == '00000XXX0XXX0XX1'
    assert lg.mv_and(x1_8v, x2_8v)[0] == '000000000XXXXXXX0XXXXXXX0XX1PRFN0XXPPPPP0XXRPRPR0XXFPPFF0XXNPRFN'

    assert lg.mv_xor(x1_2v, x2_2v)[0] == '0110'
    assert lg.mv_xor(x1_4v, x2_4v)[0] == '0XX1XXXXXXXX1XX0'
    assert lg.mv_xor(x1_8v, x2_8v)[0] == '0XX1PRFNXXXXXXXXXXXXXXXX1XX0NFRPPXXNPRFNRXXFRPNFFXXRFNPRNXXPNFRP'


def test_bparray():

    ary = lg.BPArray(4)
    assert ary.length == 1
    assert len(ary) == 1
    assert ary.width == 4

    ary = lg.BPArray((3, 2))
    assert ary.length == 2
    assert len(ary) == 2
    assert ary.width == 3

    assert lg.MVArray(lg.BPArray("01", m=2))[0] == '01'
    assert lg.MVArray(lg.BPArray("0X-1", m=4))[0] == '0X-1'
    assert lg.MVArray(lg.BPArray("0X-1PRFN", m=8))[0] == '0X-1PRFN'

    x1_2v = lg.BPArray("0011", m=2)
    x2_2v = lg.BPArray("0101", m=2)
    x1_4v = lg.BPArray("0000XXXX----1111", m=4)
    x2_4v = lg.BPArray("0X-10X-10X-10X-1", m=4)
    x1_8v = lg.BPArray("00000000XXXXXXXX--------11111111PPPPPPPPRRRRRRRRFFFFFFFFNNNNNNNN", m=8)
    x2_8v = lg.BPArray("0X-1PRFN0X-1PRFN0X-1PRFN0X-1PRFN0X-1PRFN0X-1PRFN0X-1PRFN0X-1PRFN", m=8)

    out_2v = lg.BPArray((4, 1), m=2)
    out_4v = lg.BPArray((16, 1), m=4)
    out_8v = lg.BPArray((64, 1), m=8)

    lg.bp_buf(out_2v.data, x1_2v.data)
    lg.bp_buf(out_4v.data, x1_4v.data)
    lg.bp_buf(out_8v.data, x1_8v.data)

    assert lg.MVArray(out_2v)[0] == '0011'
    assert lg.MVArray(out_4v)[0] == '0000XXXXXXXX1111'
    assert lg.MVArray(out_8v)[0] == '00000000XXXXXXXXXXXXXXXX11111111PPPPPPPPRRRRRRRRFFFFFFFFNNNNNNNN'

    lg.bp_not(out_2v.data, x1_2v.data)
    lg.bp_not(out_4v.data, x1_4v.data)
    lg.bp_not(out_8v.data, x1_8v.data)

    assert lg.MVArray(out_2v)[0] == '1100'
    assert lg.MVArray(out_4v)[0] == '1111XXXXXXXX0000'
    assert lg.MVArray(out_8v)[0] == '11111111XXXXXXXXXXXXXXXX00000000NNNNNNNNFFFFFFFFRRRRRRRRPPPPPPPP'

    lg.bp_or(out_2v.data, x1_2v.data, x2_2v.data)
    lg.bp_or(out_4v.data, x1_4v.data, x2_4v.data)
    lg.bp_or(out_8v.data, x1_8v.data, x2_8v.data)

    assert lg.MVArray(out_2v)[0] == '0111'
    assert lg.MVArray(out_4v)[0] == '0XX1XXX1XXX11111'
    assert lg.MVArray(out_8v)[0] == '0XX1PRFNXXX1XXXXXXX1XXXX11111111PXX1PRFNRXX1RRNNFXX1FNFNNXX1NNNN'

    lg.bp_and(out_2v.data, x1_2v.data, x2_2v.data)
    lg.bp_and(out_4v.data, x1_4v.data, x2_4v.data)
    lg.bp_and(out_8v.data, x1_8v.data, x2_8v.data)

    assert lg.MVArray(out_2v)[0] == '0001'
    assert lg.MVArray(out_4v)[0] == '00000XXX0XXX0XX1'
    assert lg.MVArray(out_8v)[0] == '000000000XXXXXXX0XXXXXXX0XX1PRFN0XXPPPPP0XXRPRPR0XXFPPFF0XXNPRFN'

    lg.bp_xor(out_2v.data, x1_2v.data, x2_2v.data)
    lg.bp_xor(out_4v.data, x1_4v.data, x2_4v.data)
    lg.bp_xor(out_8v.data, x1_8v.data, x2_8v.data)

    assert lg.MVArray(out_2v)[0] == '0110'
    assert lg.MVArray(out_4v)[0] == '0XX1XXXXXXXX1XX0'
    assert lg.MVArray(out_8v)[0] == '0XX1PRFNXXXXXXXXXXXXXXXX1XX0NFRPPXXNPRFNRXXFRPNFFXXRFNPRNXXPNFRP'
