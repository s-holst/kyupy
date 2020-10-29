from kyupy.packed_vectors import PackedVectors


def test_basic():
    ba = PackedVectors(8, 1, 1)
    assert '0\n0\n0\n0\n0\n0\n0\n0' == str(ba)
    ba.set_value(0, 0, 1)
    ba.set_value(1, 0, 'H')
    ba.set_value(2, 0, 'h')
    ba.set_value(3, 0, True)
    ba.set_value(4, 0, 0)
    ba.set_value(5, 0, 'L')
    ba.set_value(6, 0, 'l')
    ba.set_value(7, 0, False)
    assert '1\n1\n1\n1\n0\n0\n0\n0' == str(ba)
    ba.set_value(1, 0, '0')
    ba.set_value(5, 0, '1')
    assert '1\n0\n1\n1\n0\n1\n0\n0' == str(ba)
    ba = PackedVectors(8, 1, 2)
    assert '-\n-\n-\n-\n-\n-\n-\n-' == str(ba)
    ba.set_value(0, 0, 1)
    ba.set_value(7, 0, 0)
    ba.set_value(4, 0, 'X')
    assert '1\n-\n-\n-\nX\n-\n-\n0' == str(ba)
    ba.set_value(4, 0, '-')
    assert '1\n-\n-\n-\n-\n-\n-\n0' == str(ba)
    ba = PackedVectors(8, 2, 2)
    assert '--\n--\n--\n--\n--\n--\n--\n--' == str(ba)
    ba.set_value(0, 0, '1')
    ba.set_value(7, 1, '0')
    ba.set_values(1, 'XX')
    assert '1-\nXX\n--\n--\n--\n--\n--\n-0' == str(ba)


def test_8v():
    ba = PackedVectors(1, 8, 3)
    assert '--------' == str(ba)
    ba.set_values(0, r'-x01^v\/')
    assert r'-X01PNFR' == str(ba)
    ba.set_values(0, '-XLHPNFR')
    assert r'-X01PNFR' == str(ba)
    ba.set_values(0, '-xlhpnfr')
    assert r'-X01PNFR' == str(ba)
    p1 = PackedVectors(1, 8, 1)
    p2 = PackedVectors(1, 8, 1)
    p1.set_values(0, '01010101')
    p2.set_values(0, '00110011')
    p = PackedVectors.from_pair(p1, p2)
    assert r'0FR10FR1' == str(p)
    p1 = PackedVectors(1, 8, 2)
    p2 = PackedVectors(1, 8, 2)
    p1.set_values(0, '0101-X-X')
    p2.set_values(0, '00110011')
    p = PackedVectors.from_pair(p1, p2)
    assert r'0FR1----' == str(p)
    p1.set_values(0, '0101-X-X')
    p2.set_values(0, '-X-X--XX')
    p = PackedVectors.from_pair(p1, p2)
    assert r'--------' == str(p)


def test_slicing():
    lv = PackedVectors(3, 2, 1)
    assert '00\n00\n00' == str(lv)
    lv.set_value(1, 0, '1')
    lv.set_value(1, 1, '1')
    assert '00' == lv[0]
    assert '11' == lv[1]
    assert 3 == len(lv)
    lv2 = lv[1:3]
    assert 2 == len(lv2)
    assert '11' == lv2[0]
    assert '00' == lv2[1]


def test_copy():
    lv1 = PackedVectors(8, 1, 1)
    lv1.set_values_for_position(0, '01010101')
    lv2 = PackedVectors(8, 1, 1)
    lv2.set_values_for_position(0, '00100101')
    diff = lv1.diff(lv2)
    lv3 = lv1.copy(selection_mask=diff)
    assert str(lv3) == '1\n0\n1'
    lv4 = lv1.copy(selection_mask=~diff)
    assert str(lv4) == '0\n0\n1\n0\n1'
    lv5 = lv3 + lv4
    assert str(lv5) == '1\n0\n1\n0\n0\n1\n0\n1'

