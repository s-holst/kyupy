import numpy as np
import importlib.util
if importlib.util.find_spec('numba') is not None:
    import numba
else:
    from . import numba
    print('Numba unavailable. Falling back to pure python')


_pop_count_lut = np.asarray([bin(x).count('1') for x in range(256)])


def popcount(a):
    return np.sum(_pop_count_lut[a])


_bit_in_lut = np.array([2 ** x for x in range(7, -1, -1)], dtype='uint8')


@numba.njit
def bit_in(a, pos):
    return a[pos >> 3] & _bit_in_lut[pos & 7]

