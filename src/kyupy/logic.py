"""Core module for handling 2-, 4-, and 8-valued logic data and signal values.

Logic values are stored in numpy arrays with data type ``np.uint8``.
There are no explicit data structures in KyuPy for holding patterns, pattern sets or vectors.
However, there are conventions on logic value encoding and on the order of axes.
Utility functions defined here follow these conventions.

8 logic values are defined as integer constants.

* For 2-valued logic: ``ZERO`` and ``ONE``
* 4-valued logic adds: ``UNASSIGNED`` and ``UNKNOWN``
* 8-valued logic adds: ``RISE``, ``FALL``, ``PPULSE``, and ``NPULSE``.

In general, the bits in these constants have the following meaning:

* bit0: Final/settled binary value of a signal
* bit1: Initial binary value of a signal
* bit2: Activity or transitions are present on a signal

Except when bit0 differs from bit1, but bit2 (activity) is 0:

* bit0 = 1, bit1 = 0, bit2 = 0 means ``UNKNOWN`` in 4-valued and 8-valued logic.
* bit0 = 0, bit1 = 1, bit2 = 0 means ``UNASSIGNED`` in 4-valued and 8-valued logic.

2-valued logic only considers bit0, but should store logic one as ``ONE=0b011`` for interoperability.
4-valued logic only considers bit0 and bit1.
8-valued logic considers all 3 bits.

Logic values are stored in numpy arrays of data type ``np.uint8``.
The axis convention is as follows:

* The **last** axis goes along patterns/vectors. I.e. ``values[...,0]`` is pattern 0, ``values[...,1]`` is pattern 1, etc.
* The **second-to-last** axis goes along the I/O and flip-flops of circuits. For a circuit ``c``, this axis is usually
  ``len(c.s_nodes)`` long. The values of all inputs, outputs and flip-flops are stored within the same array and the location
  along the second-to-last axis is determined by the order in :py:attr:`~kyupy.circuit.Circuit.s_nodes`.

Two storage formats are used in KyuPy:

* ``mv...`` (for "multi-valued"): Each logic value is stored in the least significant 3 bits of ``np.uint8``.
* ``bp...`` (for "bit-parallel"): Groups of 8 logic values are stored as three ``np.uint8``. This format is used
  for bit-parallel logic simulations. It is also more memory-efficient.

The functions in this module use the ``mv...`` and ``bp...`` prefixes to signify the storage format they operate on.

"""

from collections.abc import Iterable

import numpy as np

from . import numba, hr_bytes


ZERO = 0b000
"""Integer constant ``0b000`` for logic-0. ``'0'``, ``0``, ``False``, ``'L'``, and ``'l'`` are interpreted as ``ZERO``.
"""
UNKNOWN = 0b001
"""Integer constant ``0b001`` for unknown or conflict. ``'X'``, or any other value is interpreted as ``UNKNOWN``.
"""
UNASSIGNED = 0b010
"""Integer constant ``0b010`` for unassigned or high-impedance. ``'-'``, ``None``, ``'Z'``, and ``'z'`` are
interpreted as ``UNASSIGNED``.
"""
ONE = 0b011
"""Integer constant ``0b011`` for logic-1. ``'1'``, ``1``, ``True``, ``'H'``, and ``'h'`` are interpreted as ``ONE``.
"""
PPULSE = 0b100
"""Integer constant ``0b100`` for positive pulse, meaning initial and final values are 0, but there is some activity
on a signal. ``'P'``, ``'p'``, and ``'^'`` are interpreted as ``PPULSE``.
"""
RISE = 0b101
"""Integer constant ``0b110`` for a rising transition. ``'R'``, ``'r'``, and ``'/'`` are interpreted as ``RISE``.
"""
FALL = 0b110
"""Integer constant ``0b101`` for a falling transition. ``'F'``, ``'f'``, and ``'\\'`` are interpreted as ``FALL``.
"""
NPULSE = 0b111
"""Integer constant ``0b111`` for negative pulse, meaning initial and final values are 1, but there is some activity
on a signal. ``'N'``, ``'n'``, and ``'v'`` are interpreted as ``NPULSE``.
"""


def interpret(value):
    """Converts characters, strings, and lists of them to lists of logic constants defined above.

    :param value: A character (string of length 1), Boolean, Integer, None, or Iterable.
        Iterables (such as strings) are traversed and their individual characters are interpreted.
    :return: A logic constant or a (possibly multi-dimensional) list of logic constants.
    """
    if isinstance(value, Iterable) and not (isinstance(value, str) and len(value) == 1):
        return list(map(interpret, value))
    if value in [0, '0', False, 'L', 'l']: return ZERO
    if value in [1, '1', True, 'H', 'h']: return ONE
    if value in [None, '-', 'Z', 'z']: return UNASSIGNED
    if value in ['R', 'r', '/']: return RISE
    if value in ['F', 'f', '\\']: return FALL
    if value in ['P', 'p', '^']: return PPULSE
    if value in ['N', 'n', 'v']: return NPULSE
    return UNKNOWN


def mvarray(*a):
    """Converts (lists of) Boolean values or strings into a multi-valued array.

    The given values are interpreted and the axes are arranged as per KyuPy's convention.
    Use this function to convert strings into multi-valued arrays.
    """
    mva = np.array(interpret(a), dtype=np.uint8)
    if mva.ndim < 2: return mva
    if mva.shape[-2] > 1: return mva.swapaxes(-1, -2)
    return mva[..., 0, :]


def mv_str(mva, delim='\n'):
    """Renders a given multi-valued array into a string.
    """
    sa = np.choose(mva, np.array([*'0X-1PRFN'], dtype=np.unicode_))
    if not hasattr(mva, 'ndim') or mva.ndim == 0: return sa
    if mva.ndim == 1: return ''.join(sa)
    return delim.join([''.join(c) for c in sa.swapaxes(-1,-2)])


def _mv_not(out, inp):
    np.bitwise_xor(inp, 0b11, out=out)  # this also exchanges UNASSIGNED <-> UNKNOWN
    np.putmask(out, (inp == UNKNOWN), UNKNOWN)  # restore UNKNOWN


def mv_not(x1 : np.ndarray, out=None):
    """A multi-valued NOT operator.

    :param x1: A multi-valued array.
    :param out: An optional storage destination. If None, a new multi-valued array is returned.
    :return: A multi-valued array with the result.
    """
    out = out or np.empty(x1.shape, dtype=np.uint8)
    _mv_not(out, x1)
    return out


def _mv_or(out, *ins):
    any_unknown = (ins[0] == UNKNOWN) | (ins[0] == UNASSIGNED)
    for inp in ins[1:]: any_unknown |= (inp == UNKNOWN) | (inp == UNASSIGNED)
    any_one = (ins[0] == ONE)
    for inp in ins[1:]: any_one |= (inp == ONE)

    out[...] = ZERO
    np.putmask(out, any_one, ONE)
    for inp in ins:
        np.bitwise_or(out, inp, out=out, where=~any_one)
    np.putmask(out, (any_unknown & ~any_one), UNKNOWN)


def mv_or(x1, x2, out=None):
    """A multi-valued OR operator.

    :param x1: A multi-valued array.
    :param x2: A multi-valued array.
    :param out: An optional storage destination. If None, a new multi-valued array is returned.
    :return: A multi-valued array with the result.
    """
    out = out or np.empty(np.broadcast(x1, x2).shape, dtype=np.uint8)
    _mv_or(out, x1, x2)
    return out


def _mv_and(out, *ins):
    any_unknown = (ins[0] == UNKNOWN) | (ins[0] == UNASSIGNED)
    for inp in ins[1:]: any_unknown |= (inp == UNKNOWN) | (inp == UNASSIGNED)
    any_zero = (ins[0] == ZERO)
    for inp in ins[1:]: any_zero |= (inp == ZERO)

    out[...] = ONE
    np.putmask(out, any_zero, ZERO)
    for inp in ins:
        np.bitwise_and(out, inp | 0b100, out=out, where=~any_zero)
        np.bitwise_or(out, inp & 0b100, out=out, where=~any_zero)
    np.putmask(out, (any_unknown & ~any_zero), UNKNOWN)


def mv_and(x1, x2, out=None):
    """A multi-valued AND operator.

    :param x1: A multi-valued array.
    :param x2: A multi-valued array.
    :param out: An optional storage destination. If None, a new multi-valued array is returned.
    :return: A multi-valued array with the result.
    """
    out = out or np.empty(np.broadcast(x1, x2).shape, dtype=np.uint8)
    _mv_and(out, x1, x2)
    return out


def _mv_xor(out, *ins):
    any_unknown = (ins[0] == UNKNOWN) | (ins[0] == UNASSIGNED)
    for inp in ins[1:]: any_unknown |= (inp == UNKNOWN) | (inp == UNASSIGNED)

    out[...] = ZERO
    for inp in ins:
        np.bitwise_xor(out, inp & 0b011, out=out)
        np.bitwise_or(out, inp & 0b100, out=out)
    np.putmask(out, any_unknown, UNKNOWN)


def mv_xor(x1, x2, out=None):
    """A multi-valued XOR operator.

    :param x1: A multi-valued array.
    :param x2: A multi-valued array.
    :param out: An optional storage destination. If None, a new multi-valued array is returned.
    :return: A multi-valued array with the result.
    """
    out = out or np.empty(np.broadcast(x1, x2).shape, dtype=np.uint8)
    _mv_xor(out, x1, x2)
    return out


def mv_latch(d, t, q_prev, out=None):
    """A multi-valued latch operator.

    A latch outputs ``d`` when transparent (``t`` is high).
    It outputs ``q_prev`` when in latched state (``t`` is low).

    :param d: A multi-valued array for the data input.
    :param t: A multi-valued array for the control input.
    :param q_prev: A multi-valued array with the output value of this latch from the previous clock cycle.
    :param out: An optional storage destination. If None, a new multi-valued array is returned.
    :return: A multi-valued array for the latch output ``q``.
    """
    out = out or np.empty(np.broadcast(d, t, q_prev).shape, dtype=np.uint8)
    out[...] = t & d & 0b011
    out[...] |= ~t & 0b010 & (q_prev << 1)
    out[...] |= ~t & 0b001 & (out >> 1)
    out[...] |= ((out << 1) ^ (out << 2)) & 0b100
    unknown = (t == UNKNOWN) \
              | (t == UNASSIGNED) \
              | (((d == UNKNOWN) | (d == UNASSIGNED)) & (t != ZERO))
    np.putmask(out, unknown, UNKNOWN)
    return out


def mv_transition(init, final, out=None):
    """Computes the logic transitions from the initial values of ``init`` to the final values of ``final``.
    Pulses in the input data are ignored. If any of the inputs are ``UNKNOWN``, the result is ``UNKNOWN``.
    If both inputs are ``UNASSIGNED``, the result is ``UNASSIGNED``.

    :param init: A multi-valued array.
    :param final: A multi-valued array.
    :param out: An optional storage destination. If None, a new multi-valued array is returned.
    :return: A multi-valued array with the result.
    """
    out = out or np.empty(np.broadcast(init, final).shape, dtype=np.uint8)
    out[...] = (init & 0b010) | (final & 0b001)
    out[...] |= ((out << 1) ^ (out << 2)) & 0b100
    unknown = (init == UNKNOWN) | (init == UNASSIGNED) | (final == UNKNOWN) | (final == UNASSIGNED)
    unassigned = (init == UNASSIGNED) & (final == UNASSIGNED)
    np.putmask(out, unknown, UNKNOWN)
    np.putmask(out, unassigned, UNASSIGNED)
    return out


def mv_to_bp(mva):
    """Converts a multi-valued array into a bit-parallel array.
    """
    if mva.ndim == 1: mva = mva[..., np.newaxis]
    return np.packbits(unpackbits(mva)[...,:3], axis=-2, bitorder='little').swapaxes(-1,-2)


def bparray(*a):
    """Converts (lists of) Boolean values or strings into a bit-parallel array.

    The given values are interpreted and the axes are arranged as per KyuPy's convention.
    Use this function to convert strings into bit-parallel arrays.
    """
    return mv_to_bp(mvarray(*a))


def bp_to_mv(bpa):
    """Converts a bit-parallel array into a multi-valued array.
    """
    return packbits(np.unpackbits(bpa, axis=-1, bitorder='little').swapaxes(-1,-2))


def bp4v_buf(out, inp):
    unknown = inp[..., 0, :] ^ inp[..., 1, :]
    out[..., 0, :] = inp[..., 0, :] | unknown
    out[..., 1, :] = inp[..., 1, :] & ~unknown
    return out


def bp8v_buf(out, inp):
    unknown = (inp[..., 0, :] ^ inp[..., 1, :]) & ~inp[..., 2, :]
    out[..., 0, :] = inp[..., 0, :] | unknown
    out[..., 1, :] = inp[..., 1, :] & ~unknown
    out[..., 2, :] = inp[..., 2, :] & ~unknown
    return out


def bp4v_not(out, inp):
    unknown = inp[..., 0, :] ^ inp[..., 1, :]
    out[..., 0, :] = ~inp[..., 0, :] | unknown
    out[..., 1, :] = ~inp[..., 1, :] & ~unknown
    return out


def bp8v_not(out, inp):
    unknown = (inp[..., 0, :] ^ inp[..., 1, :]) & ~inp[..., 2, :]
    out[..., 0, :] = ~inp[..., 0, :] | unknown
    out[..., 1, :] = ~inp[..., 1, :] & ~unknown
    out[..., 2, :] = inp[..., 2, :] & ~unknown
    return out


def bp4v_or(out, *ins):
    out[...] = 0
    any_unknown = ins[0][..., 0, :] ^ ins[0][..., 1, :]
    for inp in ins[1:]: any_unknown |= inp[..., 0, :] ^ inp[..., 1, :]
    any_one = ins[0][..., 0, :] & ins[0][..., 1, :]
    for inp in ins[1:]: any_one |= inp[..., 0, :] & inp[..., 1, :]
    for inp in ins:
        out[..., 0, :] |= inp[..., 0, :] | any_unknown
        out[..., 1, :] |= inp[..., 1, :] & (~any_unknown | any_one)
    return out


def bp8v_or(out, *ins):
    out[...] = 0
    any_unknown = (ins[0][..., 0, :] ^ ins[0][..., 1, :]) & ~ins[0][..., 2, :]
    for inp in ins[1:]: any_unknown |= (inp[..., 0, :] ^ inp[..., 1, :]) & ~inp[..., 2, :]
    any_one = ins[0][..., 0, :] & ins[0][..., 1, :] & ~ins[0][..., 2, :]
    for inp in ins[1:]: any_one |= inp[..., 0, :] & inp[..., 1, :] & ~inp[..., 2, :]
    for inp in ins:
        out[..., 0, :] |= inp[..., 0, :] | any_unknown
        out[..., 1, :] |= inp[..., 1, :] & (~any_unknown | any_one)
        out[..., 2, :] |= inp[..., 2, :] & (~any_unknown | any_one) & ~any_one
    return out


def bp4v_and(out, *ins):
    out[...] = 0xff
    any_unknown = ins[0][..., 0, :] ^ ins[0][..., 1, :]
    for inp in ins[1:]: any_unknown |= inp[..., 0, :] ^ inp[..., 1, :]
    any_zero = ~ins[0][..., 0, :] & ~ins[0][..., 1, :]
    for inp in ins[1:]: any_zero |= ~inp[..., 0, :] & ~inp[..., 1, :]
    for inp in ins:
        out[..., 0, :] &= inp[..., 0, :] | (any_unknown & ~any_zero)
        out[..., 1, :] &= inp[..., 1, :] & ~any_unknown
    return out


def bp8v_and(out, *ins):
    out[...] = 0xff
    any_unknown = (ins[0][..., 0, :] ^ ins[0][..., 1, :]) & ~ins[0][..., 2, :]
    for inp in ins[1:]: any_unknown |= (inp[..., 0, :] ^ inp[..., 1, :]) & ~inp[..., 2, :]
    any_zero = ~ins[0][..., 0, :] & ~ins[0][..., 1, :] & ~ins[0][..., 2, :]
    for inp in ins[1:]: any_zero |= ~inp[..., 0, :] & ~inp[..., 1, :] & ~inp[..., 2, :]
    out[..., 2, :] = 0
    for inp in ins:
        out[..., 0, :] &= inp[..., 0, :] | (any_unknown & ~any_zero)
        out[..., 1, :] &= inp[..., 1, :] & ~any_unknown
        out[..., 2, :] |= inp[..., 2, :] & (~any_unknown | any_zero) & ~any_zero
    return out


def bp4v_xor(out, *ins):
    out[...] = 0
    any_unknown = ins[0][..., 0, :] ^ ins[0][..., 1, :]
    for inp in ins[1:]: any_unknown |= inp[..., 0, :] ^ inp[..., 1, :]
    for inp in ins:
        out[..., 0, :] ^= inp[..., 0, :]
        out[..., 1, :] ^= inp[..., 1, :]
    out[..., 0, :] |= any_unknown
    out[..., 1, :] &= ~any_unknown
    return out


def bp8v_xor(out, *ins):
    out[...] = 0
    any_unknown = (ins[0][..., 0, :] ^ ins[0][..., 1, :]) & ~ins[0][..., 2, :]
    for inp in ins[1:]: any_unknown |= (inp[..., 0, :] ^ inp[..., 1, :]) & ~inp[..., 2, :]
    for inp in ins:
        out[..., 0, :] ^= inp[..., 0, :]
        out[..., 1, :] ^= inp[..., 1, :]
        out[..., 2, :] |= inp[..., 2, :]
    out[..., 0, :] |= any_unknown
    out[..., 1, :] &= ~any_unknown
    out[..., 2, :] &= ~any_unknown
    return out


def bp8v_latch(out, d, t, q_prev):
    any_unknown = (t[..., 0, :] ^ t[..., 1, :]) & ~t[..., 2, :]
    any_unknown |= ((d[..., 0, :] ^ d[..., 1, :]) & ~d[..., 2, :]) & (t[..., 0, :] | t[..., 1, :] | t[..., 2, :])
    out[..., 1, :] = (d[..., 1, :] & t[..., 1, :]) | (q_prev[..., 0, :] & ~t[..., 1, :])
    out[..., 0, :] = (d[..., 0, :] & t[..., 0, :]) | (out[..., 1, :] & ~t[..., 0, :])
    out[..., 2, :] = out[..., 1, :] ^ out[..., 0, :]
    out[..., 0, :] |= any_unknown
    out[..., 1, :] &= ~any_unknown
    out[..., 2, :] &= ~any_unknown
    return out


_bit_in_lut = np.array([2 ** x for x in range(7, -1, -1)], dtype='uint8')


@numba.njit
def bit_in(a, pos):
    return a[pos >> 3] & _bit_in_lut[pos & 7]


def unpackbits(a : np.ndarray):
    """Unpacks the bits of given ndarray ``a``.

    Similar to ``np.unpackbits``, but accepts any dtype, preserves the shape of ``a`` and
    adds a new last axis with the bits of each item. Bits are in 'little'-order, i.e.,
    a[...,0] is the least significant bit of each item.
    """
    return np.unpackbits(a.view(np.uint8), bitorder='little').reshape(*a.shape, 8*a.itemsize)


def packbits(a, dtype=np.uint8):
    """Packs the values of a boolean-valued array ``a`` along its last axis into bits.

    Similar to ``np.packbits``, but returns an array of given dtype and the shape of ``a`` with the last axis removed.
    The last axis of `a` is truncated or padded according to the bit-width of the given dtype.
    Signed integer datatypes are padded with the most significant bit, all others are padded with `0`.
    """
    dtype = np.dtype(dtype)
    bits = 8 * dtype.itemsize
    a = a[...,:bits]
    if a.shape[-1] < bits:
        p = [(0,0)]*(len(a.shape)-1) + [(0, bits-a.shape[-1])]
        a = np.pad(a, p, 'edge') if dtype.name[0] == 'i' else np.pad(a, p, 'constant', constant_values=0)
    return np.packbits(a, bitorder='little').view(dtype).reshape(a.shape[:-1])
