"""This module contains definitions and data structures for 2-, 4-, and 8-valued logic operations.

8 logic values are defined as integer constants.

* For 2-valued logic: ``ZERO`` and ``ONE``
* 4-valued logic adds: ``UNASSIGNED`` and ``UNKNOWN``
* 8-valued logic adds: ``RISE``, ``FALL``, ``PPULSE``, and ``NPULSE``.

The bits in these constants have the following meaning:

  * bit 0: Final/settled binary value of a signal
  * bit 1: Initial binary value of a signal
  * bit 2: Activity or transitions are present on a signal

Special meaning is given to values where bits 0 and 1 differ, but bit 2 (activity) is 0.
These values are interpreted as ``UNKNOWN`` or ``UNASSIGNED`` in 4-valued and 8-valued logic.

In general, 2-valued logic only considers bit 0, 4-valued logic considers bits 0 and 1, and 8-valued logic
considers all 3 bits.
The only exception is constant ``ONE=0b11`` which has two bits set for all logics including 2-valued logic.
"""

import math
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
    if value in [0, '0', False, 'L', 'l']:
        return ZERO
    if value in [1, '1', True, 'H', 'h']:
        return ONE
    if value in [None, '-', 'Z', 'z']:
        return UNASSIGNED
    if value in ['R', 'r', '/']:
        return RISE
    if value in ['F', 'f', '\\']:
        return FALL
    if value in ['P', 'p', '^']:
        return PPULSE
    if value in ['N', 'n', 'v']:
        return NPULSE
    return UNKNOWN


_bit_in_lut = np.array([2 ** x for x in range(7, -1, -1)], dtype='uint8')


@numba.njit
def bit_in(a, pos):
    return a[pos >> 3] & _bit_in_lut[pos & 7]


class MVArray:
    """An n-dimensional array of m-valued logic values.

    This class wraps a numpy.ndarray of type uint8 and adds support for encoding and
    interpreting 2-valued, 4-valued, and 8-valued logic values.
    Each logic value is stored as an uint8, manipulations of individual values are cheaper than in
    :py:class:`BPArray`.

    :param a: If a tuple is given, it is interpreted as desired shape. To make an array of ``n`` vectors
        compatible with a simulator ``sim``, use ``(len(sim.interface), n)``. If a :py:class:`BPArray` or
        :py:class:`MVArray` is given, a deep copy is made. If a string, a list of strings, a list of characters,
        or a list of lists of characters are given, the data is interpreted best-effort and the array is
        initialized accordingly.
    :param m: The arity of the logic. Can be set to 2, 4, or 8. If None is given, the arity of a given
        :py:class:`BPArray` or :py:class:`MVArray` is used, or, if the array is initialized differently, 8 is used.
    """

    def __init__(self, a, m=None):
        self.m = m or 8
        assert self.m in [2, 4, 8]

        # Try our best to interpret given a.
        if isinstance(a, MVArray):
            self.data = a.data.copy()
            """The wrapped 2-dimensional ndarray of logic values.

            * Axis 0 is PI/PO/FF position, the length of this axis is called "width".
            * Axis 1 is vector/pattern, the length of this axis is called "length".
            """
            self.m = m or a.m
        elif hasattr(a, 'data'):  # assume it is a BPArray. Can't use isinstance() because BPArray isn't declared yet.
            self.data = np.zeros((a.width, a.length), dtype=np.uint8)
            self.m = m or a.m
            for i in range(a.data.shape[-2]):
                self.data[...] <<= 1
                self.data[...] |= np.unpackbits(a.data[..., -i-1, :], axis=1)[:, :a.length]
            if a.data.shape[-2] == 1:
                self.data *= 3
        elif isinstance(a, int):
            self.data = np.full((a, 1), UNASSIGNED, dtype=np.uint8)
        elif isinstance(a, tuple):
            self.data = np.full(a, UNASSIGNED, dtype=np.uint8)
        else:
            if isinstance(a, str): a = [a]
            self.data = np.asarray(interpret(a), dtype=np.uint8)
            self.data = self.data[:, np.newaxis] if self.data.ndim == 1 else np.moveaxis(self.data, -2, -1)

        # Cast data to m-valued logic.
        if self.m == 2:
            self.data[...] = ((self.data & 0b001) & ((self.data >> 1) & 0b001) | (self.data == RISE)) * ONE
        elif self.m == 4:
            self.data[...] = (self.data & 0b011) & ((self.data != FALL) * ONE) | ((self.data == RISE) * ONE)
        elif self.m == 8:
            self.data[...] = self.data & 0b111

        self.length = self.data.shape[-1]
        self.width = self.data.shape[-2]

    def __repr__(self):
        return f'<MVArray length={self.length} width={self.width} m={self.m} mem={hr_bytes(self.data.nbytes)}>'

    def __str__(self):
        return str([self[idx] for idx in range(self.length)])

    def __getitem__(self, vector_idx):
        """Returns a string representing the desired vector."""
        chars = ["0", "X", "-", "1", "P", "R", "F", "N"]
        return ''.join(chars[v] for v in self.data[:, vector_idx])

    def __len__(self):
        return self.length


def mv_cast(*args, m=8):
    return [a if isinstance(a, MVArray) else MVArray(a, m=m) for a in args]


def mv_getm(*args):
    return max([a.m for a in args if isinstance(a, MVArray)] + [0]) or 8


def _mv_not(m, out, inp):
    np.bitwise_xor(inp, 0b11, out=out)  # this also exchanges UNASSIGNED <-> UNKNOWN
    if m > 2:
        np.putmask(out, (inp == UNKNOWN), UNKNOWN)  # restore UNKNOWN


def mv_not(x1, out=None):
    """A multi-valued NOT operator.

    :param x1: An :py:class:`MVArray` or data the :py:class:`MVArray` constructor accepts.
    :param out: Optionally an :py:class:`MVArray` as storage destination. If None, a new :py:class:`MVArray`
        is returned.
    :return: An :py:class:`MVArray` with the result.
    """
    m = mv_getm(x1)
    x1 = mv_cast(x1, m=m)[0]
    out = out or MVArray(x1.data.shape, m=m)
    _mv_not(m, out.data, x1.data)
    return out


def _mv_or(m, out, *ins):
    if m > 2:
        any_unknown = (ins[0] == UNKNOWN) | (ins[0] == UNASSIGNED)
        for inp in ins[1:]: any_unknown |= (inp == UNKNOWN) | (inp == UNASSIGNED)
        any_one = (ins[0] == ONE)
        for inp in ins[1:]: any_one |= (inp == ONE)

        out[...] = ZERO
        np.putmask(out, any_one, ONE)
        for inp in ins:
            np.bitwise_or(out, inp, out=out, where=~any_one)
        np.putmask(out, (any_unknown & ~any_one), UNKNOWN)
    else:
        out[...] = ZERO
        for inp in ins: np.bitwise_or(out, inp, out=out)


def mv_or(x1, x2, out=None):
    """A multi-valued OR operator.

    :param x1: An :py:class:`MVArray` or data the :py:class:`MVArray` constructor accepts.
    :param x2: An :py:class:`MVArray` or data the :py:class:`MVArray` constructor accepts.
    :param out: Optionally an :py:class:`MVArray` as storage destination. If None, a new :py:class:`MVArray`
        is returned.
    :return: An :py:class:`MVArray` with the result.
    """
    m = mv_getm(x1, x2)
    x1, x2 = mv_cast(x1, x2, m=m)
    out = out or MVArray(np.broadcast(x1.data, x2.data).shape, m=m)
    _mv_or(m, out.data, x1.data, x2.data)
    return out


def _mv_and(m, out, *ins):
    if m > 2:
        any_unknown = (ins[0] == UNKNOWN) | (ins[0] == UNASSIGNED)
        for inp in ins[1:]: any_unknown |= (inp == UNKNOWN) | (inp == UNASSIGNED)
        any_zero = (ins[0] == ZERO)
        for inp in ins[1:]: any_zero |= (inp == ZERO)

        out[...] = ONE
        np.putmask(out, any_zero, ZERO)
        for inp in ins:
            np.bitwise_and(out, inp | 0b100, out=out, where=~any_zero)
            if m > 4: np.bitwise_or(out, inp & 0b100, out=out, where=~any_zero)
        np.putmask(out, (any_unknown & ~any_zero), UNKNOWN)
    else:
        out[...] = ONE
        for inp in ins: np.bitwise_and(out, inp, out=out)


def mv_and(x1, x2, out=None):
    """A multi-valued AND operator.

    :param x1: An :py:class:`MVArray` or data the :py:class:`MVArray` constructor accepts.
    :param x2: An :py:class:`MVArray` or data the :py:class:`MVArray` constructor accepts.
    :param out: Optionally an :py:class:`MVArray` as storage destination. If None, a new :py:class:`MVArray`
        is returned.
    :return: An :py:class:`MVArray` with the result.
    """
    m = mv_getm(x1, x2)
    x1, x2 = mv_cast(x1, x2, m=m)
    out = out or MVArray(np.broadcast(x1.data, x2.data).shape, m=m)
    _mv_and(m, out.data, x1.data, x2.data)
    return out


def _mv_xor(m, out, *ins):
    if m > 2:
        any_unknown = (ins[0] == UNKNOWN) | (ins[0] == UNASSIGNED)
        for inp in ins[1:]: any_unknown |= (inp == UNKNOWN) | (inp == UNASSIGNED)

        out[...] = ZERO
        for inp in ins:
            np.bitwise_xor(out, inp & 0b011, out=out)
            if m > 4: np.bitwise_or(out, inp & 0b100, out=out)
        np.putmask(out, any_unknown, UNKNOWN)
    else:
        out[...] = ZERO
        for inp in ins: np.bitwise_xor(out, inp, out=out)


def mv_xor(x1, x2, out=None):
    """A multi-valued XOR operator.

    :param x1: An :py:class:`MVArray` or data the :py:class:`MVArray` constructor accepts.
    :param x2: An :py:class:`MVArray` or data the :py:class:`MVArray` constructor accepts.
    :param out: Optionally an :py:class:`MVArray` as storage destination. If None, a new :py:class:`MVArray`
        is returned.
    :return: An :py:class:`MVArray` with the result.
    """
    m = mv_getm(x1, x2)
    x1, x2 = mv_cast(x1, x2, m=m)
    out = out or MVArray(np.broadcast(x1.data, x2.data).shape, m=m)
    _mv_xor(m, out.data, x1.data, x2.data)
    return out


def mv_transition(init, final, out=None):
    """Computes the logic transitions from the initial values of ``init`` to the final values of ``final``.
    Pulses in the input data are ignored. If any of the inputs are ``UNKNOWN``, the result is ``UNKNOWN``.
    If both inputs are ``UNASSIGNED``, the result is ``UNASSIGNED``.

    :param init: An :py:class:`MVArray` or data the :py:class:`MVArray` constructor accepts.
    :param final: An :py:class:`MVArray` or data the :py:class:`MVArray` constructor accepts.
    :param out: Optionally an :py:class:`MVArray` as storage destination. If None, a new :py:class:`MVArray`
        is returned.
    :return: An :py:class:`MVArray` with the result.
    """
    m = mv_getm(init, final)
    init, final = mv_cast(init, final, m=m)
    init = init.data
    final = final.data
    out = out or MVArray(np.broadcast(init, final).shape, m=8)
    out.data[...] = (init & 0b010) | (final & 0b001)
    out.data[...] |= ((out.data << 1) ^ (out.data << 2)) & 0b100
    unknown = (init == UNKNOWN) | (init == UNASSIGNED) | (final == UNKNOWN) | (final == UNASSIGNED)
    unassigned = (init == UNASSIGNED) & (final == UNASSIGNED)
    np.putmask(out.data, unknown, UNKNOWN)
    np.putmask(out.data, unassigned, UNASSIGNED)
    return out


class BPArray:
    """An n-dimensional array of m-valued logic values that uses bit-parallel storage.

    The primary use of this format is in aiding efficient bit-parallel logic simulation.
    The secondary benefit over :py:class:`MVArray` is its memory efficiency.
    Accessing individual values is more expensive than with :py:class:`MVArray`.
    Therefore it may be more efficient to unpack the data into an :py:class:`MVArray` and pack it again into a
    :py:class:`BPArray` for simulation.

    See :py:class:`MVArray` for constructor parameters.
    """

    def __init__(self, a, m=None):
        if not isinstance(a, MVArray) and not isinstance(a, BPArray):
            a = MVArray(a, m)
            self.m = a.m
        if isinstance(a, MVArray):
            if m is not None and m != a.m:
                a = MVArray(a, m)  # cast data
            self.m = a.m
            assert self.m in [2, 4, 8]
            nwords = math.ceil(math.log2(self.m))
            nbytes = (a.data.shape[-1] - 1) // 8 + 1
            self.data = np.zeros(a.data.shape[:-1] + (nwords, nbytes), dtype=np.uint8)
            """The wrapped 3-dimensional ndarray.

            * Axis 0 is PI/PO/FF position, the length of this axis is called "width".
            * Axis 1 has length ``ceil(log2(m))`` for storing all bits.
            * Axis 2 are the vectors/patterns packed into uint8 words.
            """
            for i in range(self.data.shape[-2]):
                self.data[..., i, :] = np.packbits((a.data >> i) & 1, axis=-1)
        else:  # we have a BPArray
            self.data = a.data.copy()  # TODO: support conversion to different m
            self.m = a.m
        self.length = a.length
        self.width = a.width

    def __repr__(self):
        return f'<BPArray length={self.length} width={self.width} m={self.m} mem={hr_bytes(self.data.nbytes)}>'

    def __len__(self):
        return self.length


def bp_buf(out, inp):
    md = out.shape[-2]
    assert md == inp.shape[-2]
    if md > 1:
        unknown = inp[..., 0, :] ^ inp[..., 1, :]
        if md > 2: unknown &= ~inp[..., 2, :]
        out[..., 0, :] = inp[..., 0, :] | unknown
        out[..., 1, :] = inp[..., 1, :] & ~unknown
        if md > 2: out[..., 2, :] = inp[..., 2, :] & ~unknown
    else:
        out[..., 0, :] = inp[..., 0, :]


def bp_not(out, inp):
    md = out.shape[-2]
    assert md == inp.shape[-2]
    if md > 1:
        unknown = inp[..., 0, :] ^ inp[..., 1, :]
        if md > 2: unknown &= ~inp[..., 2, :]
        out[..., 0, :] = ~inp[..., 0, :] | unknown
        out[..., 1, :] = ~inp[..., 1, :] & ~unknown
        if md > 2: out[..., 2, :] = inp[..., 2, :] & ~unknown
    else:
        out[..., 0, :] = ~inp[..., 0, :]


def bp_or(out, *ins):
    md = out.shape[-2]
    for inp in ins: assert md == inp.shape[-2]
    out[...] = 0
    if md == 1:
        for inp in ins: out[..., 0, :] |= inp[..., 0, :]
    elif md == 2:
        any_unknown = ins[0][..., 0, :] ^ ins[0][..., 1, :]
        for inp in ins[1:]: any_unknown |= inp[..., 0, :] ^ inp[..., 1, :]
        any_one = ins[0][..., 0, :] & ins[0][..., 1, :]
        for inp in ins[1:]: any_one |= inp[..., 0, :] & inp[..., 1, :]
        for inp in ins:
            out[..., 0, :] |= inp[..., 0, :] | any_unknown
            out[..., 1, :] |= inp[..., 1, :] & (~any_unknown | any_one)
    else:
        any_unknown = (ins[0][..., 0, :] ^ ins[0][..., 1, :]) & ~ins[0][..., 2, :]
        for inp in ins[1:]: any_unknown |= (inp[..., 0, :] ^ inp[..., 1, :]) & ~inp[..., 2, :]
        any_one = ins[0][..., 0, :] & ins[0][..., 1, :] & ~ins[0][..., 2, :]
        for inp in ins[1:]: any_one |= inp[..., 0, :] & inp[..., 1, :] & ~inp[..., 2, :]
        for inp in ins:
            out[..., 0, :] |= inp[..., 0, :] | any_unknown
            out[..., 1, :] |= inp[..., 1, :] & (~any_unknown | any_one)
            out[..., 2, :] |= inp[..., 2, :] & (~any_unknown | any_one) & ~any_one


def bp_and(out, *ins):
    md = out.shape[-2]
    for inp in ins: assert md == inp.shape[-2]
    out[...] = 0xff
    if md == 1:
        for inp in ins: out[..., 0, :] &= inp[..., 0, :]
    elif md == 2:
        any_unknown = ins[0][..., 0, :] ^ ins[0][..., 1, :]
        for inp in ins[1:]: any_unknown |= inp[..., 0, :] ^ inp[..., 1, :]
        any_zero = ~ins[0][..., 0, :] & ~ins[0][..., 1, :]
        for inp in ins[1:]: any_zero |= ~inp[..., 0, :] & ~inp[..., 1, :]
        for inp in ins:
            out[..., 0, :] &= inp[..., 0, :] | (any_unknown & ~any_zero)
            out[..., 1, :] &= inp[..., 1, :] & ~any_unknown
    else:
        any_unknown = (ins[0][..., 0, :] ^ ins[0][..., 1, :]) & ~ins[0][..., 2, :]
        for inp in ins[1:]: any_unknown |= (inp[..., 0, :] ^ inp[..., 1, :]) & ~inp[..., 2, :]
        any_zero = ~ins[0][..., 0, :] & ~ins[0][..., 1, :] & ~ins[0][..., 2, :]
        for inp in ins[1:]: any_zero |= ~inp[..., 0, :] & ~inp[..., 1, :] & ~inp[..., 2, :]
        out[..., 2, :] = 0
        for inp in ins:
            out[..., 0, :] &= inp[..., 0, :] | (any_unknown & ~any_zero)
            out[..., 1, :] &= inp[..., 1, :] & ~any_unknown
            out[..., 2, :] |= inp[..., 2, :] & (~any_unknown | any_zero) & ~any_zero


def bp_xor(out, *ins):
    md = out.shape[-2]
    for inp in ins: assert md == inp.shape[-2]
    out[...] = 0
    if md == 1:
        for inp in ins: out[..., 0, :] ^= inp[..., 0, :]
    elif md == 2:
        any_unknown = ins[0][..., 0, :] ^ ins[0][..., 1, :]
        for inp in ins[1:]: any_unknown |= inp[..., 0, :] ^ inp[..., 1, :]
        for inp in ins: out[...] ^= inp
        out[..., 0, :] |= any_unknown
        out[..., 1, :] &= ~any_unknown
    else:
        any_unknown = (ins[0][..., 0, :] ^ ins[0][..., 1, :]) & ~ins[0][..., 2, :]
        for inp in ins[1:]: any_unknown |= (inp[..., 0, :] ^ inp[..., 1, :]) & ~inp[..., 2, :]
        for inp in ins:
            out[..., 0, :] ^= inp[..., 0, :]
            out[..., 1, :] ^= inp[..., 1, :]
            out[..., 2, :] |= inp[..., 2, :]
        out[..., 0, :] |= any_unknown
        out[..., 1, :] &= ~any_unknown
        out[..., 2, :] &= ~any_unknown
