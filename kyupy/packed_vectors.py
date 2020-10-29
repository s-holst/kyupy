import numpy as np
from .bittools import popcount, bit_in


class PackedVectors:
    def __init__(self, nvectors=8, width=1, vdim=1, from_cache=None):
        if from_cache is not None:
            self.bits = np.array(from_cache)
            self.width, self.vdim, nbytes = self.bits.shape
        else:
            self.bits = np.zeros((width, vdim, (nvectors - 1) // 8 + 1), dtype='uint8')
            self.vdim = vdim
            self.width = width
        self.nvectors = nvectors
        m1 = np.array([2 ** x for x in range(7, -1, -1)], dtype='uint8')
        m0 = ~m1
        self.mask = np.rollaxis(np.vstack((m0, m1)), 1)

    @classmethod
    def from_pair(cls, init, final):
        assert init.nvectors == final.nvectors
        assert len(init.bits) == len(final.bits)
        init_v = init.bits[:, 0]
        if init.vdim == 3:
            init_c = (init.bits[:, 0] ^ init.bits[:, 1]) | init.bits[:, 2]
        elif init.vdim == 2:
            init_c = init.bits[:, 1]
        else:
            init_c = ~np.zeros_like(init.bits[:, 0])
        final_v = final.bits[:, 0]
        if final.vdim == 3:
            final_c = (final.bits[:, 0] ^ final.bits[:, 1]) | final.bits[:, 2]
            final_v = ~final.bits[:, 1]
        elif final.vdim == 2:
            final_c = final.bits[:, 1]
        else:
            final_c = ~np.zeros_like(final.bits[:, 0])
        c = init_c & final_c
        a0 = init_v & c
        a1 = ~final_v & c
        a2 = (init_v ^ final_v) & c
        p = PackedVectors(init.nvectors, len(init.bits), 3)
        p.bits[:, 0] = a0
        p.bits[:, 1] = a1
        p.bits[:, 2] = a2
        return p
        
    def transition_vectors(self):
        a = PackedVectors(self.nvectors-1, self.width, 3)
        for pos in range(self.width):
            for vidx in range(self.nvectors-1):
                tr = self.get_value(vidx, pos) + self.get_value(vidx+1, pos)
                if tr == '00':
                    a.set_value(vidx, pos, '0')
                elif tr == '11':
                    a.set_value(vidx, pos, '1')
                elif tr == '01':
                    a.set_value(vidx, pos, 'R')
                elif tr == '10':
                    a.set_value(vidx, pos, 'F')
                elif tr == '--':
                    a.set_value(vidx, pos, '-')
                else:
                    a.set_value(vidx, pos, 'X')
        return a
        
    def __add__(self, other):
        a = PackedVectors(self.nvectors + other.nvectors, self.width, max(self.vdim, other.vdim))
        # a.bits[:self.bits.shape[0], 0] = self.bits[:, 0]
        # if self.vdim == 2:
        #    a.bits[:self.bits.shape[0], 1] = self.care_bits
        # elif self.vdim == 3:
        #    a.bits[:self.bits.shape[0], 1] = ~self.value_bits
        #    a.bits[:self.bits.shape[0], 2] = self.toggle_bits
        for i in range(self.nvectors):
            a[i] = self[i]
        for i in range(len(other)):
            a[self.nvectors+i] = other[i]
        return a

    def __len__(self):
        return self.nvectors
    
    def randomize(self, one_probability=0.5):
        for data in self.bits:
            data[0] = np.packbits((np.random.rand(self.nvectors) < one_probability).astype(int))
            if self.vdim == 2:
                data[1] = 255
            elif self.vdim == 3:
                data[1] = ~np.packbits((np.random.rand(self.nvectors) < one_probability).astype(int))
                data[2] = data[0] ^ ~data[1]
            
    def copy(self, selection_mask=None):
        if selection_mask is not None:
            cpy = PackedVectors(popcount(selection_mask), len(self.bits), self.vdim)
            cur = 0
            for vidx in range(self.nvectors):
                if bit_in(selection_mask, vidx):
                    cpy[cur] = self[vidx]
                    cur += 1
        else:
            cpy = PackedVectors(self.nvectors, len(self.bits), self.vdim)
            np.copyto(cpy.bits, self.bits)
        return cpy

    @property
    def care_bits(self):
        if self.vdim == 1:
            return self.bits[:, 0] | 255
        elif self.vdim == 2:
            return self.bits[:, 1]
        elif self.vdim == 3:
            return (self.bits[:, 0] ^ self.bits[:, 1]) | self.bits[:, 2]

    @property
    def initial_bits(self):
        return self.bits[:, 0]

    @property
    def value_bits(self):
        if self.vdim == 3:
            return ~self.bits[:, 1]
        else:
            return self.bits[:, 0]

    @property
    def toggle_bits(self):
        if self.vdim == 3:
            return self.bits[:, 2]
        else:
            return self.bits[:, 0] & 0

    def get_value(self, vector, position):
        if vector >= self.nvectors:
            raise IndexError(f'vector out of range: {vector} >= {self.nvectors}')
        a = self.bits[position, :, vector // 8]
        m = self.mask[vector % 8]
        if self.vdim == 1:
            return '1' if a[0] & m[1] else '0'
        elif self.vdim == 2:
            if a[0] & m[1]:
                return '1' if a[1] & m[1] else 'X'
            else:
                return '0' if a[1] & m[1] else '-'
        elif self.vdim == 3:
            if a[2] & m[1]:
                if a[0] & m[1]:
                    return 'F' if a[1] & m[1] else 'N'
                else:
                    return 'P' if a[1] & m[1] else 'R'
            else:
                if a[0] & m[1]:
                    return 'X' if a[1] & m[1] else '1'
                else:
                    return '0' if a[1] & m[1] else '-'                

    def get_values_for_position(self, position):
        return ''.join(self.get_value(x, position) for x in range(self.nvectors))

    def set_value(self, vector, position, v):
        if vector >= self.nvectors:
            raise IndexError(f'vector out of range: {vector} >= {self.nvectors}')
        a = self.bits[position, :, vector // 8]
        m = self.mask[vector % 8]
        if self.vdim == 1:
            self._set_value_vd1(a, m, v)
        elif self.vdim == 2:
            self._set_value_vd2(a, m, v)
        elif self.vdim == 3:
            self._set_value_vd3(a, m, v)
    
    def set_values(self, vector, v, mapping=None, inversions=None):
        if vector >= self.nvectors:
            raise IndexError(f'vector out of range: {vector} >= {self.nvectors}')
        if not mapping:
            mapping = [y for y in range(len(v))]
        if inversions is None:
            inversions = [False] * len(v)
        for i, c in enumerate(v):
            if inversions[i]:
                if c == '1':
                    c = '0'
                elif c == '0':
                    c = '1'
                elif c == 'H':
                    c = 'L'
                elif c == 'L':
                    c = 'H'
                elif c == 'R':
                    c = 'F'
                elif c == 'F':
                    c = 'R'
            self.set_value(vector, mapping[i], c)
    
    def set_values_for_position(self, position, values):
        for i, v in enumerate(values):
            self.set_value(i, position, v)
            
    def __setitem__(self, vector, value):
        for i, c in enumerate(value):
            self.set_value(vector, i, c)

    def __getitem__(self, vector):
        if isinstance(vector, slice):
            first = self.get_values_for_position(0)[vector]
            ret = PackedVectors(len(first), self.width, self.vdim)
            ret.set_values_for_position(0, first)
            for pos in range(1, self.width):
                ret.set_values_for_position(pos, self.get_values_for_position(pos)[vector])
            return ret
        return ''.join(self.get_value(vector, pos) for pos in range(len(self.bits)))

    @staticmethod
    def _set_value_vd1(a, m, v):
        if v in [True, 1, '1', 'H', 'h']:
            a[0] |= m[1]
        else:
            a[0] &= m[0]
    
    @staticmethod
    def _set_value_vd2(a, m, v):
        if v in [True, 1, '1', 'H', 'h']:
            a[0] |= m[1]
            a[1] |= m[1]
        elif v in [False, 0, '0', 'L', 'l']:
            a[0] &= m[0]
            a[1] |= m[1]
        elif v in ['X', 'x']:
            a[0] |= m[1]
            a[1] &= m[0]
        else:
            a[0] &= m[0]
            a[1] &= m[0]

    #   i fb act
    # a 0 1 2
    # - 0 0 0  None, '-'
    # 0 0 1 0  False, 0, '0', 'l', 'L'
    # 1 1 0 0  True, 1, '1', 'h', 'H'
    # X 1 1 0  'x', 'X'
    # / 0 0 1  '/', 'r', 'R'
    # ^ 0 1 1  '^', 'p', 'P'
    # v 1 0 1  'v', 'n', 'N'
    # \ 1 1 1  '\', 'f', 'F'
    @staticmethod
    def _set_value_vd3(a, m, v):
        if v in [False, 0, '0', 'L', 'l']:
            a[0] &= m[0]
            a[1] |= m[1]
            a[2] &= m[0]
        elif v in [True, 1, '1', 'H', 'h']:
            a[0] |= m[1]
            a[1] &= m[0]
            a[2] &= m[0]
        elif v in ['X', 'x']:
            a[0] |= m[1]
            a[1] |= m[1]
            a[2] &= m[0]
        elif v in ['/', 'r', 'R']:
            a[0] &= m[0]
            a[1] &= m[0]
            a[2] |= m[1]
        elif v in ['^', 'p', 'P']:
            a[0] &= m[0]
            a[1] |= m[1]
            a[2] |= m[1]
        elif v in ['v', 'n', 'N']:
            a[0] |= m[1]
            a[1] &= m[0]
            a[2] |= m[1]
        elif v in ['\\', 'f', 'F']:
            a[0] |= m[1]
            a[1] |= m[1]
            a[2] |= m[1]
        else:
            a[0] &= m[0]
            a[1] &= m[0]
            a[2] &= m[0]
                                    
    def __repr__(self):
        return f'<PackedVectors nvectors={self.nvectors}, width={self.width}, vdim={self.vdim}>'

    def __str__(self):
        lst = []
        for p in range(self.nvectors):
            lst.append(''.join(self.get_value(p, w) for w in range(len(self.bits))))
        if len(lst) == 0: return ''
        if len(lst[0]) > 64:
            lst = [s[:32] + '...' + s[-32:] for s in lst]
        if len(lst) <= 16:
            return '\n'.join(lst)
        else:
            return '\n'.join(lst[:8]) + '\n...\n' + '\n'.join(lst[-8:])
            
    def diff(self, other, out=None):
        if out is None:
            out = np.zeros((self.width, self.bits.shape[-1]), dtype='uint8')
        out[...] = (self.value_bits ^ other.value_bits) & self.care_bits & other.care_bits
        return out
