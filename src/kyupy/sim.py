
from collections import defaultdict
from bisect import bisect, insort_left

import numpy as np

BUF1 = np.uint16(0b1010_1010_1010_1010)
INV1 = ~BUF1

AND2 = np.uint16(0b1000_1000_1000_1000)
AND3 = np.uint16(0b1000_0000_1000_0000)
AND4 = np.uint16(0b1000_0000_0000_0000)

NAND2, NAND3, NAND4 = ~AND2, ~AND3, ~AND4

OR2 = np.uint16(0b1110_1110_1110_1110)
OR3 = np.uint16(0b1111_1110_1111_1110)
OR4 = np.uint16(0b1111_1111_1111_1110)

NOR2, NOR3, NOR4 = ~OR2, ~OR3, ~OR4

XOR2 = np.uint16(0b0110_0110_0110_0110)
XOR3 = np.uint16(0b1001_0110_1001_0110)
XOR4 = np.uint16(0b0110_1001_1001_0110)

XNOR2, XNOR3, XNOR4 = ~XOR2, ~XOR3, ~XOR4

AO21 = np.uint16(0b1111_1000_1111_1000)  # (i0 & i1) | i2
AO22 = np.uint16(0b1111_1000_1000_1000)  # (i0 & i1) | (i2 & i3)
OA21 = np.uint16(0b1110_0000_1110_0000)  # (i0 | i1) & i2
OA22 = np.uint16(0b1110_1110_1110_0000)  # (i0 | i1) & (i2 | i3)

AOI21, AOI22, OAI21, OAI22 = ~AO21, ~AO22, ~OA21, ~OA22

AO211 = np.uint16(0b1111_1111_1111_1000)  # (i0 & i1) | i2 | i3
OA211 = np.uint16(0b1110_0000_0000_0000)  # (i0 | i1) & i2 & i3

AOI211, OAI211 = ~AO211, ~OA211

MUX21 = np.uint16(0b1100_1010_1100_1010)  # z = i1 if i2 else i0 (i2 is select)

names = dict([(v, k) for k, v in globals().items() if isinstance(v, np.uint16)])

kind_prefixes = {
    'nand': (NAND4, NAND3, NAND2),
    'nor': (NOR4, NOR3, NOR2),
    'and': (AND4, AND3, AND2),
    'or': (OR4, OR3, OR2),
    'isolor': (OR2, OR2, OR2),
    'xor': (XOR4, XOR3, XOR2),
    'xnor': (XNOR4, XNOR3, XNOR2),

    'not': (INV1, INV1, INV1),
    'inv': (INV1, INV1, INV1),
    'ibuf': (INV1, INV1, INV1),
    '__const1__': (INV1, INV1, INV1),
    'tieh': (INV1, INV1, INV1),

    'buf': (BUF1, BUF1, BUF1),
    'nbuf': (BUF1, BUF1, BUF1),
    'delln': (BUF1, BUF1, BUF1),
    '__const0__': (BUF1, BUF1, BUF1),
    'tiel': (BUF1, BUF1, BUF1),

    'ao211': (AO211, AO211, AO211),
    'oa211': (OA211, OA211, OA211),
    'aoi211': (AOI211, AOI211, AOI211),
    'oai211': (OAI211, OAI211, OAI211),

    'ao22': (AO22, AO22, AO22),
    'aoi22': (AOI22, AOI22, AOI22),
    'ao21': (AO21, AO21, AO21),
    'aoi21': (AOI21, AOI21, AOI21),

    'oa22': (OA22, OA22, OA22),
    'oai22': (OAI22, OAI22, OAI22),
    'oa21': (OA21, OA21, OA21),
    'oai21': (OAI21, OAI21, OAI21),

    'mux21': (MUX21, MUX21, MUX21),
}

class Heap:
    def __init__(self):
        self.chunks = dict()  # map start location to chunk size
        self.released = list()  # chunks that were released
        self.current_size = 0
        self.max_size = 0

    def alloc(self, size):
        for idx, loc in enumerate(self.released):
            if self.chunks[loc] == size:
                del self.released[idx]
                return loc
            if self.chunks[loc] > size:  # split chunk
                chunksize = self.chunks[loc]
                self.chunks[loc] = size
                self.chunks[loc + size] = chunksize - size
                self.released[idx] = loc + size  # move released pointer: loc -> loc+size
                return loc
        # no previously released chunk; make new one
        loc = self.current_size
        self.chunks[loc] = size
        self.current_size += size
        self.max_size = max(self.max_size, self.current_size)
        return loc

    def free(self, loc):
        size = self.chunks[loc]
        if loc + size == self.current_size:  # end of managed area, remove chunk
            del self.chunks[loc]
            self.current_size -= size
            # check and remove prev chunk if free
            if len(self.released) > 0:
                prev = self.released[-1]
                if prev + self.chunks[prev] == self.current_size:
                    chunksize = self.chunks[prev]
                    del self.chunks[prev]
                    del self.released[-1]
                    self.current_size -= chunksize
            return
        released_idx = bisect(self.released, loc)
        if released_idx < len(self.released) and loc + size == self.released[released_idx]:  # next chunk is free, merge
            chunksize = size + self.chunks[loc + size]
            del self.chunks[loc + size]
            self.chunks[loc] = chunksize
            size = self.chunks[loc]
            self.released[released_idx] = loc
        else:
            insort_left(self.released, loc)  # put in a new release
        if released_idx > 0:  # check if previous chunk is free
            prev = self.released[released_idx - 1]
            if prev + self.chunks[prev] == loc:  # previous chunk is adjacent to freed one, merge
                chunksize = size + self.chunks[prev]
                del self.chunks[loc]
                self.chunks[prev] = chunksize
                del self.released[released_idx]

    def __repr__(self):
        r = []
        for loc in sorted(self.chunks.keys()):
            size = self.chunks[loc]
            released_idx = bisect(self.released, loc)
            is_released = released_idx > 0 and len(self.released) > 0 and self.released[released_idx - 1] == loc
            r.append(f'{loc:5d}: {"free" if is_released else "used"} {size}')
        return "\n".join(r)


class SimOps:
    """A static scheduler that translates a Circuit into a topologically sorted list of basic logic operations (self.ops) and
    a memory mapping (self.c_locs, self.c_caps) for use in simulators.

    :param circuit: The circuit to create a schedule for.
    :param strip_forks: If enabled, the scheduler will not include fork nodes to safe simulation time.
        Stripping forks will cause interconnect delay annotations of lines read by fork nodes to be ignored.
    :param c_reuse: If enabled, memory of intermediate signal waveforms will be re-used. This greatly reduces
        memory footprint, but intermediate signal waveforms become unaccessible after a propagation.
    """
    def __init__(self, circuit, c_caps=1, c_caps_min=1, a_ctrl=None, c_reuse=False, strip_forks=False):
        self.circuit = circuit
        self.s_len = len(circuit.s_nodes)

        if isinstance(c_caps, int):
            c_caps = [c_caps] * (len(circuit.lines)+3)

        if a_ctrl is None:
            a_ctrl = np.zeros((len(circuit.lines)+3, 3), dtype=np.int32)  # add 3 for zero, tmp, tmp2
            a_ctrl[:,0] = -1

        # special locations and offsets in c_locs/c_caps
        self.zero_idx = len(circuit.lines)
        self.tmp_idx = self.zero_idx + 1
        self.tmp2_idx = self.tmp_idx + 1
        self.ppi_offset = self.tmp2_idx + 1
        self.ppo_offset = self.ppi_offset + self.s_len
        self.c_locs_len = self.ppo_offset + self.s_len

        # translate circuit structure into self.ops
        ops = []
        interface_dict = dict((n, i) for i, n in enumerate(circuit.s_nodes))
        for n in circuit.topological_order():
            if n in interface_dict:
                inp_idx = self.ppi_offset + interface_dict[n]
                if len(n.outs) > 0 and n.outs[0] is not None:  # first output of a PI/PPI
                    ops.append((BUF1, n.outs[0].index, inp_idx, self.zero_idx, self.zero_idx, self.zero_idx, *a_ctrl[n.outs[0]]))
                if 'dff' in n.kind.lower():  # second output of DFF is inverted
                    if len(n.outs) > 1 and n.outs[1] is not None:
                        ops.append((INV1, n.outs[1].index, inp_idx, self.zero_idx, self.zero_idx, self.zero_idx, *a_ctrl[n.outs[1]]))
                else:  # if not DFF, no output is inverted.
                    for o_line in n.outs[1:]:
                        if o_line is not None:
                            ops.append((BUF1, o_line.index, inp_idx, self.zero_idx, self.zero_idx, self.zero_idx, *a_ctrl[o_line]))
                continue
            # regular node, not PI/PPI or PO/PPO
            o0_idx = n.outs[0].index if len(n.outs) > 0 and n.outs[0] is not None else self.tmp_idx
            i0_idx = n.ins[0].index if len(n.ins) > 0 and n.ins[0] is not None else self.zero_idx
            i1_idx = n.ins[1].index if len(n.ins) > 1 and n.ins[1] is not None else self.zero_idx
            i2_idx = n.ins[2].index if len(n.ins) > 2 and n.ins[2] is not None else self.zero_idx
            i3_idx = n.ins[3].index if len(n.ins) > 3 and n.ins[3] is not None else self.zero_idx
            kind = n.kind.lower()
            if kind == '__fork__':
                if not strip_forks:
                    for o_line in n.outs:
                        if o_line is not None:
                            ops.append((BUF1, o_line.index, i0_idx, i1_idx, i2_idx, i3_idx, *a_ctrl[o_line]))
                continue
            sp = None
            for prefix, prims in kind_prefixes.items():
                if kind.startswith(prefix):
                    sp = prims[0]
                    if i3_idx == self.zero_idx:
                        sp = prims[1]
                        if i2_idx == self.zero_idx:
                            sp = prims[2]
                    break
            if sp is None:
                print('unknown cell type', kind)
            else:
                ops.append((sp, o0_idx, i0_idx, i1_idx, i2_idx, i3_idx, *a_ctrl[o0_idx]))

        self.ops = np.asarray(ops, dtype='int32')

        # create a map from fanout lines to stem lines for fork stripping
        stems = np.zeros(self.c_locs_len, dtype='int32') - 1  # default to -1: 'no fanout line'
        if strip_forks:
            for f in circuit.forks.values():
                prev_line = f.ins[0]
                while prev_line.driver.kind == '__fork__':
                    prev_line = prev_line.driver.ins[0]
                stem_idx = prev_line.index
                for ol in f.outs:
                    if ol is not None:
                        stems[ol] = stem_idx

        # calculate level (distance from PI/PPI) and reference count for each line
        levels = np.zeros(self.c_locs_len, dtype='int32')
        ref_count = np.zeros(self.c_locs_len, dtype='int32')
        level_starts = [0]
        current_level = 1
        for i, op in enumerate(self.ops):
            # if we fork-strip, always take the stems for determining fan-in level
            i0_idx = stems[op[2]] if stems[op[2]] >= 0 else op[2]
            i1_idx = stems[op[3]] if stems[op[3]] >= 0 else op[3]
            i2_idx = stems[op[4]] if stems[op[4]] >= 0 else op[4]
            i3_idx = stems[op[5]] if stems[op[5]] >= 0 else op[5]
            if levels[i0_idx] >= current_level or levels[i1_idx] >= current_level or levels[i2_idx] >= current_level or levels[i3_idx] >= current_level:
                current_level += 1
                level_starts.append(i)
            levels[op[1]] = current_level  # set level of the output line
            ref_count[i0_idx] += 1
            ref_count[i1_idx] += 1
            ref_count[i2_idx] += 1
            ref_count[i3_idx] += 1
        self.level_starts = np.asarray(level_starts, dtype='int32')
        self.level_stops = np.asarray(level_starts[1:] + [len(self.ops)], dtype='int32')

        # combinational signal allocation table. maps line and interface indices to self.c memory locations
        self.c_locs = np.full((self.c_locs_len,), -1, dtype=np.int32)
        self.c_caps = np.zeros((self.c_locs_len,), dtype=np.int32)

        h = Heap()

        # allocate and keep memory for special fields
        self.c_locs[self.zero_idx], self.c_caps[self.zero_idx] = h.alloc(c_caps_min), c_caps_min
        self.c_locs[self.tmp_idx], self.c_caps[self.tmp_idx] = h.alloc(c_caps_min), c_caps_min
        self.c_locs[self.tmp2_idx], self.c_caps[self.tmp2_idx] = h.alloc(c_caps_min), c_caps_min
        ref_count[self.zero_idx] += 1
        ref_count[self.tmp_idx] += 1
        ref_count[self.tmp2_idx] += 1

        # allocate and keep memory for PI/PPI, keep memory for PO/PPO (allocated later)
        for i, n in enumerate(circuit.s_nodes):
            if len(n.outs) > 0:
                self.c_locs[self.ppi_offset + i], self.c_caps[self.ppi_offset + i] = h.alloc(c_caps_min), c_caps_min
                ref_count[self.ppi_offset + i] += 1
            if len(n.ins) > 0:
                i0_idx = stems[n.ins[0]] if stems[n.ins[0]] >= 0 else n.ins[0]
                ref_count[i0_idx] += 1

        # allocate memory for the rest of the circuit
        for op_start, op_stop in zip(self.level_starts, self.level_stops):
            free_set = set()
            for op in self.ops[op_start:op_stop]:
                # if we fork-strip, always take the stems
                i0_idx = stems[op[2]] if stems[op[2]] >= 0 else op[2]
                i1_idx = stems[op[3]] if stems[op[3]] >= 0 else op[3]
                i2_idx = stems[op[4]] if stems[op[4]] >= 0 else op[4]
                i3_idx = stems[op[5]] if stems[op[5]] >= 0 else op[5]
                ref_count[i0_idx] -= 1
                ref_count[i1_idx] -= 1
                ref_count[i2_idx] -= 1
                ref_count[i3_idx] -= 1
                if ref_count[i0_idx] <= 0: free_set.add(self.c_locs[i0_idx])
                if ref_count[i1_idx] <= 0: free_set.add(self.c_locs[i1_idx])
                if ref_count[i2_idx] <= 0: free_set.add(self.c_locs[i2_idx])
                if ref_count[i3_idx] <= 0: free_set.add(self.c_locs[i3_idx])
                o_idx = op[1]
                cap = max(c_caps_min, c_caps[o_idx])
                self.c_locs[o_idx], self.c_caps[o_idx] = h.alloc(cap), cap
            if c_reuse:
                for loc in free_set:
                    h.free(loc)

        # copy memory location and capacity from stems to fanout lines
        for lidx, stem in enumerate(stems):
            if stem >= 0:  # if at a fanout line
                self.c_locs[lidx], self.c_caps[lidx] = self.c_locs[stem], self.c_caps[stem]

        # copy memory location to PO/PPO area
        for i, n in enumerate(circuit.s_nodes):
            if len(n.ins) > 0:
                self.c_locs[self.ppo_offset + i], self.c_caps[self.ppo_offset + i] = self.c_locs[n.ins[0]], self.c_caps[n.ins[0]]

        self.c_len = h.max_size

        d = defaultdict(int)
        for op in self.ops[:,0]: d[names[op]] += 1
        self.prim_counts = dict(d)

        self.pi_s_locs = np.flatnonzero(self.c_locs[self.ppi_offset+np.arange(len(self.circuit.io_nodes))] >= 0)
        self.po_s_locs = np.flatnonzero(self.c_locs[self.ppo_offset+np.arange(len(self.circuit.io_nodes))] >= 0)
        self.ppio_s_locs = np.arange(len(self.circuit.io_nodes), self.s_len)

        self.pippi_s_locs = np.concatenate([self.pi_s_locs, self.ppio_s_locs])
        self.poppo_s_locs = np.concatenate([self.po_s_locs, self.ppio_s_locs])

        self.pi_c_locs = self.c_locs[self.ppi_offset+self.pi_s_locs]
        self.po_c_locs = self.c_locs[self.ppo_offset+self.po_s_locs]
        self.ppi_c_locs = self.c_locs[self.ppi_offset+self.ppio_s_locs]
        self.ppo_c_locs = self.c_locs[self.ppo_offset+self.ppio_s_locs]

        self.pippi_c_locs = np.concatenate([self.pi_c_locs, self.ppi_c_locs])
        self.poppo_c_locs = np.concatenate([self.po_c_locs, self.ppo_c_locs])
