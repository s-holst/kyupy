"""High-throughput combinational logic timing simulators.

These simulators work similarly to :py:class:`~kyupy.logic_sim.LogicSim`.
They propagate values through the combinational circuit from (pseudo) primary inputs to (pseudo) primary outputs.
Instead of propagating logic values, these simulators propagate signal histories (waveforms).
They are designed to run many simulations in parallel and while their latencies are quite high, they can achieve
high throughput.

The simulators are not event-based and are not capable of simulating sequential circuits directly.

Two simulators are available: :py:class:`WaveSim` runs on the CPU, and the derived class
:py:class:`WaveSimCuda` runs on the GPU.
"""

import math
from bisect import bisect, insort_left

import numpy as np

from . import numba, cuda, hr_bytes


TMAX = np.float32(2 ** 127)
"""A large 32-bit floating point value used to mark the end of a waveform."""
TMAX_OVL = np.float32(1.1 * 2 ** 127)
"""A large 32-bit floating point value used to mark the end of a waveform that
may be incomplete due to an overflow."""
TMIN = np.float32(-2 ** 127)
"""A large negative 32-bit floating point value used at the beginning of waveforms that start with logic-1."""


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


class WaveSim:
    """A waveform-based combinational logic timing simulator running on CPU.

    :param circuit: The circuit to simulate.
    :param timing: The timing annotation of the circuit (see :py:func:`kyupy.sdf.DelayFile.annotation` for details)
    :param sims: The number of parallel simulations.
    :param wavecaps: The number of floats available in each waveform. Waveforms are encoding the signal switching
        history by storing transition times. The waveform capacity roughly corresponds to the number of transitions
        that can be stored. A capacity of ``n`` can store at least ``n-2`` transitions. If more transitions are
        generated during simulation, the latest glitch is removed (freeing up two transition times) and an overflow
        flag is set. If an integer is given, all waveforms are set to that same capacity. With an array of length
        ``len(circuit.lines)`` the capacity can be controlled for each intermediate waveform individually.
    :param strip_forks: If enabled, the simulator will not evaluate fork nodes explicitly. This saves simulation time
        by reducing the number of nodes to simulate, but (interconnect) delay annotations of lines read by fork nodes
        are ignored.
    :param keep_waveforms: If disabled, memory of intermediate signal waveforms will be re-used. This greatly reduces
        memory footprint, but intermediate signal waveforms become unaccessible after a propagation.
    """
    def __init__(self, circuit, timing, sims=8, wavecaps=16, strip_forks=False, keep_waveforms=True):
        self.circuit = circuit
        self.sims = sims
        self.overflows = 0
        self.interface = list(circuit.interface) + [n for n in circuit.nodes if 'dff' in n.kind.lower()]

        self.lst_eat_valid = False

        self.cdata = np.zeros((len(self.interface), sims, 7), dtype='float32')

        if isinstance(wavecaps, int):
            wavecaps = [wavecaps] * len(circuit.lines)

        intf_wavecap = 4  # sufficient for storing only 1 transition.

        # indices for state allocation table (sat)
        self.zero_idx = len(circuit.lines)
        self.tmp_idx = self.zero_idx + 1
        self.ppi_offset = self.tmp_idx + 1
        self.ppo_offset = self.ppi_offset + len(self.interface)
        self.sat_length = self.ppo_offset + len(self.interface)

        # translate circuit structure into self.ops
        ops = []
        interface_dict = dict((n, i) for i, n in enumerate(self.interface))
        for n in circuit.topological_order():
            if n in interface_dict:
                inp_idx = self.ppi_offset + interface_dict[n]
                if len(n.outs) > 0 and n.outs[0] is not None:  # first output of a PI/PPI
                    ops.append((0b1010, n.outs[0].index, inp_idx, self.zero_idx))
                if 'dff' in n.kind.lower():  # second output of DFF is inverted
                    if len(n.outs) > 1 and n.outs[1] is not None:
                        ops.append((0b0101, n.outs[1].index, inp_idx, self.zero_idx))
                else:  # if not DFF, no output is inverted.
                    for o_line in n.outs[1:]:
                        if o_line is not None:
                            ops.append((0b1010, o_line.index, inp_idx, self.zero_idx))
            else:  # regular node, not PI/PPI or PO/PPO
                o0_idx = n.outs[0].index if len(n.outs) > 0 and n.outs[0] is not None else self.tmp_idx
                i0_idx = n.ins[0].index if len(n.ins) > 0 and n.ins[0] is not None else self.zero_idx
                i1_idx = n.ins[1].index if len(n.ins) > 1 and n.ins[1] is not None else self.zero_idx
                kind = n.kind.lower()
                if kind == '__fork__':
                    if not strip_forks:
                        for o_line in n.outs:
                            ops.append((0b1010, o_line.index, i0_idx, i1_idx))
                elif kind.startswith('nand'):
                    ops.append((0b0111, o0_idx, i0_idx, i1_idx))
                elif kind.startswith('nor'):
                    ops.append((0b0001, o0_idx, i0_idx, i1_idx))
                elif kind.startswith('and'):
                    ops.append((0b1000, o0_idx, i0_idx, i1_idx))
                elif kind.startswith('or'):
                    ops.append((0b1110, o0_idx, i0_idx, i1_idx))
                elif kind.startswith('xor'):
                    ops.append((0b0110, o0_idx, i0_idx, i1_idx))
                elif kind.startswith('xnor'):
                    ops.append((0b1001, o0_idx, i0_idx, i1_idx))
                elif kind.startswith('not') or kind.startswith('inv') or kind.startswith('ibuf'):
                    ops.append((0b0101, o0_idx, i0_idx, i1_idx))
                elif kind.startswith('buf') or kind.startswith('nbuf'):
                    ops.append((0b1010, o0_idx, i0_idx, i1_idx))
                elif kind.startswith('__const1__') or kind.startswith('tieh'):
                    ops.append((0b0101, o0_idx, i0_idx, i1_idx))
                elif kind.startswith('__const0__') or kind.startswith('tiel'):
                    ops.append((0b1010, o0_idx, i0_idx, i1_idx))
                else:
                    print('unknown gate type', kind)
        self.ops = np.asarray(ops, dtype='int32')

        # create a map from fanout lines to stem lines for fork stripping
        stems = np.zeros(self.sat_length, dtype='int32') - 1  # default to -1: 'no fanout line'
        if strip_forks:
            for f in circuit.forks.values():
                prev_line = f.ins[0]
                while prev_line.driver.kind == '__fork__':
                    prev_line = prev_line.driver.ins[0]
                stem_idx = prev_line.index
                for ol in f.outs:
                    stems[ol] = stem_idx

        # calculate level (distance from PI/PPI) and reference count for each line
        levels = np.zeros(self.sat_length, dtype='int32')
        ref_count = np.zeros(self.sat_length, dtype='int32')
        level_starts = [0]
        current_level = 1
        for i, op in enumerate(self.ops):
            # if we fork-strip, always take the stems for determining fan-in level
            i0_idx = stems[op[2]] if stems[op[2]] >= 0 else op[2]
            i1_idx = stems[op[3]] if stems[op[3]] >= 0 else op[3]
            if levels[i0_idx] >= current_level or levels[i1_idx] >= current_level:
                current_level += 1
                level_starts.append(i)
            levels[op[1]] = current_level  # set level of the output line
            ref_count[i0_idx] += 1
            ref_count[i1_idx] += 1
        self.level_starts = np.asarray(level_starts, dtype='int32')
        self.level_stops = np.asarray(level_starts[1:] + [len(self.ops)], dtype='int32')

        # state allocation table. maps line and interface indices to self.state memory locations
        self.sat = np.zeros((self.sat_length, 3), dtype='int')
        self.sat[:, 0] = -1

        h = Heap()

        # allocate and keep memory for special fields
        self.sat[self.zero_idx] = h.alloc(intf_wavecap), intf_wavecap, 0
        self.sat[self.tmp_idx] = h.alloc(intf_wavecap), intf_wavecap, 0
        ref_count[self.zero_idx] += 1
        ref_count[self.tmp_idx] += 1

        # allocate and keep memory for PI/PPI, keep memory for PO/PPO (allocated later)
        for i, n in enumerate(self.interface):
            if len(n.outs) > 0:
                self.sat[self.ppi_offset + i] = h.alloc(intf_wavecap), intf_wavecap, 0
                ref_count[self.ppi_offset + i] += 1
            if len(n.ins) > 0:
                i0_idx = stems[n.ins[0]] if stems[n.ins[0]] >= 0 else n.ins[0]
                ref_count[i0_idx] += 1

        # allocate memory for the rest of the circuit
        for op_start, op_stop in zip(self.level_starts, self.level_stops):
            free_list = []
            for op in self.ops[op_start:op_stop]:
                # if we fork-strip, always take the stems
                i0_idx = stems[op[2]] if stems[op[2]] >= 0 else op[2]
                i1_idx = stems[op[3]] if stems[op[3]] >= 0 else op[3]
                ref_count[i0_idx] -= 1
                ref_count[i1_idx] -= 1
                if ref_count[i0_idx] <= 0: free_list.append(self.sat[i0_idx, 0])
                if ref_count[i1_idx] <= 0: free_list.append(self.sat[i1_idx, 0])
                o_idx = op[1]
                cap = wavecaps[o_idx]
                self.sat[o_idx] = h.alloc(cap), cap, 0
            if not keep_waveforms:
                for loc in free_list:
                    h.free(loc)

        # copy memory location and capacity from stems to fanout lines
        for lidx, stem in enumerate(stems):
            if stem >= 0:  # if at a fanout line
                self.sat[lidx] = self.sat[stem]

        # copy memory location to PO/PPO area
        for i, n in enumerate(self.interface):
            if len(n.ins) > 0:
                self.sat[self.ppo_offset + i] = self.sat[n.ins[0]]

        # pad timing
        self.timing = np.zeros((self.sat_length, 2, 2))
        self.timing[:len(timing)] = timing

        # allocate self.state
        self.state = np.zeros((h.max_size, sims), dtype='float32') + TMAX

        m1 = np.array([2 ** x for x in range(7, -1, -1)], dtype='uint8')
        m0 = ~m1
        self.mask = np.rollaxis(np.vstack((m0, m1)), 1)

    def __repr__(self):
        total_mem = self.state.nbytes + self.sat.nbytes + self.ops.nbytes + self.cdata.nbytes
        return f'<WaveSim {self.circuit.name} sims={self.sims} ops={len(self.ops)} ' + \
               f'levels={len(self.level_starts)} mem={hr_bytes(total_mem)}>'

    def get_line_delay(self, line, polarity):
        """Returns the current delay of the given ``line`` and ``polarity`` in the simulation model."""
        return self.timing[line, 0, polarity]

    def set_line_delay(self, line, polarity, delay):
        """Sets a new ``delay`` for the given ``line`` and ``polarity`` in the simulation model."""
        self.timing[line, 0, polarity] = delay

    def assign(self, vectors, time=0.0, offset=0):
        """Assigns new values to the primary inputs and state-elements.

        :param vectors: The values to assign preferably in 8-valued logic. The values are converted to
            appropriate waveforms with or one transition (``RISE``, ``FALL``) no transitions
            (``ZERO``, ``ONE``, and others).
        :type vectors: :py:class:`~kyupy.logic.BPArray`
        :param time: The transition time of the generated waveforms.
        :param offset: The offset into the vector set. The vector assigned to the first simulator is
            ``vectors[offset]``.
        """
        nvectors = min(len(vectors) - offset, self.sims)
        for i in range(len(self.interface)):
            ppi_loc = self.sat[self.ppi_offset + i, 0]
            if ppi_loc < 0: continue
            for p in range(nvectors):
                vector = p + offset
                a = vectors.data[i, :, vector // 8]
                m = self.mask[vector % 8]
                toggle = 0
                if len(a) <= 2:
                    if a[0] & m[1]:
                        self.state[ppi_loc, p] = TMIN
                        toggle += 1
                else:
                    if a[1] & m[1]:
                        self.state[ppi_loc, p] = TMIN
                        toggle += 1
                    if (a[2] & m[1]) and ((a[0] & m[1]) != (a[1] & m[1])):
                        self.state[ppi_loc + toggle, p] = time
                        toggle += 1
                self.state[ppi_loc + toggle, p] = TMAX

    def propagate(self, sims=None, sd=0.0, seed=1):
        """Propagates all waveforms from the (pseudo) primary inputs to the (pseudo) primary outputs.

        :param sims: Number of parallel simulations to execute. If None, all available simulations are performed.
        :param sd: Standard deviation for injection of random delay variation. Active, if value is positive.
        :param seed: Random seed for delay variations.
        """
        sims = min(sims or self.sims, self.sims)
        for op_start, op_stop in zip(self.level_starts, self.level_stops):
            self.overflows += level_eval(self.ops, op_start, op_stop, self.state, self.sat, 0, sims,
                                         self.timing, sd, seed)
        self.lst_eat_valid = False

    def wave(self, line, vector):
        # """Returns the desired waveform from the simulation state. Only valid, if simulator was
        # instantiated with ``keep_waveforms=True``."""
        if line < 0:
            return [TMAX]
        mem, wcap, _ = self.sat[line]
        if mem < 0:
            return [TMAX]
        return self.state[mem:mem + wcap, vector]

    def wave_ppi(self, i, vector):
        return self.wave(self.ppi_offset + i, vector)

    def wave_ppo(self, o, vector):
        return self.wave(self.ppo_offset + o, vector)

    def capture(self, time=TMAX, sd=0.0, seed=1, cdata=None, offset=0):
        """Simulates a capture operation at all state-elements and primary outputs.

        The capture analyzes the propagated waveforms at and around the given capture time and returns
        various results for each capture operation.

        :param time: The desired capture time. By default, a capture of the settled value is performed.
        :param sd: A standard deviation for uncertainty in the actual capture time.
        :param seed: The random seed for a capture with uncertainty.
        :param cdata: An array to copy capture data into (optional). See the return value for details.
        :param offset: An offset into the supplied capture data array.
        :return: The capture data as numpy array.

            The 3-dimensional capture data array contains for each interface node (axis 0),
            and each test (axis 1), seven values:

            0. Probability of capturing a 1 at the given capture time (same as next value, if no
               standard deviation given).
            1. A capture value decided by random sampling according to above probability and given seed.
            2. The final value (assume a very late capture time).
            3. True, if there was a premature capture (capture error), i.e. final value is different
               from captured value.
            4. Earliest arrival time. The time at which the output transitioned from its initial value.
            5. Latest stabilization time. The time at which the output transitioned to its final value.
            6. Overflow indicator. If non-zero, some signals in the input cone of this output had more
               transitions than specified in ``wavecaps``. Some transitions have been discarded, the
               final values in the waveforms are still valid.
        """
        for i, node in enumerate(self.interface):
            if len(node.ins) == 0: continue
            for p in range(self.sims):
                self.cdata[i, p] = self.capture_wave(self.ppo_offset + i, p, time, sd, seed)
        if cdata is not None:
            assert offset < cdata.shape[1]
            cap_dim = min(cdata.shape[1] - offset, self.sims)
            cdata[:, offset:cap_dim + offset] = self.cdata[:, 0:cap_dim]
        self.lst_eat_valid = True
        return self.cdata

    def reassign(self, time=0.0):
        """Re-assigns the last capture to the appropriate pseudo-primary inputs. Generates a new set of
        waveforms at the PPIs that start with the previous final value of that PPI, and transitions at the
        given time to the value captured in a previous simulation. :py:func:`~WaveSim.capture` must be called
        prior to this function. The final value of each PPI is taken from the randomly sampled concrete logic
        values in the capture data.

        :param time: The transition time at the inputs (usually 0.0).
        """
        for i in range(len(self.interface)):
            ppi_loc = self.sat[self.ppi_offset + i, 0]
            ppo_loc = self.sat[self.ppo_offset + i, 0]
            if ppi_loc < 0 or ppo_loc < 0: continue
            for sidx in range(self.sims):
                ival = self.val(self.ppi_offset + i, sidx, TMAX) > 0.5
                oval = self.cdata[i, sidx, 1] > 0.5
                toggle = 0
                if ival:
                    self.state[ppi_loc, sidx] = TMIN
                    toggle += 1
                if ival != oval:
                    self.state[ppi_loc + toggle, sidx] = time
                    toggle += 1
                self.state[ppi_loc + toggle, sidx] = TMAX

    def eat(self, line, vector):
        eat = TMAX
        for t in self.wave(line, vector):
            if t >= TMAX: break
            if t <= TMIN: continue
            eat = min(eat, t)
        return eat

    def lst(self, line, vector):
        lst = TMIN
        for t in self.wave(line, vector):
            if t >= TMAX: break
            if t <= TMIN: continue
            lst = max(lst, t)
        return lst

    def lst_ppo(self, o, vector):
        if not self.lst_eat_valid:
            self.capture()
        return self.cdata[o, vector, 5]

    def toggles(self, line, vector):
        tog = 0
        for t in self.wave(line, vector):
            if t >= TMAX: break
            if t <= TMIN: continue
            tog += 1
        return tog

    def _vals(self, idx, vector, times, sd=0.0):
        s_sqrt2 = sd * math.sqrt(2)
        m = 0.5
        accs = [0.0] * len(times)
        values = [0] * len(times)
        for t in self.wave(idx, vector):
            if t >= TMAX: break
            for idx, time in enumerate(times):
                if t < time:
                    values[idx] = values[idx] ^ 1
            m = -m
            if t <= TMIN: continue
            if s_sqrt2 > 0:
                for idx, time in enumerate(times):
                    accs[idx] += m * (1 + math.erf((t - time) / s_sqrt2))
        if (m < 0) and (s_sqrt2 > 0):
            for idx, time in enumerate(times):
                accs[idx] += 1
        if s_sqrt2 == 0:
            return values
        return accs

    def vals(self, line, vector, times, sd=0):
        return self._vals(line, vector, times, sd)

    def val(self, line, vector, time=TMAX, sd=0):
        return self.capture_wave(line, vector, time, sd)[0]

    def vals_ppo(self, o, vector, times, sd=0):
        return self._vals(self.ppo_offset + o, vector, times, sd)

    def val_ppo(self, o, vector, time=TMAX, sd=0):
        if not self.lst_eat_valid:
            self.capture(time, sd)
        return self.cdata[o, vector, 0]

    def capture_wave(self, line, vector, time=TMAX, sd=0.0, seed=1):
        s_sqrt2 = sd * math.sqrt(2)
        m = 0.5
        acc = 0.0
        eat = TMAX
        lst = TMIN
        tog = 0
        ovl = 0
        val = int(0)
        final = int(0)
        for t in self.wave(line, vector):
            if t >= TMAX:
                if t == TMAX_OVL:
                    ovl = 1
                break
            m = -m
            final ^= 1
            if t < time:
                val ^= 1
            if t <= TMIN: continue
            if s_sqrt2 > 0:
                acc += m * (1 + math.erf((t - time) / s_sqrt2))
            eat = min(eat, t)
            lst = max(lst, t)
            tog += 1
        if s_sqrt2 > 0:
            if m < 0:
                acc += 1
            if acc >= 0.99:
                val = 1
            elif acc > 0.01:
                seed = (seed << 4) + (vector << 20) + (line-self.ppo_offset << 1)
                seed = int(0xDEECE66D) * seed + 0xB
                seed = int(0xDEECE66D) * seed + 0xB
                rnd = float((seed >> 8) & 0xffffff) / float(1 << 24)
                val = rnd < acc
            else:
                val = 0
        else:
            acc = val

        return acc, val, final, (val != final), eat, lst, ovl


@numba.njit
def level_eval(ops, op_start, op_stop, state, sat, st_start, st_stop, line_times, sd, seed):
    overflows = 0
    for op_idx in range(op_start, op_stop):
        op = ops[op_idx]
        for st_idx in range(st_start, st_stop):
            overflows += wave_eval(op, state, sat, st_idx, line_times, sd, seed)
    return overflows


@numba.njit
def rand_gauss(seed, sd):
    clamp = 0.5
    if sd <= 0.0:
        return 1.0
    while True:
        x = -6.0
        for _ in range(12):
            seed = int(0xDEECE66D) * seed + 0xB
            x += float((seed >> 8) & 0xffffff) / float(1 << 24)
        x *= sd
        if abs(x) <= clamp:
            break
    return x + 1.0


@numba.njit
def wave_eval(op, state, sat, st_idx, line_times, sd=0.0, seed=0):
    lut, z_idx, a_idx, b_idx = op
    overflows = int(0)

    _seed = (seed << 4) + (z_idx << 20) + (st_idx << 1)

    a_mem = sat[a_idx, 0]
    b_mem = sat[b_idx, 0]
    z_mem, z_cap, _ = sat[z_idx]

    a_cur = int(0)
    b_cur = int(0)
    z_cur = lut & 1
    if z_cur == 1:
        state[z_mem, st_idx] = TMIN

    a = state[a_mem, st_idx] + line_times[a_idx, 0, z_cur] * rand_gauss(_seed ^ a_mem ^ z_cur, sd)
    b = state[b_mem, st_idx] + line_times[b_idx, 0, z_cur] * rand_gauss(_seed ^ b_mem ^ z_cur, sd)

    previous_t = TMIN

    current_t = min(a, b)
    inputs = int(0)

    while current_t < TMAX:
        z_val = z_cur & 1
        if b < a:
            b_cur += 1
            b = state[b_mem + b_cur, st_idx]
            b += line_times[b_idx, 0, z_val ^ 1] * rand_gauss(_seed ^ b_mem ^ z_val ^ 1, sd)
            thresh = line_times[b_idx, 1, z_val] * rand_gauss(_seed ^ b_mem ^ z_val, sd)
            inputs ^= 2
            next_t = b
        else:
            a_cur += 1
            a = state[a_mem + a_cur, st_idx]
            a += line_times[a_idx, 0, z_val ^ 1] * rand_gauss(_seed ^ a_mem ^ z_val ^ 1, sd)
            thresh = line_times[a_idx, 1, z_val] * rand_gauss(_seed ^ a_mem ^ z_val, sd)
            inputs ^= 1
            next_t = a

        if (z_cur & 1) != ((lut >> inputs) & 1):
            # we generate a toggle in z_mem, if:
            #   ( it is the first toggle in z_mem OR
            #   following toggle is earlier OR
            #   pulse is wide enough ) AND enough space in z_mem.
            if z_cur == 0 or next_t < current_t or (current_t - previous_t) > thresh:
                if z_cur < (z_cap - 1):
                    state[z_mem + z_cur, st_idx] = current_t
                    previous_t = current_t
                    z_cur += 1
                else:
                    overflows += 1
                    previous_t = state[z_mem + z_cur - 1, st_idx]
                    z_cur -= 1
            else:
                z_cur -= 1
                if z_cur > 0:
                    previous_t = state[z_mem + z_cur - 1, st_idx]
                else:
                    previous_t = TMIN
        current_t = min(a, b)

    if overflows > 0:
        state[z_mem + z_cur, st_idx] = TMAX_OVL
    else:
        state[z_mem + z_cur, st_idx] = a if a > b else b  # propagate overflow flags by storing biggest TMAX from input

    return overflows


class WaveSimCuda(WaveSim):
    """A GPU-accelerated waveform-based combinational logic timing simulator.

    The API is the same as for :py:class:`WaveSim`.
    All internal memories are mirrored into GPU memory upon construction.
    Some operations like access to single waveforms can involve large communication overheads.
    """
    def __init__(self, circuit, timing, sims=8, wavecaps=16, strip_forks=False, keep_waveforms=True):
        super().__init__(circuit, timing, sims, wavecaps, strip_forks, keep_waveforms)

        self.tdata = np.zeros((len(self.interface), 3, (sims - 1) // 8 + 1), dtype='uint8')

        self.d_state = cuda.to_device(self.state)
        self.d_sat = cuda.to_device(self.sat)
        self.d_ops = cuda.to_device(self.ops)
        self.d_timing = cuda.to_device(self.timing)
        self.d_tdata = cuda.to_device(self.tdata)
        self.d_cdata = cuda.to_device(self.cdata)

        self._block_dim = (32, 16)

    def __repr__(self):
        total_mem = self.state.nbytes + self.sat.nbytes + self.ops.nbytes + self.timing.nbytes + \
                    self.tdata.nbytes + self.cdata.nbytes
        return f'<WaveSimCuda {self.circuit.name} sims={self.sims} ops={len(self.ops)} ' + \
               f'levels={len(self.level_starts)} mem={hr_bytes(total_mem)}>'

    def get_line_delay(self, line, polarity):
        return self.d_timing[line, 0, polarity]

    def set_line_delay(self, line, polarity, delay):
        self.d_timing[line, 0, polarity] = delay

    def assign(self, vectors, time=0.0, offset=0):
        assert (offset % 8) == 0
        byte_offset = offset // 8
        assert byte_offset < vectors.data.shape[-1]
        pdim = min(vectors.data.shape[-1] - byte_offset, self.tdata.shape[-1])

        self.tdata[..., 0:pdim] = vectors.data[..., byte_offset:pdim + byte_offset]
        if vectors.m == 2:
            self.tdata[:, 2, 0:pdim] = 0
        cuda.to_device(self.tdata, to=self.d_tdata)

        grid_dim = self._grid_dim(self.sims, len(self.interface))
        assign_kernel[grid_dim, self._block_dim](self.d_state, self.d_sat, self.ppi_offset,
                                                 len(self.interface), self.d_tdata, time)

    def _grid_dim(self, x, y):
        gx = math.ceil(x / self._block_dim[0])
        gy = math.ceil(y / self._block_dim[1])
        return gx, gy

    def propagate(self, sims=None, sd=0.0, seed=1):
        sims = min(sims or self.sims, self.sims)
        for op_start, op_stop in zip(self.level_starts, self.level_stops):
            grid_dim = self._grid_dim(sims, op_stop - op_start)
            wave_kernel[grid_dim, self._block_dim](self.d_ops, op_start, op_stop, self.d_state, self.sat, int(0),
                                                   sims, self.d_timing, sd, seed)
        cuda.synchronize()
        self.lst_eat_valid = False

    def wave(self, line, vector):
        if line < 0:
            return [TMAX]
        mem, wcap, _ = self.sat[line]
        if mem < 0:
            return [TMAX]
        return self.d_state[mem:mem + wcap, vector]

    def capture(self, time=TMAX, sd=0, seed=1, cdata=None, offset=0):
        grid_dim = self._grid_dim(self.sims, len(self.interface))
        capture_kernel[grid_dim, self._block_dim](self.d_state, self.d_sat, self.ppo_offset,
                                                  self.d_cdata, time, sd * math.sqrt(2), seed)
        self.cdata[...] = self.d_cdata
        if cdata is not None:
            assert offset < cdata.shape[1]
            cap_dim = min(cdata.shape[1] - offset, self.sims)
            cdata[:, offset:cap_dim + offset] = self.cdata[:, 0:cap_dim]
        self.lst_eat_valid = True
        return self.cdata

    def reassign(self, time=0.0):
        grid_dim = self._grid_dim(self.sims, len(self.interface))
        reassign_kernel[grid_dim, self._block_dim](self.d_state, self.d_sat, self.ppi_offset, self.ppo_offset,
                                                   self.d_cdata, time)
        cuda.synchronize()

    def wavecaps(self):
        gx = math.ceil(len(self.circuit.lines) / 512)
        wavecaps_kernel[gx, 512](self.d_state, self.d_sat, self.sims)
        self.sat[...] = self.d_sat
        return self.sat[..., 2]


@cuda.jit()
def wavecaps_kernel(state, sat, sims):
    idx = cuda.grid(1)
    if idx >= len(sat): return

    lidx, lcap, _ = sat[idx]
    if lidx < 0: return

    wcap = 0
    for sidx in range(sims):
        for tidx in range(lcap):
            t = state[lidx + tidx, sidx]
            if tidx > wcap:
                wcap = tidx
            if t >= TMAX: break

    sat[idx, 2] = wcap + 1


@cuda.jit()
def reassign_kernel(state, sat, ppi_offset, ppo_offset, cdata, ppi_time):
    vector, y = cuda.grid(2)
    if vector >= state.shape[-1]: return
    if ppo_offset + y >= len(sat): return

    ppo, _, _ = sat[ppo_offset + y]
    ppi, ppi_cap, _ = sat[ppi_offset + y]
    if ppo < 0: return
    if ppi < 0: return

    ppo_val = int(cdata[y, vector, 1])
    ppi_val = int(0)
    for tidx in range(ppi_cap):
        t = state[ppi + tidx, vector]
        if t >= TMAX: break
        ppi_val ^= 1

    # make new waveform at PPI
    toggle = 0
    if ppi_val:
        state[ppi + toggle, vector] = TMIN
        toggle += 1
    if ppi_val != ppo_val:
        state[ppi + toggle, vector] = ppi_time
        toggle += 1
    state[ppi + toggle, vector] = TMAX


@cuda.jit()
def capture_kernel(state, sat, ppo_offset, cdata, time, s_sqrt2, seed):
    x, y = cuda.grid(2)
    if ppo_offset + y >= len(sat): return
    line, tdim, _ = sat[ppo_offset + y]
    if line < 0: return
    if x >= state.shape[-1]: return
    vector = x
    m = 0.5
    acc = 0.0
    eat = TMAX
    lst = TMIN
    tog = 0
    ovl = 0
    val = int(0)
    final = int(0)
    for tidx in range(tdim):
        t = state[line + tidx, vector]
        if t >= TMAX:
            if t == TMAX_OVL:
                ovl = 1
            break
        m = -m
        final ^= 1
        if t < time:
            val ^= 1
        if t <= TMIN: continue
        if s_sqrt2 > 0:
            acc += m * (1 + math.erf((t - time) / s_sqrt2))
        eat = min(eat, t)
        lst = max(lst, t)
        tog += 1
    if s_sqrt2 > 0:
        if m < 0:
            acc += 1
        if acc >= 0.99:
            val = 1
        elif acc > 0.01:
            seed = (seed << 4) + (vector << 20) + (y << 1)
            seed = int(0xDEECE66D) * seed + 0xB
            seed = int(0xDEECE66D) * seed + 0xB
            rnd = float((seed >> 8) & 0xffffff) / float(1 << 24)
            val = rnd < acc
        else:
            val = 0
    else:
        acc = val

    cdata[y, vector, 0] = acc
    cdata[y, vector, 1] = val
    cdata[y, vector, 2] = final
    cdata[y, vector, 3] = (val != final)
    cdata[y, vector, 4] = eat
    cdata[y, vector, 5] = lst
    cdata[y, vector, 6] = ovl


@cuda.jit()
def assign_kernel(state, sat, ppi_offset, intf_len, tdata, time):
    x, y = cuda.grid(2)
    if y >= intf_len: return
    line = sat[ppi_offset + y, 0]
    if line < 0: return
    sdim = state.shape[-1]
    if x >= sdim: return
    vector = x
    a0 = tdata[y, 0, vector // 8]
    a1 = tdata[y, 1, vector // 8]
    a2 = tdata[y, 2, vector // 8]
    m = np.uint8(1 << (7 - (vector % 8)))
    toggle = 0
    if a1 & m:
        state[line + toggle, x] = TMIN
        toggle += 1
    if (a2 & m) and ((a0 & m) != (a1 & m)):
        state[line + toggle, x] = time
        toggle += 1
    state[line + toggle, x] = TMAX


@cuda.jit(device=True)
def rand_gauss_dev(seed, sd):
    clamp = 0.5
    if sd <= 0.0:
        return 1.0
    while True:
        x = -6.0
        for _ in range(12):
            seed = int(0xDEECE66D) * seed + 0xB
            x += float((seed >> 8) & 0xffffff) / float(1 << 24)
        x *= sd
        if abs(x) <= clamp:
            break
    return x + 1.0


@cuda.jit()
def wave_kernel(ops, op_start, op_stop, state, sat, st_start, st_stop, line_times, sd, seed):
    x, y = cuda.grid(2)
    st_idx = st_start + x
    op_idx = op_start + y
    if st_idx >= st_stop: return
    if op_idx >= op_stop: return
    lut = ops[op_idx, 0]
    z_idx = ops[op_idx, 1]
    a_idx = ops[op_idx, 2]
    b_idx = ops[op_idx, 3]
    overflows = int(0)

    _seed = (seed << 4) + (z_idx << 20) + (st_idx << 1)

    a_mem = sat[a_idx, 0]
    b_mem = sat[b_idx, 0]
    z_mem, z_cap, _ = sat[z_idx]

    a_cur = int(0)
    b_cur = int(0)
    z_cur = lut & 1
    if z_cur == 1:
        state[z_mem, st_idx] = TMIN

    a = state[a_mem, st_idx] + line_times[a_idx, 0, z_cur] * rand_gauss_dev(_seed ^ a_mem ^ z_cur, sd)
    b = state[b_mem, st_idx] + line_times[b_idx, 0, z_cur] * rand_gauss_dev(_seed ^ b_mem ^ z_cur, sd)

    previous_t = TMIN

    current_t = min(a, b)
    inputs = int(0)

    while current_t < TMAX:
        z_val = z_cur & 1
        if b < a:
            b_cur += 1
            b = state[b_mem + b_cur, st_idx]
            b += line_times[b_idx, 0, z_val ^ 1] * rand_gauss_dev(_seed ^ b_mem ^ z_val ^ 1, sd)
            thresh = line_times[b_idx, 1, z_val] * rand_gauss_dev(_seed ^ b_mem ^ z_val, sd)
            inputs ^= 2
            next_t = b
        else:
            a_cur += 1
            a = state[a_mem + a_cur, st_idx]
            a += line_times[a_idx, 0, z_val ^ 1] * rand_gauss_dev(_seed ^ a_mem ^ z_val ^ 1, sd)
            thresh = line_times[a_idx, 1, z_val] * rand_gauss_dev(_seed ^ a_mem ^ z_val, sd)
            inputs ^= 1
            next_t = a

        if (z_cur & 1) != ((lut >> inputs) & 1):
            # we generate a toggle in z_mem, if:
            #   ( it is the first toggle in z_mem OR
            #   following toggle is earlier OR
            #   pulse is wide enough ) AND enough space in z_mem.
            if z_cur == 0 or next_t < current_t or (current_t - previous_t) > thresh:
                if z_cur < (z_cap - 1):
                    state[z_mem + z_cur, st_idx] = current_t
                    previous_t = current_t
                    z_cur += 1
                else:
                    overflows += 1
                    previous_t = state[z_mem + z_cur - 1, st_idx]
                    z_cur -= 1
            else:
                z_cur -= 1
                if z_cur > 0:
                    previous_t = state[z_mem + z_cur - 1, st_idx]
                else:
                    previous_t = TMIN
        current_t = min(a, b)

    if overflows > 0:
        state[z_mem + z_cur, st_idx] = TMAX_OVL
    else:
        state[z_mem + z_cur, st_idx] = a if a > b else b  # propagate overflow flags by storing biggest TMAX from input
