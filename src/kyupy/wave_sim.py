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

import numpy as np

from . import numba, cuda, sim, cdiv


TMAX = np.float32(2 ** 127)
"""A large 32-bit floating point value used to mark the end of a waveform."""
TMAX_OVL = np.float32(1.1 * 2 ** 127)
"""A large 32-bit floating point value used to mark the end of a waveform that
may be incomplete due to an overflow."""
TMIN = np.float32(-2 ** 127)
"""A large negative 32-bit floating point value used at the beginning of waveforms that start with logic-1."""


class WaveSim(sim.SimOps):
    """A waveform-based combinational logic timing simulator running on CPU.

    :param circuit: The circuit to simulate.
    :param delays: One or more delay annotations for the circuit (see :py:func:`kyupy.sdf.DelayFile.iopaths` for details).
        Each parallel simulation may use the same delays or different delays, depending on the use-case (see :py:attr:`simctl_int`).
    :param sims: The number of parallel simulations.
    :param c_caps: The number of floats available in each waveform. Values must be positive and a multiple of 4.
        Waveforms encode the signal switching history by storing transition times.
        The waveform capacity roughly corresponds to the number of transitions
        that can be stored. A capacity of ``n`` can store at least ``n-2`` transitions. If more transitions are
        generated during simulation, the latest glitch is removed (freeing up two transition times) and an overflow
        flag is set. If an integer is given, all waveforms are set to that same capacity. With an array of length
        ``len(circuit.lines)`` the capacity is set individually for each intermediate waveform.
    :param a_ctrl: An integer array controlling the accumulation of weighted switching activity during simulation.
        Its shape must be ``(len(circuit.lines), 3)``. ``a_ctrl[...,0]`` is the index into the accumulation buffer, -1 means ignore.
        ``a_ctrl[...,1]`` is the (integer) weight for a rising transition, ``a_ctrl[...,2]`` is the (integer) weight for
        a falling transition. The accumulation buffer (:py:attr:`abuf`) is allocated automatically if ``a_ctrl`` is given.
    :param c_reuse: If enabled, memory of intermediate signal waveforms will be re-used. This greatly reduces
        memory footprint, but intermediate signal waveforms may become unaccessible after a propagation.
    :param strip_forks: If enabled, the simulator will not evaluate fork nodes explicitly. This saves simulation time
        and memory by reducing the number of nodes to simulate, but (interconnect) delay annotations of lines read by fork nodes
        are ignored.
    """
    def __init__(self, circuit, delays, sims=8, c_caps=16, a_ctrl=None, c_reuse=False, strip_forks=False):
        super().__init__(circuit, c_caps=c_caps, c_caps_min=4, a_ctrl=a_ctrl, c_reuse=c_reuse, strip_forks=strip_forks)
        self.sims = sims
        if delays.ndim == 3: delays = np.expand_dims(delays, axis=0)
        self.delays = np.zeros((len(delays), self.c_locs_len, 2, 2), dtype=delays.dtype)
        self.delays[:, :delays.shape[1]] = delays

        self.c = np.zeros((self.c_len, sims), dtype=np.float32) + TMAX
        self.s = np.zeros((11, self.s_len, sims), dtype=np.float32)
        """Information about the logic values and transitions around the sequential elements (flip-flops) and ports.

        The first 3 values are read by :py:func:`s_to_c`.
        The remaining values are written by :py:func:`c_to_s`.

        The elements are as follows:

        * ``s[0]`` (P)PI initial value
        * ``s[1]`` (P)PI transition time
        * ``s[2]`` (P)PI final value
        * ``s[3]`` (P)PO initial value
        * ``s[4]`` (P)PO earliest arrival time (EAT): The time at which the output transitioned from its initial value.
        * ``s[5]`` (P)PO latest stabilization time (LST): The time at which the output settled to its final value.
        * ``s[6]`` (P)PO final value
        * ``s[7]`` (P)PO capture value: probability of capturing a 1 at a given capture time
        * ``s[8]`` (P)PO sampled capture value: decided by random sampling according to a given seed.
        * ``s[9]`` (P)PO sampled capture slack: (capture time - LST) - decided by random sampling according to a given seed.
        * ``s[10]`` Overflow indicator: If non-zero, some signals in the input cone of this output had more
          transitions than specified in ``c_caps``. Some transitions have been discarded, the
          final values in the waveforms are still valid.
        """

        self.abuf_len = self.ops[:,6].max() + 1
        self.abuf = np.zeros((self.abuf_len, sims), dtype=np.int32) if self.abuf_len > 0 else np.zeros((1, 1), dtype=np.int32)

        self.simctl_int = np.zeros((2, sims), dtype=np.int32)
        """Integer array for per-simulation delay configuration.

        * ``simctl_int[0]`` delay dataset or random seed for picking a delay. By default, each sim has a unique seed.
        * ``simctl_int[1]`` Method for picking a delay:
            * 0: seed parameter of :py:func:`c_prop` directly specifies dataset for all simulations
            * 1: ``simctl_int[0]`` specifies dataset on a per-simulation basis
            * 2 (default): ``simctl_int[0]`` and seed parameter of :py:func:`c_prop` together are a random seed for picking a delay dataset.
        """
        self.simctl_int[0] = range(sims)  # unique seed for each sim by default, zero this to pick same delays for all sims.
        self.simctl_int[1] = 2  # random picking by default.

        self.nbytes = sum([a.nbytes for a in (self.c, self.s, self.c_locs, self.c_caps, self.ops, self.simctl_int)])

    def __repr__(self):
        dev = 'GPU' if hasattr(self.c, 'copy_to_host') else 'CPU'
        return f'{{name: "{self.circuit.name}", device: "{dev}", sims: {self.sims}, ops: {len(self.ops)}, ' + \
               f'levels: {len(self.level_starts)}, nbytes: {self.nbytes}}}'

    def s_to_c(self):
        """Transfers values of sequential elements and primary inputs to the combinational portion.

        Waveforms are generated on the input lines of the combinational circuit based on the data in :py:attr:`s`.
        """
        sins = self.s[:, self.pippi_s_locs]
        cond = (sins[2] != 0) + 2*(sins[0] != 0)  # choices order: 0 R F 1
        self.c[self.pippi_c_locs] = np.choose(cond, [TMAX, sins[1], TMIN, TMIN])
        self.c[self.pippi_c_locs+1] = np.choose(cond, [TMAX, TMAX, sins[1], TMAX])
        self.c[self.pippi_c_locs+2] = TMAX

    def c_prop(self, sims=None, seed=1):
        """Propagates all waveforms from the (pseudo) primary inputs to the (pseudo) primary outputs.

        :param sims: Number of parallel simulations to execute. If None, all available simulations are performed.
        :param seed: Seed for picking delays. See also: :py:attr:`simctl_int`.
        """
        sims = min(sims or self.sims, self.sims)
        for op_start, op_stop in zip(self.level_starts, self.level_stops):
            level_eval_cpu(self.ops, op_start, op_stop, self.c, self.c_locs, self.c_caps, self.abuf, 0, sims, self.delays, self.simctl_int, seed)

    def c_to_s(self, time=TMAX, sd=0.0, seed=1):
        """Simulates a capture operation at all sequential elements and primary outputs.

        Propagated waveforms at the outputs of the combinational circuit at and around the given capture time are analyzed and
        the results are stored in :py:attr:`s`.

        :param time: The desired capture time. By default, a capture of the settled value is performed.
        :param sd: A standard deviation for uncertainty in the actual capture time.
        :param seed: The random seed for a capture with uncertainty.
        """
        for s_loc, c_loc, c_len in zip(self.poppo_s_locs, self.c_locs[self.ppo_offset+self.poppo_s_locs], self.c_caps[self.ppo_offset+self.poppo_s_locs]):
            for vector in range(self.sims):
                self.s[3:, s_loc, vector] = wave_capture_cpu(self.c, c_loc, c_len, vector, time=time, sd=sd, seed=seed)

    def s_ppo_to_ppi(self, time=0.0):
        """Re-assigns the last sampled capture of the PPOs to the appropriate pseudo-primary inputs (PPIs).
        Each PPI transition is constructed from the final value of the previous assignment, the
        given time, and the sampled captured value of its PPO. Reads and modifies :py:attr:`s`.

        :param time: The transition time at the inputs (usually 0.0).
        """
        self.s[0, self.ppio_s_locs] = self.s[2, self.ppio_s_locs]
        self.s[1, self.ppio_s_locs] = time
        self.s[2, self.ppio_s_locs] = self.s[8, self.ppio_s_locs]


def _wave_eval(op, cbuf, c_locs, c_caps, sim, delays, simctl_int, seed=0):
    overflows = int(0)

    lut = op[0]
    z_idx = op[1]
    a_idx = op[2]
    b_idx = op[3]
    c_idx = op[4]
    d_idx = op[5]

    if len(delays) > 1:
        if simctl_int[1] == 0:
            delays = delays[seed]
        elif simctl_int[1] == 1:
            delays = delays[simctl_int[0]]
        else:
            _rnd = (seed << 4) + (z_idx << 20) + simctl_int[0]
            for _ in range(4):
                _rnd = int(0xDEECE66D) * _rnd + 0xB
            delays = delays[_rnd % len(delays)]
    else:
        delays = delays[0]

    a_mem = c_locs[a_idx]
    b_mem = c_locs[b_idx]
    c_mem = c_locs[c_idx]
    d_mem = c_locs[d_idx]
    z_mem = c_locs[z_idx]
    z_cap = c_caps[z_idx]

    a_cur = int(0)
    b_cur = int(0)
    c_cur = int(0)
    d_cur = int(0)
    z_cur = lut & 1
    if z_cur == 1:
        cbuf[z_mem, sim] = TMIN

    z_val = z_cur

    a = cbuf[a_mem + a_cur, sim] + delays[a_idx, 0, z_val]
    b = cbuf[b_mem + b_cur, sim] + delays[b_idx, 0, z_val]
    c = cbuf[c_mem + c_cur, sim] + delays[c_idx, 0, z_val]
    d = cbuf[d_mem + d_cur, sim] + delays[d_idx, 0, z_val]

    previous_t = TMIN

    current_t = min(a, b, c, d)
    inputs = int(0)

    while current_t < TMAX:
        if a == current_t:
            a_cur += 1
            inputs ^= 1
            thresh = delays[a_idx, a_cur & 1, z_val]
            a = cbuf[a_mem + a_cur, sim] + delays[a_idx, a_cur & 1, z_val]
            next_t = cbuf[a_mem + a_cur, sim] + delays[a_idx, (a_cur & 1) ^ 1, z_val ^ 1]
        elif b == current_t:
            b_cur += 1
            inputs ^= 2
            thresh = delays[b_idx, b_cur & 1, z_val]
            b = cbuf[b_mem + b_cur, sim] + delays[b_idx, b_cur & 1, z_val]
            next_t = cbuf[b_mem + b_cur, sim] + delays[b_idx, (b_cur & 1) ^ 1, z_val ^ 1]
        elif c == current_t:
            c_cur += 1
            inputs ^= 4
            thresh = delays[c_idx, c_cur & 1, z_val]
            c = cbuf[c_mem + c_cur, sim] + delays[c_idx, c_cur & 1, z_val]
            next_t = cbuf[c_mem + c_cur, sim] + delays[c_idx, (c_cur & 1) ^ 1, z_val ^ 1]
        else:
            d_cur += 1
            inputs ^= 8
            thresh = delays[d_idx, d_cur & 1, z_val]
            d = cbuf[d_mem + d_cur, sim] + delays[d_idx, d_cur & 1, z_val]
            next_t = cbuf[d_mem + d_cur, sim] + delays[d_idx, (d_cur & 1) ^ 1, z_val ^ 1]

        if (z_cur & 1) != ((lut >> inputs) & 1):
            # we generate an edge in z_mem, if ...
            if (z_cur == 0                            # it is the first edge in z_mem ...
                or next_t < current_t                 # -OR- the next edge on SAME input is EARLIER (need current edge to filter BOTH in next iteration) ...
                or (current_t - previous_t) > thresh  # -OR- the generated hazard is wider than pulse threshold.
                ):
                if z_cur < (z_cap - 1):  # enough space in z_mem?
                    cbuf[z_mem + z_cur, sim] = current_t
                    previous_t = current_t
                    z_cur += 1
                else:
                    overflows += 1
                    previous_t = cbuf[z_mem + z_cur - 1, sim]
                    z_cur -= 1
            else:
                z_cur -= 1
                previous_t = cbuf[z_mem + z_cur - 1, sim] if z_cur > 0 else TMIN

            # output value of cell changed. update all delayed inputs.
            z_val = z_val ^ 1
            a = cbuf[a_mem + a_cur, sim] + delays[a_idx, a_cur & 1, z_val]
            b = cbuf[b_mem + b_cur, sim] + delays[b_idx, b_cur & 1, z_val]
            c = cbuf[c_mem + c_cur, sim] + delays[c_idx, c_cur & 1, z_val]
            d = cbuf[d_mem + d_cur, sim] + delays[d_idx, d_cur & 1, z_val]

        current_t = min(a, b, c, d)

    # generate or propagate overflow flag
    cbuf[z_mem + z_cur, sim] = TMAX_OVL if overflows > 0 else max(a, b, c, d)

    nrise = max(0, (z_cur+1) // 2 - (cbuf[z_mem, sim] == TMIN))
    nfall = z_cur // 2

    return nrise, nfall


wave_eval_cpu = numba.njit(_wave_eval)


@numba.njit
def level_eval_cpu(ops, op_start, op_stop, c, c_locs, c_caps, abuf, sim_start, sim_stop, delays, simctl_int, seed):
    for op_idx in range(op_start, op_stop):
        op = ops[op_idx]
        for sim in range(sim_start, sim_stop):
            nrise, nfall = wave_eval_cpu(op, c, c_locs, c_caps, sim, delays, simctl_int[:, sim], seed)
            a_loc = op[6]
            a_wr = op[7]
            a_wf = op[8]
            if a_loc >= 0:
                abuf[a_loc, sim] += nrise*a_wr + nfall*a_wf


@numba.njit
def wave_capture_cpu(c, c_loc, c_len, vector, time=TMAX, sd=0.0, seed=1):
    s_sqrt2 = sd * math.sqrt(2)
    m = 0.5
    acc = 0.0
    eat = TMAX
    lst = TMIN
    tog = 0
    ovl = 0
    val = int(0)
    final = int(0)
    w = c[c_loc:c_loc+c_len, vector]
    for t in w:
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
            seed = (seed << 4) + (vector << 20) + c_loc
            seed = int(0xDEECE66D) * seed + 0xB
            seed = int(0xDEECE66D) * seed + 0xB
            rnd = float((seed >> 8) & 0xffffff) / float(1 << 24)
            val = rnd < acc
        else:
            val = 0
    else:
        acc = val

    return (w[0] <= TMIN), eat, lst, final, acc, val, 0, ovl


class WaveSimCuda(WaveSim):
    """A GPU-accelerated waveform-based combinational logic timing simulator.

    The API is identical to :py:class:`WaveSim`. See there for complete documentation.

    All internal memories are mirrored into GPU memory upon construction.
    Some operations like access to single waveforms can involve large communication overheads.
    """
    def __init__(self, circuit, delays, sims=8, c_caps=16, a_ctrl=None, c_reuse=False, strip_forks=False):
        super().__init__(circuit, delays, sims, c_caps, a_ctrl=a_ctrl, c_reuse=c_reuse, strip_forks=strip_forks)

        self.c = cuda.to_device(self.c)
        self.s = cuda.to_device(self.s)
        self.ops = cuda.to_device(self.ops)
        self.c_locs = cuda.to_device(self.c_locs)
        self.c_caps = cuda.to_device(self.c_caps)
        self.delays = cuda.to_device(self.delays)
        self.simctl_int = cuda.to_device(self.simctl_int)
        self.abuf = cuda.to_device(self.abuf)

        self._block_dim = (32, 16)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['c'] = np.array(self.c)
        state['s'] = np.array(self.s)
        state['ops'] = np.array(self.ops)
        state['c_locs'] = np.array(self.c_locs)
        state['c_caps'] = np.array(self.c_caps)
        state['delays'] = np.array(self.delays)
        state['simctl_int'] = np.array(self.simctl_int)
        state['abuf'] = np.array(self.abuf)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.c = cuda.to_device(self.c)
        self.s = cuda.to_device(self.s)
        self.ops = cuda.to_device(self.ops)
        self.c_locs = cuda.to_device(self.c_locs)
        self.c_caps = cuda.to_device(self.c_caps)
        self.delays = cuda.to_device(self.delays)
        self.simctl_int = cuda.to_device(self.simctl_int)
        self.abuf = cuda.to_device(self.abuf)

    def s_to_c(self):
        grid_dim = self._grid_dim(self.sims, self.s_len)
        wave_assign_gpu[grid_dim, self._block_dim](self.c, self.s, self.c_locs, self.ppi_offset)

    def _grid_dim(self, x, y): return cdiv(x, self._block_dim[0]), cdiv(y, self._block_dim[1])

    def c_prop(self, sims=None, seed=1):
        sims = min(sims or self.sims, self.sims)
        for op_start, op_stop in zip(self.level_starts, self.level_stops):
            grid_dim = self._grid_dim(sims, op_stop - op_start)
            wave_eval_gpu[grid_dim, self._block_dim](self.ops, op_start, op_stop, self.c, self.c_locs, self.c_caps, self.abuf, int(0),
                sims, self.delays, self.simctl_int, seed)
        cuda.synchronize()

    def c_to_s(self, time=TMAX, sd=0.0, seed=1):
        grid_dim = self._grid_dim(self.sims, self.s_len)
        wave_capture_gpu[grid_dim, self._block_dim](self.c, self.s, self.c_locs, self.c_caps, self.ppo_offset,
            time, sd * math.sqrt(2), seed)

    def s_ppo_to_ppi(self, time=0.0):
        grid_dim = self._grid_dim(self.sims, self.s_len)
        ppo_to_ppi_gpu[grid_dim, self._block_dim](self.s, self.c_locs, time, self.ppi_offset, self.ppo_offset)


@cuda.jit()
def wave_assign_gpu(c, s, c_locs, ppi_offset):
    x, y = cuda.grid(2)
    if y >= s.shape[1]: return
    c_loc = c_locs[ppi_offset + y]
    if c_loc < 0: return
    if x >= c.shape[-1]: return
    value = int(s[2, y, x] >= 0.5) | (2*int(s[0, y, x] >= 0.5))
    ttime = s[1, y, x]
    if value == 0:
        c[c_loc, x] = TMAX
        c[c_loc+1, x] = TMAX
    elif value == 1:
        c[c_loc, x] = ttime
        c[c_loc+1, x] = TMAX
    elif value == 2:
        c[c_loc, x] = TMIN
        c[c_loc+1, x] = ttime
    else:
        c[c_loc, x] = TMIN
        c[c_loc+1, x] = TMAX
    c[c_loc+2, x] = TMAX


_wave_eval_gpu = cuda.jit(_wave_eval, device=True)


@cuda.jit()
def wave_eval_gpu(ops, op_start, op_stop, cbuf, c_locs, c_caps, abuf, sim_start, sim_stop, delays, simctl_int, seed):
    x, y = cuda.grid(2)
    sim = sim_start + x
    op_idx = op_start + y
    if sim >= sim_stop: return
    if op_idx >= op_stop: return

    op = ops[op_idx]
    a_loc = op[6]
    a_wr = op[7]
    a_wf = op[8]

    nrise, nfall = _wave_eval_gpu(op, cbuf, c_locs, c_caps, sim, delays, simctl_int[:, sim], seed)

    # accumulate WSA into abuf
    if a_loc >= 0:
        cuda.atomic.add(abuf, (a_loc, sim), nrise*a_wr + nfall*a_wf)


@cuda.jit()
def wave_capture_gpu(c, s, c_locs, c_caps, ppo_offset, time, s_sqrt2, seed):
    x, y = cuda.grid(2)
    if ppo_offset + y >= len(c_locs): return
    line = c_locs[ppo_offset + y]
    tdim = c_caps[ppo_offset + y]
    if line < 0: return
    if x >= c.shape[-1]: return
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
        t = c[line + tidx, vector]
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

    s[3, y, vector] = (c[line, vector] <= TMIN)
    s[4, y, vector] = eat
    s[5, y, vector] = lst
    s[6, y, vector] = final
    s[7, y, vector] = acc
    s[8, y, vector] = val
    s[9, y, vector] = 0  # TODO
    s[10, y, vector] = ovl


@cuda.jit()
def ppo_to_ppi_gpu(s, c_locs, time, ppi_offset, ppo_offset):
    x, y = cuda.grid(2)
    if y >= s.shape[1]: return
    if x >= s.shape[2]: return

    if c_locs[ppi_offset + y] < 0: return
    if c_locs[ppo_offset + y] < 0: return

    s[0, y, x] = s[2, y, x]
    s[1, y, x] = time
    s[2, y, x] = s[8, y, x]
