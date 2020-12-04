import numpy as np
import math
from .wave_sim import WaveSim
from . import cuda

TMAX = np.float32(2 ** 127)  # almost np.PINF for 32-bit floating point values
TMAX_OVL = np.float32(1.1 * 2 ** 127)  # almost np.PINF with overflow mark
TMIN = np.float32(-2 ** 127)  # almost np.NINF for 32-bit floating point values


class WaveSimCuda(WaveSim):
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

    def get_line_delay(self, line, polarity):
        return self.d_timing[line, 0, polarity]

    def set_line_delay(self, line, polarity, delay):
        self.d_timing[line, 0, polarity] = delay

    def assign(self, vectors, time=0.0, offset=0):
        assert (offset % 8) == 0
        byte_offset = offset // 8
        assert byte_offset < vectors.bits.shape[-1]
        pdim = min(vectors.bits.shape[-1] - byte_offset, self.tdata.shape[-1])

        self.tdata[..., 0:pdim] = vectors.bits[..., byte_offset:pdim + byte_offset]
        if vectors.vdim == 1:
            self.tdata[:, 1, 0:pdim] = ~self.tdata[:, 1, 0:pdim]
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
        if sims is None:
            sims = self.sims
        else:
            sims = min(sims, self.sims)
        for op_start, op_stop in zip(self.level_starts, self.level_stops):
            grid_dim = self._grid_dim(sims, op_stop - op_start)
            wave_kernel[grid_dim, self._block_dim](self.d_ops, op_start, op_stop, self.d_state, self.sat, int(0),
                                                   sims, self.d_timing, sd, seed)
        cuda.synchronize()
        self.lst_eat_valid = False

    def wave(self, line, vector):
        if line < 0:
            return None
        mem, wcap, _ = self.sat[line]
        if mem < 0:
            return None
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

    ppo, ppo_cap, _ = sat[ppo_offset + y]
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
    if a0 & m:
        state[line + toggle, x] = TMIN
        toggle += 1
    if (a2 & m) and ((a0 & m) == (a1 & m)):
        state[line + toggle, x] = time
        toggle += 1
    state[line + toggle, x] = TMAX


@cuda.jit(device=True)
def rand_gauss(seed, sd):
    clamp = 0.5
    if sd <= 0.0:
        return 1.0
    while True:
        x = -6.0
        for i in range(12):
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
