"""A high-throughput combinational logic simulator.

The class :py:class:`~kyupy.logic_sim.LogicSim` performs parallel simulations of the combinational part of a circuit.
The logic operations are performed bit-parallel on packed numpy arrays (see bit-parallel (bp) array description in :py:mod:`~kyupy.logic`).
Simple sequential circuits can be simulated by repeated assignments and propagations.
However, this simulator ignores the clock network and simply assumes that all state-elements are clocked all the time.
"""

import math

import numpy as np

from . import numba, logic, hr_bytes, sim
from .circuit import Circuit

class LogicSim(sim.SimOps):
    """A bit-parallel naïve combinational simulator for 2-, 4-, or 8-valued logic.

    :param circuit: The circuit to simulate.
    :param sims: The number of parallel logic simulations to perform.
    :param m: The arity of the logic, must be 2, 4, or 8.
    :param c_reuse: If True, intermediate signal values may get overwritten when not needed anymore to save memory.
    :param strip_forks: If True, forks are not included in the simulation model to save memory and simulation time.
    """
    def __init__(self, circuit: Circuit, sims: int = 8, m: int = 8, c_reuse: bool = False, strip_forks: bool = False):
        assert m in [2, 4, 8]
        super().__init__(circuit, c_reuse=c_reuse, strip_forks=strip_forks)
        self.m = m
        self.mdim = math.ceil(math.log2(m))
        self.sims = sims
        nbytes = (sims - 1) // 8 + 1

        self.c = np.zeros((self.c_len, self.mdim, nbytes), dtype=np.uint8)
        self.s = np.zeros((2, self.s_len, 3, nbytes), dtype=np.uint8)
        """Logic values of the sequential elements (flip-flops) and ports.

        It is a pair of arrays in bit-parallel (bp) storage format:

        * ``s[0]`` Assigned values. Simulator will read (P)PI value from here.
        * ``s[1]`` Result values. Simulator will write (P)PO values here.

        Access this array to assign new values to the (P)PIs or read values from the (P)POs.
        """
        self.s[:,:,1,:] = 255  # unassigned

    def __repr__(self):
        return f'{{name: "{self.circuit.name}", sims: {self.sims}, m: {self.m}, c_bytes: {self.c.nbytes}}}'

    def s_to_c(self):
        """Copies the values from ``s[0]`` the inputs of the combinational portion.
        """
        self.c[self.pippi_c_locs] = self.s[0, self.pippi_s_locs, :self.mdim]

    def c_prop(self, inject_cb=None):
        """Propagate the input values through the combinational circuit towards the outputs.

        Performs all logic operations in topological order.
        If the circuit is sequential (it contains flip-flops), one call simulates one clock cycle.

        :param inject_cb: A callback function for manipulating intermediate signal values.
            This function is called with a line and its new logic values (in bit-parallel format) after
            evaluation of a node. The callback may manipulate the given values in-place, the simulation
            resumes with the manipulated values after the callback returns.
        :type inject_cb: ``f(Line, ndarray)``
        """
        t0 = self.c_locs[self.tmp_idx]
        t1 = self.c_locs[self.tmp2_idx]
        if self.m == 2:
            if inject_cb is None:
                _prop_cpu(self.ops, self.c_locs, self.c)
            else:
                for op, o0, i0, i1, i2, i3 in self.ops[:,:6]:
                    o0, i0, i1, i2, i3 = [self.c_locs[x] for x in (o0, i0, i1, i2, i3)]
                    if op == sim.BUF1: self.c[o0]=self.c[i0]
                    elif op == sim.INV1: self.c[o0] = ~self.c[i0]
                    elif op == sim.AND2: self.c[o0] = self.c[i0] & self.c[i1]
                    elif op == sim.AND3: self.c[o0] = self.c[i0] & self.c[i1] & self.c[i2]
                    elif op == sim.AND4: self.c[o0] = self.c[i0] & self.c[i1] & self.c[i2] & self.c[i3]
                    elif op == sim.NAND2: self.c[o0] = ~(self.c[i0] & self.c[i1])
                    elif op == sim.NAND3: self.c[o0] = ~(self.c[i0] & self.c[i1] & self.c[i2])
                    elif op == sim.NAND4: self.c[o0] = ~(self.c[i0] & self.c[i1] & self.c[i2] & self.c[i3])
                    elif op == sim.OR2: self.c[o0] = self.c[i0] | self.c[i1]
                    elif op == sim.OR3: self.c[o0] = self.c[i0] | self.c[i1] | self.c[i2]
                    elif op == sim.OR4: self.c[o0] = self.c[i0] | self.c[i1] | self.c[i2] | self.c[i3]
                    elif op == sim.NOR2: self.c[o0] = ~(self.c[i0] | self.c[i1])
                    elif op == sim.NOR3: self.c[o0] = ~(self.c[i0] | self.c[i1] | self.c[i2])
                    elif op == sim.NOR4: self.c[o0] = ~(self.c[i0] | self.c[i1] | self.c[i2] | self.c[i3])
                    elif op == sim.XOR2: self.c[o0] = self.c[i0] ^ self.c[i1]
                    elif op == sim.XOR3: self.c[o0] = self.c[i0] ^ self.c[i1] ^ self.c[i2]
                    elif op == sim.XOR4: self.c[o0] = self.c[i0] ^ self.c[i1] ^ self.c[i2] ^ self.c[i3]
                    elif op == sim.XNOR2: self.c[o0] = ~(self.c[i0] ^ self.c[i1])
                    elif op == sim.XNOR3: self.c[o0] = ~(self.c[i0] ^ self.c[i1] ^ self.c[i2])
                    elif op == sim.XNOR4: self.c[o0] = ~(self.c[i0] ^ self.c[i1] ^ self.c[i2] ^ self.c[i3])
                    elif op == sim.AO21: self.c[o0] = (self.c[i0] & self.c[i1]) | self.c[i2]
                    elif op == sim.AOI21: self.c[o0] = ~((self.c[i0] & self.c[i1]) | self.c[i2])
                    elif op == sim.OA21: self.c[o0] = (self.c[i0] | self.c[i1]) & self.c[i2]
                    elif op == sim.OAI21: self.c[o0] = ~((self.c[i0] | self.c[i1]) & self.c[i2])
                    elif op == sim.AO22: self.c[o0] = (self.c[i0] & self.c[i1]) | (self.c[i2] & self.c[i3])
                    elif op == sim.AOI22: self.c[o0] = ~((self.c[i0] & self.c[i1]) | (self.c[i2] & self.c[i3]))
                    elif op == sim.OA22: self.c[o0] = (self.c[i0] | self.c[i1]) & (self.c[i2] | self.c[i3])
                    elif op == sim.OAI22: self.c[o0] = ~((self.c[i0] | self.c[i1]) & (self.c[i2] | self.c[i3]))
                    elif op == sim.AO211: self.c[o0] =  (self.c[i0] & self.c[i1]) | self.c[i2] | self.c[i3]
                    elif op == sim.AOI211:self.c[o0] = ~((self.c[i0] & self.c[i1]) | self.c[i2] | self.c[i3])
                    elif op == sim.OA211: self.c[o0] =  (self.c[i0] | self.c[i1]) & self.c[i2] & self.c[i3]
                    elif op == sim.OAI211:self.c[o0] = ~((self.c[i0] | self.c[i1]) & self.c[i2] & self.c[i3])
                    elif op == sim.MUX21: self.c[o0] = (self.c[i0] & ~self.c[i2]) | (self.c[i1] & self.c[i2])
                    else: print(f'unknown op {op}')
                    inject_cb(o0, self.s[o0])
        elif self.m == 4:
            for op, o0, i0, i1, i2, i3 in self.ops[:,:6]:
                o0, i0, i1, i2, i3 = [self.c_locs[x] for x in (o0, i0, i1, i2, i3)]
                if op == sim.BUF1: self.c[o0]=self.c[i0]
                elif op == sim.INV1: logic.bp4v_not(self.c[o0], self.c[i0])
                elif op == sim.AND2: logic.bp4v_and(self.c[o0], self.c[i0], self.c[i1])
                elif op == sim.AND3: logic.bp4v_and(self.c[o0], self.c[i0], self.c[i1], self.c[i2])
                elif op == sim.AND4: logic.bp4v_and(self.c[o0], self.c[i0], self.c[i1], self.c[i2], self.c[i3])
                elif op == sim.NAND2: logic.bp4v_and(self.c[o0], self.c[i0], self.c[i1]); logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.NAND3: logic.bp4v_and(self.c[o0], self.c[i0], self.c[i1], self.c[i2]); logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.NAND4: logic.bp4v_and(self.c[o0], self.c[i0], self.c[i1], self.c[i2], self.c[i3]); logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.OR2: logic.bp4v_or(self.c[o0], self.c[i0], self.c[i1])
                elif op == sim.OR3: logic.bp4v_or(self.c[o0], self.c[i0], self.c[i1], self.c[i2])
                elif op == sim.OR4: logic.bp4v_or(self.c[o0], self.c[i0], self.c[i1], self.c[i2], self.c[i3])
                elif op == sim.NOR2: logic.bp4v_or(self.c[o0], self.c[i0], self.c[i1]); logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.NOR3: logic.bp4v_or(self.c[o0], self.c[i0], self.c[i1], self.c[i2]); logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.NOR4: logic.bp4v_or(self.c[o0], self.c[i0], self.c[i1], self.c[i2], self.c[i3]); logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.XOR2: logic.bp4v_xor(self.c[o0], self.c[i0], self.c[i1])
                elif op == sim.XOR3: logic.bp4v_xor(self.c[o0], self.c[i0], self.c[i1], self.c[i2])
                elif op == sim.XOR4: logic.bp4v_xor(self.c[o0], self.c[i0], self.c[i1], self.c[i2], self.c[i3])
                elif op == sim.XNOR2: logic.bp4v_xor(self.c[o0], self.c[i0], self.c[i1]); logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.XNOR3: logic.bp4v_xor(self.c[o0], self.c[i0], self.c[i1], self.c[i2]); logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.XNOR4: logic.bp4v_xor(self.c[o0], self.c[i0], self.c[i1], self.c[i2], self.c[i3]); logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.AO21:
                    logic.bp4v_and(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp4v_or(self.c[o0], self.c[t0], self.c[i2])
                elif op == sim.AOI21:
                    logic.bp4v_and(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp4v_or(self.c[o0], self.c[t0], self.c[i2])
                    logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.OA21:
                    logic.bp4v_or(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp4v_and(self.c[o0], self.c[t0], self.c[i2])
                elif op == sim.OAI21:
                    logic.bp4v_or(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp4v_and(self.c[o0], self.c[t0], self.c[i2])
                    logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.AO22:
                    logic.bp4v_and(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp4v_and(self.c[t1], self.c[i2], self.c[i3])
                    logic.bp4v_or(self.c[o0], self.c[t0], self.c[t1])
                elif op == sim.AOI22:
                    logic.bp4v_and(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp4v_and(self.c[t1], self.c[i2], self.c[i3])
                    logic.bp4v_or(self.c[o0], self.c[t0], self.c[t1])
                    logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.OA22:
                    logic.bp4v_or(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp4v_or(self.c[t1], self.c[i2], self.c[i3])
                    logic.bp4v_and(self.c[o0], self.c[t0], self.c[t1])
                elif op == sim.OAI22:
                    logic.bp4v_or(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp4v_or(self.c[t1], self.c[i2], self.c[i3])
                    logic.bp4v_and(self.c[o0], self.c[t0], self.c[t1])
                    logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.AO211:
                    logic.bp4v_and(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp4v_or(self.c[o0], self.c[t0], self.c[i2], self.c[i3])
                elif op == sim.AOI211:
                    logic.bp4v_and(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp4v_or(self.c[o0], self.c[t0], self.c[i2], self.c[i3])
                    logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.OA211:
                    logic.bp4v_or(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp4v_and(self.c[o0], self.c[t0], self.c[i2], self.c[i3])
                elif op == sim.OAI211:
                    logic.bp4v_or(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp4v_and(self.c[o0], self.c[t0], self.c[i2], self.c[i3])
                    logic.bp4v_not(self.c[o0], self.c[o0])
                elif op == sim.MUX21:
                    logic.bp4v_not(self.c[t1], self.c[i2])
                    logic.bp4v_and(self.c[t0], self.c[i0], self.c[t1])
                    logic.bp4v_and(self.c[t1], self.c[i1], self.c[i2])
                    logic.bp4v_or(self.c[o0], self.c[t0], self.c[t1])
                else: print(f'unknown op {op}')
        else:
            for op, o0, i0, i1, i2, i3 in self.ops[:,:6]:
                o0, i0, i1, i2, i3 = [self.c_locs[x] for x in (o0, i0, i1, i2, i3)]
                if op == sim.BUF1: self.c[o0]=self.c[i0]
                elif op == sim.INV1: logic.bp8v_not(self.c[o0], self.c[i0])
                elif op == sim.AND2: logic.bp8v_and(self.c[o0], self.c[i0], self.c[i1])
                elif op == sim.AND3: logic.bp8v_and(self.c[o0], self.c[i0], self.c[i1], self.c[i2])
                elif op == sim.AND4: logic.bp8v_and(self.c[o0], self.c[i0], self.c[i1], self.c[i2], self.c[i3])
                elif op == sim.NAND2: logic.bp8v_and(self.c[o0], self.c[i0], self.c[i1]); logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.NAND3: logic.bp8v_and(self.c[o0], self.c[i0], self.c[i1], self.c[i2]); logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.NAND4: logic.bp8v_and(self.c[o0], self.c[i0], self.c[i1], self.c[i2], self.c[i3]); logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.OR2: logic.bp8v_or(self.c[o0], self.c[i0], self.c[i1])
                elif op == sim.OR3: logic.bp8v_or(self.c[o0], self.c[i0], self.c[i1], self.c[i2])
                elif op == sim.OR4: logic.bp8v_or(self.c[o0], self.c[i0], self.c[i1], self.c[i2], self.c[i3])
                elif op == sim.NOR2: logic.bp8v_or(self.c[o0], self.c[i0], self.c[i1]); logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.NOR3: logic.bp8v_or(self.c[o0], self.c[i0], self.c[i1], self.c[i2]); logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.NOR4: logic.bp8v_or(self.c[o0], self.c[i0], self.c[i1], self.c[i2], self.c[i3]); logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.XOR2: logic.bp8v_xor(self.c[o0], self.c[i0], self.c[i1])
                elif op == sim.XOR3: logic.bp8v_xor(self.c[o0], self.c[i0], self.c[i1], self.c[i2])
                elif op == sim.XOR4: logic.bp8v_xor(self.c[o0], self.c[i0], self.c[i1], self.c[i2], self.c[i3])
                elif op == sim.XNOR2: logic.bp8v_xor(self.c[o0], self.c[i0], self.c[i1]); logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.XNOR3: logic.bp8v_xor(self.c[o0], self.c[i0], self.c[i1], self.c[i2]); logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.XNOR4: logic.bp8v_xor(self.c[o0], self.c[i0], self.c[i1], self.c[i2], self.c[i3]); logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.AO21:
                    logic.bp8v_and(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp8v_or(self.c[o0], self.c[t0], self.c[i2])
                elif op == sim.AOI21:
                    logic.bp8v_and(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp8v_or(self.c[o0], self.c[t0], self.c[i2])
                    logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.OA21:
                    logic.bp8v_or(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp8v_and(self.c[o0], self.c[t0], self.c[i2])
                elif op == sim.OAI21:
                    logic.bp8v_or(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp8v_and(self.c[o0], self.c[t0], self.c[i2])
                    logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.AO22:
                    logic.bp8v_and(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp8v_and(self.c[t1], self.c[i2], self.c[i3])
                    logic.bp8v_or(self.c[o0], self.c[t0], self.c[t1])
                elif op == sim.AOI22:
                    logic.bp8v_and(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp8v_and(self.c[t1], self.c[i2], self.c[i3])
                    logic.bp8v_or(self.c[o0], self.c[t0], self.c[t1])
                    logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.OA22:
                    logic.bp8v_or(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp8v_or(self.c[t1], self.c[i2], self.c[i3])
                    logic.bp8v_and(self.c[o0], self.c[t0], self.c[t1])
                elif op == sim.OAI22:
                    logic.bp8v_or(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp8v_or(self.c[t1], self.c[i2], self.c[i3])
                    logic.bp8v_and(self.c[o0], self.c[t0], self.c[t1])
                    logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.AO211:
                    logic.bp8v_and(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp8v_or(self.c[o0], self.c[t0], self.c[i2], self.c[i3])
                elif op == sim.AOI211:
                    logic.bp8v_and(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp8v_or(self.c[o0], self.c[t0], self.c[i2], self.c[i3])
                    logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.OA211:
                    logic.bp8v_or(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp8v_and(self.c[o0], self.c[t0], self.c[i2], self.c[i3])
                elif op == sim.OAI211:
                    logic.bp8v_or(self.c[t0], self.c[i0], self.c[i1])
                    logic.bp8v_and(self.c[o0], self.c[t0], self.c[i2], self.c[i3])
                    logic.bp8v_not(self.c[o0], self.c[o0])
                elif op == sim.MUX21:
                    logic.bp8v_not(self.c[t1], self.c[i2])
                    logic.bp8v_and(self.c[t0], self.c[i0], self.c[t1])
                    logic.bp8v_and(self.c[t1], self.c[i1], self.c[i2])
                    logic.bp8v_or(self.c[o0], self.c[t0], self.c[t1])
                else: print(f'unknown op {op}')
                if inject_cb is not None: inject_cb(o0, self.s[o0])

    def c_to_s(self):
        """Copies (captures) the results of the combinational portion to ``s[1]``.
        """
        self.s[1, self.poppo_s_locs, :self.mdim] = self.c[self.poppo_c_locs]
        if self.mdim == 1:
            self.s[1, self.poppo_s_locs, 1:2] = self.c[self.poppo_c_locs]

    def s_ppo_to_ppi(self):
        """Constructs a new assignment based on the current data in ``s``.

        Use this function for simulating consecutive clock cycles.

        For 2-valued or 4-valued simulations, all valued from PPOs (in ``s[1]``) and copied to the PPIs (in ``s[0]``).
        For 8-valued simulations, PPI transitions are constructed from the final values of the assignment (in ``s[0]``) and the
        final values of the results (in ``s[1]``).
        """
        # TODO: handle latches correctly
        if self.mdim < 3:
            self.s[0, self.ppio_s_locs] = self.s[1, self.ppio_s_locs]
        else:
            self.s[0, self.ppio_s_locs, 1] = self.s[0, self.ppio_s_locs, 0]  # initial value is previously assigned final value
            self.s[0, self.ppio_s_locs, 0] = self.s[1, self.ppio_s_locs, 0]  # final value is newly captured final value
            self.s[0, self.ppio_s_locs, 2] = self.s[0, self.ppio_s_locs, 0] ^ self.s[0, self.ppio_s_locs, 1]  # TODO: not correct for X, -

    def cycle(self, cycles: int = 1, inject_cb=None):
        """Repeatedly assigns a state, propagates it, captures the new state, and transfers PPOs to PPIs.

        :param cycles: The number of cycles to simulate.
        :param inject_cb: A callback function for manipulating intermediate signal values. See :py:func:`c_prop`.
        """
        for _ in range(cycles):
            self.s_to_c()
            self.c_prop(inject_cb)
            self.c_to_s()
            self.s_ppo_to_ppi()


@numba.njit
def _prop_cpu(ops, c_locs, c):
    for op, o0, i0, i1, i2, i3 in ops[:,:6]:
        o0, i0, i1, i2, i3 = [c_locs[x] for x in (o0, i0, i1, i2, i3)]
        if op == sim.BUF1: c[o0]=c[i0]
        elif op == sim.INV1: c[o0] = ~c[i0]
        elif op == sim.AND2: c[o0] = c[i0] & c[i1]
        elif op == sim.AND3: c[o0] = c[i0] & c[i1] & c[i2]
        elif op == sim.AND4: c[o0] = c[i0] & c[i1] & c[i2] & c[i3]
        elif op == sim.NAND2: c[o0] = ~(c[i0] & c[i1])
        elif op == sim.NAND3: c[o0] = ~(c[i0] & c[i1] & c[i2])
        elif op == sim.NAND4: c[o0] = ~(c[i0] & c[i1] & c[i2] & c[i3])
        elif op == sim.OR2: c[o0] = c[i0] | c[i1]
        elif op == sim.OR3: c[o0] = c[i0] | c[i1] | c[i2]
        elif op == sim.OR4: c[o0] = c[i0] | c[i1] | c[i2] | c[i3]
        elif op == sim.NOR2: c[o0] = ~(c[i0] | c[i1])
        elif op == sim.NOR3: c[o0] = ~(c[i0] | c[i1] | c[i2])
        elif op == sim.NOR4: c[o0] = ~(c[i0] | c[i1] | c[i2] | c[i3])
        elif op == sim.XOR2: c[o0] = c[i0] ^ c[i1]
        elif op == sim.XOR3: c[o0] = c[i0] ^ c[i1] ^ c[i2]
        elif op == sim.XOR4: c[o0] = c[i0] ^ c[i1] ^ c[i2] ^ c[i3]
        elif op == sim.XNOR2: c[o0] = ~(c[i0] ^ c[i1])
        elif op == sim.XNOR3: c[o0] = ~(c[i0] ^ c[i1] ^ c[i2])
        elif op == sim.XNOR4: c[o0] = ~(c[i0] ^ c[i1] ^ c[i2] ^ c[i3])
        elif op == sim.AO21: c[o0] = (c[i0] & c[i1]) | c[i2]
        elif op == sim.OA21: c[o0] = (c[i0] | c[i1]) & c[i2]
        elif op == sim.AO22: c[o0] = (c[i0] & c[i1]) | (c[i2] & c[i3])
        elif op == sim.OA22: c[o0] = (c[i0] | c[i1]) & (c[i2] | c[i3])
        elif op == sim.AOI21: c[o0] = ~((c[i0] & c[i1]) | c[i2])
        elif op == sim.OAI21: c[o0] = ~((c[i0] | c[i1]) & c[i2])
        elif op == sim.AOI22: c[o0] = ~((c[i0] & c[i1]) | (c[i2] & c[i3]))
        elif op == sim.OAI22: c[o0] = ~((c[i0] | c[i1]) & (c[i2] | c[i3]))
        elif op == sim.AO211: c[o0] = (c[i0] & c[i1]) | c[i2] | c[i3]
        elif op == sim.OA211: c[o0] = (c[i0] | c[i1]) & c[i2] & c[i3]
        elif op == sim.AOI211: c[o0] = ~((c[i0] & c[i1]) | c[i2] | c[i3])
        elif op == sim.OAI211: c[o0] = ~((c[i0] | c[i1]) & c[i2] & c[i3])
        elif op == sim.MUX21: c[o0] = (c[i0] & ~c[i2]) | (c[i1] & c[i2])
        else: print(f'unknown op {op}')
