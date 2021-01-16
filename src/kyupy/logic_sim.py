"""A high-throughput combinational logic simulator.

The class :py:class:`~kyupy.logic_sim.LogicSim` performs parallel simulations of the combinational part of a circuit.
The logic operations are performed bit-parallel on packed numpy arrays.
Simple sequential circuits can be simulated by repeated assignments and propagations.
However, this simulator ignores the clock network and simply assumes that all state-elements are clocked all the time.
"""

import math

import numpy as np

from . import logic, hr_bytes


class LogicSim:
    """A bit-parallel naïve combinational simulator for 2-, 4-, or 8-valued logic.

    :param circuit: The circuit to simulate.
    :type circuit: :py:class:`~kyupy.circuit.Circuit`
    :param sims: The number of parallel logic simulations to perform.
    :type sims: int
    :param m: The arity of the logic, must be 2, 4, or 8.
    :type m: int
    """
    def __init__(self, circuit, sims=8, m=8):
        assert m in [2, 4, 8]
        self.m = m
        mdim = math.ceil(math.log2(m))
        self.circuit = circuit
        self.sims = sims
        nbytes = (sims - 1) // 8 + 1
        self.interface = list(circuit.interface) + [n for n in circuit.nodes if 'dff' in n.kind.lower()]
        self.width = len(self.interface)
        """The number of bits in the circuit state (number of ports + number of state-elements)."""
        self.state = np.zeros((len(circuit.lines), mdim, nbytes), dtype='uint8')
        self.state_epoch = np.zeros(len(circuit.nodes), dtype='int8') - 1
        self.tmp = np.zeros((5, mdim, nbytes), dtype='uint8')
        self.zero = np.zeros((mdim, nbytes), dtype='uint8')
        self.epoch = 0

        known_fct = [(f[:-4], getattr(self, f)) for f in dir(self) if f.endswith('_fct')]
        self.node_fct = []
        for n in circuit.nodes:
            t = n.kind.lower().replace('__fork__', 'fork')
            t = t.replace('nbuff', 'fork')
            t = t.replace('input', 'fork')
            t = t.replace('output', 'fork')
            t = t.replace('__const0__', 'const0')
            t = t.replace('__const1__', 'const1')
            t = t.replace('tieh', 'const1')
            t = t.replace('ibuff', 'not')
            t = t.replace('inv', 'not')

            fcts = [f for n, f in known_fct if t.startswith(n)]
            if len(fcts) < 1:
                raise ValueError(f'Unknown node kind {n.kind}')
            self.node_fct.append(fcts[0])

    def __repr__(self):
        return f'<LogicSim {self.circuit.name} sims={self.sims} m={self.m} state_mem={hr_bytes(self.state.nbytes)}>'

    def assign(self, stimuli):
        """Assign stimuli to the primary inputs and state-elements (flip-flops).

        :param stimuli: The input data to assign. Must be in bit-parallel storage format and in a compatible shape.
        :type stimuli: :py:class:`~kyupy.logic.BPArray`
        :returns: The given stimuli object.
        """
        for node, stim in zip(self.interface, stimuli.data if hasattr(stimuli, 'data') else stimuli):
            if len(node.outs) == 0: continue
            outputs = [self.state[line] if line else self.tmp[3] for line in node.outs]
            self.node_fct[node]([stim], outputs)
            for line in node.outs:
                if line is not None: self.state_epoch[line.reader] = self.epoch
        for n in self.circuit.nodes:
            if n.kind in ('__const1__', '__const0__'):
                outputs = [self.state[line] if line else self.tmp[3] for line in n.outs]
                self.node_fct[n]([], outputs)
                for line in n.outs:
                    if line is not None: self.state_epoch[line.reader] = self.epoch
        return stimuli

    def capture(self, responses):
        """Capture the current values at the primary outputs and in the state-elements (flip-flops).

        :param responses: A bit-parallel storage target for the responses in a compatible shape.
        :type responses: :py:class:`~kyupy.logic.BPArray`
        :returns: The given responses object.
        """
        for node, resp in zip(self.interface, responses.data if hasattr(responses, 'data') else responses):
            if len(node.ins) > 0: resp[...] = self.state[node.ins[0]]
        return responses

    def propagate(self, inject_cb=None):
        """Propagate the input values towards the outputs (Perform all logic operations in topological order).

        If the circuit is sequential (it contains flip-flops), one call simulates one clock cycle.
        Multiple clock cycles are simulated by a assign-propagate-capture loop:

        .. code-block:: python

           # initial state in state_bp
           for cycle in range(10):  # simulate 10 clock cycles
               sim.assign(state_bp)
               sim.propagate()
               sim.capture(state_bp)

        :param inject_cb: A callback function for manipulating intermediate signal values.
            This function is called with a line index and its new logic values (in bit-parallel format) after
            evaluation of a node. The callback may manipulate the given values in-place, the simulation
            resumes with the manipulated values after the callback returns.
        :type inject_cb: ``f(int, ndarray)``
        """
        for node in self.circuit.topological_order():
            if self.state_epoch[node] != self.epoch: continue
            inputs = [self.state[line] if line else self.zero for line in node.ins]
            outputs = [self.state[line] if line else self.tmp[3] for line in node.outs]
            # print('sim', node)
            self.node_fct[node](inputs, outputs)
            for line in node.outs:
                if inject_cb is not None: inject_cb(line, self.state[line])
                self.state_epoch[line.reader] = self.epoch
        self.epoch = (self.epoch + 1) % 128

    def cycle(self, state, inject_cb=None):
        """Assigns the given state, propagates it and captures the new state.

        :param state: A bit-parallel array in a compatible shape holding the current circuit state.
            The contained data is assigned to the PI and PPI and overwritten by data at the PO and PPO after
            propagation.
        :type state: :py:class:`~kyupy.logic.BPArray`
        :param inject_cb: A callback function for manipulating intermediate signal values. See :py:func:`propagate`.
        :returns: The given state object.
        """
        self.assign(state)
        self.propagate(inject_cb)
        return self.capture(state)

    @staticmethod
    def fork_fct(inputs, outputs):
        for o in outputs: o[...] = inputs[0]

    @staticmethod
    def const0_fct(_, outputs):
        for o in outputs: o[...] = 0

    @staticmethod
    def const1_fct(_, outputs):
        for o in outputs:
            o[...] = 0
            logic.bp_not(o, o)

    @staticmethod
    def not_fct(inputs, outputs):
        logic.bp_not(outputs[0], inputs[0])

    @staticmethod
    def and_fct(inputs, outputs):
        logic.bp_and(outputs[0], *inputs)

    @staticmethod
    def or_fct(inputs, outputs):
        logic.bp_or(outputs[0], *inputs)

    @staticmethod
    def xor_fct(inputs, outputs):
        logic.bp_xor(outputs[0], *inputs)

    @staticmethod
    def sdff_fct(inputs, outputs):
        logic.bp_buf(outputs[0], inputs[0])
        if len(outputs) > 1:
            logic.bp_not(outputs[1], inputs[0])

    @staticmethod
    def dff_fct(inputs, outputs):
        logic.bp_buf(outputs[0], inputs[0])
        if len(outputs) > 1:
            logic.bp_not(outputs[1], inputs[0])

    @staticmethod
    def nand_fct(inputs, outputs):
        logic.bp_and(outputs[0], *inputs)
        logic.bp_not(outputs[0], outputs[0])

    @staticmethod
    def nor_fct(inputs, outputs):
        logic.bp_or(outputs[0], *inputs)
        logic.bp_not(outputs[0], outputs[0])

    @staticmethod
    def xnor_fct(inputs, outputs):
        logic.bp_xor(outputs[0], *inputs)
        logic.bp_not(outputs[0], outputs[0])
