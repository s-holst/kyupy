import math

import numpy as np

from . import logic


class LogicSim:
    """A bit-parallel naïve combinational simulator for 2-, 4-, or 8-valued logic.
    """
    def __init__(self, circuit, sims=1, m=8):
        assert m in [2, 4, 8]
        self.m = m
        mdim = math.ceil(math.log2(m))
        self.circuit = circuit
        self.sims = sims
        nbytes = (sims - 1) // 8 + 1
        self.interface = list(circuit.interface) + [n for n in circuit.nodes if 'dff' in n.kind.lower()]
        self.state = np.zeros((len(circuit.lines), mdim, nbytes), dtype='uint8')
        self.state_epoch = np.zeros(len(circuit.nodes), dtype='int8') - 1
        self.tmp = np.zeros((5, mdim, nbytes), dtype='uint8')
        self.zero = np.zeros((mdim, nbytes), dtype='uint8')
        self.epoch = 0

        self.fork_vd1 = self.fork_vdx
        self.const0_vd1 = self.const0_vdx
        self.input_vd1 = self.fork_vd1
        self.output_vd1 = self.fork_vd1
        self.inv_vd1 = self.not_vd1
        self.ibuff_vd1 = self.not_vd1
        self.nbuff_vd1 = self.fork_vd1
        self.xor2_vd1 = self.xor_vd1
        
        self.fork_vd2 = self.fork_vdx
        self.const0_vd2 = self.const0_vdx
        self.input_vd2 = self.fork_vd2
        self.output_vd2 = self.fork_vd2
        self.inv_vd2 = self.not_vd2
        self.ibuff_vd2 = self.not_vd2
        self.nbuff_vd2 = self.fork_vd2
        self.xor2_vd2 = self.xor_vd2
        
        self.fork_vd3 = self.fork_vdx
        self.const0_vd3 = self.const0_vdx
        self.input_vd3 = self.fork_vd3
        self.output_vd3 = self.fork_vd3
        self.inv_vd3 = self.not_vd3
        self.ibuff_vd3 = self.not_vd3
        self.nbuff_vd3 = self.fork_vd3
        self.xor2_vd3 = self.xor_vd3
        
        known_fct = [(f[:-4], getattr(self, f)) for f in dir(self) if f.endswith(f'_vd{mdim}')]
        self.node_fct = []
        for n in circuit.nodes:
            t = n.kind.lower().replace('__fork__', 'fork')
            t = t.replace('__const0__', 'const0')
            t = t.replace('__const1__', 'const1')
            t = t.replace('tieh', 'const1')
            fcts = [f for n, f in known_fct if t.startswith(n)]
            if len(fcts) < 1:
                raise ValueError(f'Unknown node kind {n.kind}')
            self.node_fct.append(fcts[0])

    def assign(self, stimuli):
        """Assign stimuli to the primary inputs and state-elements (flip-flops)."""
        if hasattr(stimuli, 'data'):
            stimuli = stimuli.data
        for stim, node in zip(stimuli, self.interface):
            if len(node.outs) == 0: continue
            outputs = [self.state[line.index] if line else self.tmp[3] for line in node.outs]
            self.node_fct[node.index]([stim], outputs)
            for line in node.outs:
                if line:
                    self.state_epoch[line.reader.index] = self.epoch
        for n in self.circuit.nodes:
            if (n.kind == '__const1__') or (n.kind == '__const0__'):
                outputs = [self.state[line.index] if line else self.tmp[3] for line in n.outs]
                self.node_fct[n.index]([], outputs)
                # print('assign const')
                for line in n.outs:
                    if line:
                        self.state_epoch[line.reader.index] = self.epoch

    def capture(self, responses):
        """Capture the current values at the primary outputs and in the state-elements (flip-flops)."""
        if hasattr(responses, 'data'):
            responses = responses.data
        for resp, node in zip(responses, self.interface):
            if len(node.ins) == 0: continue
            resp[...] = self.state[node.ins[0].index]
        # print(responses)

    def propagate(self):
        """Propagate the input values towards the outputs (Perform all logic operations in topological order)."""
        for node in self.circuit.topological_order():
            if self.state_epoch[node.index] != self.epoch: continue
            inputs = [self.state[line.index] if line else self.zero for line in node.ins]
            outputs = [self.state[line.index] if line else self.tmp[3] for line in node.outs]
            # print('sim', node)
            self.node_fct[node.index](inputs, outputs)
            for line in node.outs:
                self.state_epoch[line.reader.index] = self.epoch
        self.epoch = (self.epoch + 1) % 128

    def fork_vdx(self, inputs, outputs):
        for o in outputs: o[...] = inputs[0]
    
    def const0_vdx(self, _, outputs):
        for o in outputs: o[...] = self.zero

    # 2-valued simulation

    def not_vd1(self, inputs, outputs):
        outputs[0][0] = ~inputs[0][0]

    def const1_vd1(self, _, outputs):
        for o in outputs: o[...] = self.zero
        self.not_vd1(outputs, outputs)

    def and_vd1(self, inputs, outputs):
        o = outputs[0]
        o[0] = inputs[0][0]
        for i in inputs[1:]: o[0] &= i[0]

    def or_vd1(self, inputs, outputs):
        o = outputs[0]
        o[0] = inputs[0][0]
        for i in inputs[1:]: o[0] |= i[0]

    def xor_vd1(self, inputs, outputs):
        o = outputs[0]
        o[0] = inputs[0][0]
        for i in inputs[1:]: o[0] ^= i[0]

    def sdff_vd1(self, inputs, outputs):
        outputs[0][0] = inputs[0][0]
        if len(outputs) > 1:
            outputs[1][0] = ~inputs[0][0]

    def dff_vd1(self, inputs, outputs):
        outputs[0][0] = inputs[0][0]
        if len(outputs) > 1:
            outputs[1][0] = ~inputs[0][0]

    def nand_vd1(self, inputs, outputs):
        self.and_vd1(inputs, outputs)
        self.not_vd1(outputs, outputs)

    def nor_vd1(self, inputs, outputs):
        self.or_vd1(inputs, outputs)
        self.not_vd1(outputs, outputs)

    def xnor_vd1(self, inputs, outputs):
        self.xor_vd1(inputs, outputs)
        self.not_vd1(outputs, outputs)

    # 4-valued simulation

    def not_vd2(self, inputs, outputs):
        logic.bp_not(outputs[0], inputs[0])

    def and_vd2(self, inputs, outputs):
        logic.bp_and(outputs[0], *inputs)

    def or_vd2(self, inputs, outputs):
        logic.bp_or(outputs[0], *inputs)

    def xor_vd2(self, inputs, outputs):
        logic.bp_xor(outputs[0], *inputs)

    def sdff_vd2(self, inputs, outputs):
        self.dff_vd2(inputs, outputs)
        if len(outputs) > 1:
            logic.bp_not(outputs[1], inputs[0])

    def dff_vd2(self, inputs, outputs):
        logic.bp_buf(outputs[0], inputs[0])

    def nand_vd2(self, inputs, outputs):
        self.and_vd2(inputs, outputs)
        self.not_vd2(outputs, outputs)

    def nor_vd2(self, inputs, outputs):
        self.or_vd2(inputs, outputs)
        self.not_vd2(outputs, outputs)

    def xnor_vd2(self, inputs, outputs):
        self.xor_vd2(inputs, outputs)
        self.not_vd2(outputs, outputs)
    
    def const1_vd2(self, _, outputs):
        for o in outputs: o[...] = self.zero
        self.not_vd2(outputs, outputs)

    # 8-valued simulation

    def not_vd3(self, inputs, outputs):
        logic.bp_not(outputs[0], inputs[0])

    def and_vd3(self, inputs, outputs):
        logic.bp_and(outputs[0], *inputs)

    def or_vd3(self, inputs, outputs):
        logic.bp_or(outputs[0], *inputs)

    def xor_vd3(self, inputs, outputs):
        logic.bp_xor(outputs[0], *inputs)

    def sdff_vd3(self, inputs, outputs):
        self.dff_vd3(inputs, outputs)
        if len(outputs) > 1:
            logic.bp_not(outputs[1], inputs[0])

    def dff_vd3(self, inputs, outputs):
        logic.bp_buf(outputs[0], inputs[0])

    def nand_vd3(self, inputs, outputs):
        self.and_vd3(inputs, outputs)
        self.not_vd3(outputs, outputs)

    def nor_vd3(self, inputs, outputs):
        self.or_vd3(inputs, outputs)
        self.not_vd3(outputs, outputs)

    def xnor_vd3(self, inputs, outputs):
        self.xor_vd3(inputs, outputs)
        self.not_vd3(outputs, outputs)
        
    def const1_vd3(self, _, outputs):
        for o in outputs: o[...] = self.zero
        self.not_vd3(outputs, outputs)
