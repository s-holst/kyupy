import numpy as np
from . import packed_vectors


class LogicSim:
    """A bit-parallel naive combinational logic simulator supporting 1, 4, or 8-valued logics.
    """
    def __init__(self, circuit, nvectors=1, vdim=1):
        self.circuit = circuit
        self.nvectors = nvectors
        nbytes = (nvectors - 1) // 8 + 1
        self.interface = list(circuit.interface) + [n for n in circuit.nodes if 'dff' in n.kind.lower()]
        self.state = np.zeros((len(circuit.lines), vdim, nbytes), dtype='uint8')
        self.state_epoch = np.zeros(len(circuit.nodes), dtype='int8') - 1
        self.tmp = np.zeros((5, vdim, nbytes), dtype='uint8')
        self.zero = np.zeros((vdim, nbytes), dtype='uint8')
        if vdim > 1:
            self.zero[1] = 255
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
        
        known_fct = [(f[:-4], getattr(self, f)) for f in dir(self) if f.endswith(f'_vd{vdim}')]
        self.node_fct = []
        for n in circuit.nodes:
            t = n.kind.lower().replace('__fork__', 'fork')
            t = t.replace('__const0__', 'const0')
            t = t.replace('__const1__', 'const1')
            t = t.replace('tieh', 'const1')
            # t = t.replace('xor', 'or').replace('xnor', 'nor')
            fcts = [f for n, f in known_fct if t.startswith(n)]
            if len(fcts) < 1:
                raise ValueError(f'Unknown node kind {n.kind}')
            self.node_fct.append(fcts[0])

    def assign(self, stimuli):
        if isinstance(stimuli, packed_vectors.PackedVectors):
            stimuli = stimuli.bits
        for (stim, node) in zip(stimuli, self.interface):
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
        if isinstance(responses, packed_vectors.PackedVectors):
            responses = responses.bits
        for (resp, node) in zip(responses, self.interface):
            if len(node.ins) == 0: continue
            resp[...] = self.state[node.ins[0].index]

    def propagate(self):
        for node in self.circuit.topological_order():
            if self.state_epoch[node.index] != self.epoch: continue
            inputs = [self.state[line.index] if line else self.zero for line in node.ins]
            outputs = [self.state[line.index] if line else self.tmp[3] for line in node.outs]
            # print('sim', node)
            self.node_fct[node.index](inputs, outputs)
            for line in node.outs:
                self.state_epoch[line.reader.index] = self.epoch
        self.epoch = (self.epoch + 1) % 128

    @staticmethod
    def fork_vdx(inputs, outputs):
        for o in outputs: o[...] = inputs[0]
    
    def const0_vdx(self, _, outputs):
        for o in outputs: o[...] = self.zero

    # 2-valued simulation

    @staticmethod
    def not_vd1(inputs, outputs):
        outputs[0][0] = ~inputs[0][0]

    def const1_vd1(self, _, outputs):
        for o in outputs: o[...] = self.zero
        self.not_vd1(outputs, outputs)

    @staticmethod
    def and_vd1(inputs, outputs):
        o = outputs[0]
        o[0] = inputs[0][0]
        for i in inputs[1:]: o[0] &= i[0]

    @staticmethod
    def or_vd1(inputs, outputs):
        o = outputs[0]
        o[0] = inputs[0][0]
        for i in inputs[1:]: o[0] |= i[0]

    @staticmethod
    def xor_vd1(inputs, outputs):
        o = outputs[0]
        o[0] = inputs[0][0]
        for i in inputs[1:]: o[0] ^= i[0]

    @staticmethod
    def sdff_vd1(inputs, outputs):
        outputs[0][0] = inputs[0][0]
        if len(outputs) > 1:
            outputs[1][0] = ~inputs[0][0]

    @staticmethod
    def dff_vd1(inputs, outputs):
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
    # sym [0] [1] (value, care)
    #  0   0   1
    #  1   1   1
    #  -   0   0
    #  X   1   0

    @staticmethod
    def not_vd2(inputs, outputs):
        # 4-valued not:
        # i: 0 1 - X
        # o: 1 0 X X
        # o0 1 0 1 1
        # o1 1 1 0 0

        outputs[0][0] = ~inputs[0][0] | ~inputs[0][1]  # value = 0 or DC
        outputs[0][1] = inputs[0][1]  # care = C

    def and_vd2(self, inputs, outputs):
        # 4-valued:    o[0]:     o[1]:
        #    0 1 - X   0 1 - X   0 1 - X
        # 0  0 0 0 0   0 0 0 0   1 1 1 1
        # 1  0 1 X X   0 1 1 1   1 1 0 0
        # -  0 X X X   0 1 1 1   1 0 0 0
        # X  0 X X X   0 1 1 1   1 0 0 0

        i = inputs[0]
        any0 = self.tmp[0]
        anyd = self.tmp[1]
        any0[0] = ~i[0] & i[1]
        anyd[0] = ~i[1]
        for i in inputs[1:]:
            any0[0] |= ~i[0] & i[1]
            anyd[0] |= ~i[1]
        o = outputs[0]
        o[0] = ~any0[0]  # value = no0
        o[1] = any0[0] | ~anyd[0]  # care = any0 or noDC

    def or_vd2(self, inputs, outputs):
        # 4-valued:    o[0]:     o[1]:
        #    0 1 - X   0 1 - X   0 1 - X
        # 0  0 1 X X   0 1 1 1   1 1 0 0
        # 1  1 1 1 1   1 1 1 1   1 1 1 1
        # -  X 1 X X   1 1 1 1   0 1 0 0
        # X  X 1 X X   1 1 1 1   0 1 0 0

        i = inputs[0]
        any1 = self.tmp[0]
        anyd = self.tmp[1]
        any1[0] = i[0] & i[1]
        anyd[0] = ~i[1]
        for i in inputs[1:]:
            any1[0] |= i[0] & i[1]
            anyd[0] |= ~i[1]
        o = outputs[0]
        o[0] = any1[0] | anyd[0]  # value = any1 or anyDC
        o[1] = any1[0] | ~anyd[0]  # care = any1 or noDC

    def xor_vd2(self, inputs, outputs):
        # 4-valued:    o[0]:     o[1]:
        #    0 1 - X   0 1 - X   0 1 - X
        # 0  0 1 X X   0 1 1 1   1 1 0 0
        # 1  1 0 X X   1 0 1 1   1 1 0 0
        # -  X X X X   1 1 1 1   0 0 0 0
        # X  X X X X   1 1 1 1   0 0 0 0

        i = inputs[0]
        odd1 = self.tmp[0]
        anyd = self.tmp[1]
        odd1[0] = i[0] & i[1]
        anyd[0] = ~i[1]
        for i in inputs[1:]:
            odd1[0] ^= i[0] & i[1]
            anyd[0] |= ~i[1]
        o = outputs[0]
        o[0] = odd1[0] | anyd[0]  # value = odd1 or anyDC
        o[1] = ~anyd[0]  # care = noDC

    def sdff_vd2(self, inputs, outputs):
        self.dff_vd2(inputs, outputs)
        if len(outputs) > 1:
            outputs[1][0] = ~inputs[0][0] | ~inputs[0][1]  # value = 0 or DC
            outputs[1][1] = inputs[0][1]  # care = C

    @staticmethod
    def dff_vd2(inputs, outputs):
        outputs[0][0] = inputs[0][0] | ~inputs[0][1]  # value = 1 or DC
        outputs[0][1] = inputs[0][1]  # care = C

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
    # sym [0] [1] [2] (initial value, ~final value, toggles present?)
    #  0   0   1   0
    #  1   1   0   0
    #  -   0   0   0
    #  X   1   1   0
    #  R   0   0   1  _/"
    #  F   1   1   1  "\_
    #  P   0   1   1  _/\_
    #  N   1   0   1  "\/"

    def not_vd3(self, inputs, outputs):
        # 8-valued not:
        # i: 0 1 - X R F P N
        # i0 0 1 0 1 0 1 0 1
        # i1 1 0 0 1 0 1 1 0
        # i2 0 0 0 0 1 1 1 1
        # o: 1 0 X X F R N P
        # o0 1 0 1 1 1 0 1 0
        # o1 0 1 1 1 1 0 0 1
        # o2 0 0 0 0 1 1 1 1
        i = inputs[0]
        dc = self.tmp[0]
        dc[0] = ~(i[0] ^ i[1]) & ~i[2]
        dc = self.tmp[0]
        outputs[0][0] = ~i[0] | dc[0]  # init.v = ~i0 or DC
        outputs[0][1] = ~i[1] | dc[0]  # init.v = ~i1 or DC
        outputs[0][2] = i[2]  # toggles = i2

    def and_vd3(self, inputs, outputs):
        # 8-valued:           o[0]:            o[1]:            o[2]:
        #    0 1 - X R F P N  0 1 - X R F P N  0 1 - X R F P N  0 1 - X R F P N
        # 0  0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0  1 1 1 1 1 1 1 1  0 0 0 0 0 0 0 0
        # 1  0 1 X X R F P N  0 1 1 1 0 1 0 1  1 0 1 1 0 1 1 0  0 0 0 0 1 1 1 1
        # -  0 X X X X X X X  0 1 1 1 1 1 1 1  1 1 1 1 1 1 1 1  0 0 0 0 0 0 0 0
        # X  0 X X X X X X X  0 1 1 1 1 1 1 1  1 1 1 1 1 1 1 1  0 0 0 0 0 0 0 0
        # R  0 R X X R R P R  0 0 1 1 0 0 0 0  1 0 1 1 0 0 1 0  0 1 0 0 1 1 1 1
        # F  0 F X X R F P F  0 1 1 1 0 1 0 1  1 1 1 1 0 1 1 1  0 1 0 0 1 1 1 1
        # P  0 P X X P P P P  0 0 1 1 0 0 0 0  1 1 1 1 1 1 1 1  0 1 0 0 1 1 1 1
        # N  0 N X X R F P N  0 1 1 1 0 1 0 1  1 0 1 1 0 1 1 0  0 1 0 0 1 1 1 1
        i = inputs[0]
        anyi0 = self.tmp[0]
        anyf0 = self.tmp[1]
        anyd = self.tmp[2]
        any0 = self.tmp[3]
        any_t = self.tmp[4]
        anyd[0] = ~(i[0] ^ i[1]) & ~i[2]
        anyi0[0] = ~i[0] & ~anyd[0]
        anyf0[0] = i[1] & ~anyd[0]
        any_t[0] = i[2]
        any0[0] = anyi0[0] & anyf0[0] & ~i[2]
        for i in inputs[1:]:
            dc = ~(i[0] ^ i[1]) & ~i[2]
            anyd[0] |= dc
            anyi0[0] |= ~i[0] & ~dc
            anyf0[0] |= i[1] & ~dc
            any_t[0] |= i[2]
            any0[0] |= ~i[0] & ~dc & i[1] & ~i[2]
        o = outputs[0]
        o[0] = (~anyi0[0] | anyd[0]) & ~any0[0]  # initial = no_i0 or DC
        o[1] = anyf0[0] | anyd[0]  # ~final = ~no_f0 or DC
        o[2] = any_t[0] & ~(anyd[0] | any0[0])  # toggle = anyT and noDC and no0

    def or_vd3(self, inputs, outputs):
        # 8-valued:           o[0]:            o[1]:            o[2]:
        #    0 1 - X R F P N  0 1 - X R F P N  0 1 - X R F P N  0 1 - X R F P N
        # 0  0 1 X X R F P N  0 1 1 1 0 1 0 1  1 0 1 1 0 1 1 0  0 0 0 0 1 1 1 1
        # 1  1 1 1 1 1 1 1 1  1 1 1 1 1 1 1 1  0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0
        # -  X 1 X X X X X X  1 1 1 1 1 1 1 1  1 0 1 1 1 1 1 1  0 0 0 0 0 0 0 0
        # X  X 1 X X X X X X  1 1 1 1 1 1 1 1  1 0 1 1 1 1 1 1  0 0 0 0 0 0 0 0
        # R  R 1 X X R N R R  0 1 1 1 0 1 0 0  0 0 1 1 0 0 0 0  1 0 0 0 1 1 1 1
        # F  F 1 X X N F F F  1 1 1 1 1 1 1 1  1 0 1 1 0 1 1 1  1 0 0 0 1 1 1 1
        # P  P 1 X X R F P N  0 1 1 1 0 1 0 1  1 0 1 1 0 1 1 0  1 0 0 0 1 1 1 1
        # N  N 1 X X R F N N  1 1 1 1 0 1 1 1  0 0 1 1 0 1 0 0  1 0 0 0 1 1 1 1
        i = inputs[0]
        anyi1 = self.tmp[0]
        anyf1 = self.tmp[1]
        anyd = self.tmp[2]
        any1 = self.tmp[3]
        any_t = self.tmp[4]
        anyd[0] = ~(i[0] ^ i[1]) & ~i[2]
        anyi1[0] = i[0] & ~anyd[0]
        anyf1[0] = ~i[1] & ~anyd[0]
        any_t[0] = i[2]
        any1[0] = (anyi1[0] & anyf1[0]) & ~i[2]
        for i in inputs[1:]:
            dc = ~(i[0] ^ i[1]) & ~i[2]
            anyd[0] |= dc
            anyi1[0] |= i[0] & ~dc
            anyf1[0] |= ~i[1] & ~dc
            any_t[0] |= i[2]
            any1[0] |= i[0] & ~dc & ~i[1] & ~i[2]
        o = outputs[0]
        o[0] = anyi1[0] | anyd[0]  # initial = i1 or DC
        o[1] = (~anyf1[0] | anyd[0]) & ~any1[0]  # ~final = f1 or DC
        o[2] = any_t[0] & ~(anyd[0] | any1[0])  # toggle = anyT and no(DC or 1)

    def xor_vd3(self, inputs, outputs):
        # 8-valued:           o[0]:            o[1]:            o[2]:
        #    0 1 - X R F P N  0 1 - X R F P N  0 1 - X R F P N  0 1 - X R F P N
        # 0  0 1 X X R F P N  0 1 1 1 0 1 0 1  1 0 1 1 0 1 1 0  0 0 0 0 1 1 1 1
        # 1  1 0 X X F R N P  1 0 1 1 1 0 1 0  0 1 1 1 1 0 0 1  0 0 0 0 1 1 1 1
        # -  X X X X X X X X  1 1 1 1 1 1 1 1  1 1 1 1 1 1 1 1  0 0 0 0 0 0 0 0
        # X  X X X X X X X X  1 1 1 1 1 1 1 1  1 1 1 1 1 1 1 1  0 0 0 0 0 0 0 0
        # R  R F X X P N R F  0 1 1 1 0 1 0 1  0 1 1 1 1 0 0 1  1 1 0 0 1 1 1 1
        # F  F R X X N P F R  1 0 1 1 1 0 1 0  1 0 1 1 0 1 1 0  1 1 0 0 1 1 1 1
        # P  P N X X R F P N  0 1 1 1 0 1 0 1  1 0 1 1 0 1 1 0  1 1 0 0 1 1 1 1
        # N  N P X X F R N P  1 0 1 1 1 0 1 0  0 1 1 1 1 0 0 1  1 1 0 0 1 1 1 1
        i = inputs[0]
        odd0 = self.tmp[0]
        odd1 = self.tmp[1]
        anyd = self.tmp[2]
        anyt = self.tmp[3]
        odd0[0] = i[0]
        odd1[0] = i[1]
        anyd[0] = ~(i[0] ^ i[1]) & ~i[2]
        anyt[0] = i[2]
        for i in inputs[1:]:
            odd0[0] ^= i[0]
            odd1[0] ^= i[1]
            anyd[0] |= ~(i[0] ^ i[1]) & ~i[2]
            anyt[0] |= i[2]
        o = outputs[0]
        o[0] = odd0[0] | anyd[0]
        o[1] = ~odd1[0] | anyd[0]
        o[2] = anyt[0] & ~anyd[0]
        
    def sdff_vd3(self, inputs, outputs):
        self.dff_vd3(inputs, outputs)
        if len(outputs) > 1:
            i = inputs[0]
            dc = self.tmp[0]
            dc[0] = ~(i[0] ^ i[1]) & ~i[2]
            outputs[1][0] = ~i[0] | dc[0]  # value = 1 or DC
            outputs[1][1] = ~i[1] | dc[0]  # value = 1 or DC
            outputs[1][2] = i[2]  # toggle = T

    def dff_vd3(self, inputs, outputs):
        i = inputs[0]
        dc = self.tmp[0]
        dc[0] = ~(i[0] ^ i[1]) & ~i[2]
        outputs[0][0] = i[0] | dc[0]  # value = 1 or DC
        outputs[0][1] = i[1] | dc[0]  # value = 1 or DC
        outputs[0][2] = i[2]  # toggle = T

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
