from collections import deque


class GrowingList(list):
    def __setitem__(self, index, value):
        if index >= len(self):
            self.extend([None] * (index + 1 - len(self)))
        super().__setitem__(index, value)


class IndexList(list):
    def __delitem__(self, index):
        if index == len(self) - 1:
            super().__delitem__(index)
        else:
            replacement = self.pop()
            replacement.index = index
            super().__setitem__(index, replacement)


class Node:
    """A Node is a named entity in a circuit (e.g. a gate, a standard cell,
    a named signal, or a fan-out point) that has connections to other nodes.
    Each node contains:

    * `self.index`: a circuit-unique integer index.
    * `self.kind`: a type describing its function (e.g. 'AND', 'NOR').
      The type '__fork__' is special. It signifies a named signal
      or a fan-out in the circuit. Any other type is considered a physical cell.
    * `self.name`: a name. Names must be unique among all forks and all cells
      in the circuit. However, a fork (`self.kind=='__fork__'`) and a cell with
      the same name may coexist.
    * `self.ins`: a list of input connections (objects of class `Line`)
    * `self.outs`: a list of output connections (objects of class `Line`).
    """
    def __init__(self, circuit, name, kind='__fork__'):
        if kind == '__fork__':
            if name in circuit.forks:
                raise ValueError(f'fork of name {name} already exists.')
            circuit.forks[name] = self
        else:
            if name in circuit.cells:
                raise ValueError(f'cell of name {name} already exists.')
            circuit.cells[name] = self
        self.index = len(circuit.nodes)
        circuit.nodes.append(self)
        self.circuit = circuit
        self.name = name
        self.kind = kind
        self.ins = GrowingList()
        self.outs = GrowingList()

    def __repr__(self):
        ins = ' '.join([f'<{line.index}' if line is not None else '<None' for line in self.ins])
        outs = ' '.join([f'>{line.index}' if line is not None else '>None' for line in self.outs])
        return f'{self.index}:{self.kind}"{self.name}" {ins} {outs}'

    def remove(self):
        if self.circuit is not None:
            del self.circuit.nodes[self.index]
            if self.kind == '__fork__':
                del self.circuit.forks[self.name]
            else:
                del self.circuit.cells[self.name]
            self.circuit = None


class Line:
    """A Line is a directional 1:1 connection between two Nodes. It always
    connects an output of a node (called `driver`) to an input of a node
    (called `reader`) and has a circuit-unique index (`self.index`).

    Furthermore, `self.driver_pin` and `self.reader_pin` are the
    integer indices of the connected pins of the nodes. They always correspond
    to the positions of the line in the connection lists of the nodes:

    * `self.driver.outs[self.driver_pin] == self`
    * `self.reader.ins[self.reader_pin] == self`

    A Line always connects a single driver to a single reader. If a signal fans out to
    multiple readers, a '__fork__' Node needs to be added.
    """
    def __init__(self, circuit, driver, reader):
        self.index = len(circuit.lines)
        circuit.lines.append(self)
        if type(driver) is Node:
            self.driver = driver
            self.driver_pin = len(driver.outs)
            for pin, line in enumerate(driver.outs):
                if line is None:
                    self.driver_pin = pin
                    break
        else:
            self.driver, self.driver_pin = driver
        if type(reader) is Node:
            self.reader = reader
            self.reader_pin = len(reader.ins)
            for pin, line in enumerate(reader.ins):
                if line is None:
                    self.reader_pin = pin
                    break
        else:
            self.reader, self.reader_pin = reader
        self.driver.outs[self.driver_pin] = self
        self.reader.ins[self.reader_pin] = self

    def remove(self):
        circuit = None
        if self.driver is not None:
            self.driver.outs[self.driver_pin] = None
            circuit = self.driver.circuit
        if self.reader is not None:
            self.reader.ins[self.reader_pin] = None
            circuit = self.reader.circuit
        if circuit is not None:
            del circuit.lines[self.index]
        self.driver = None
        self.reader = None

    def __repr__(self):
        return f'{self.index}'

    def __lt__(self, other):
        return self.index < other.index


class Circuit:
    """A Circuit is a container for interconnected nodes and lines.

    All contained lines have unique indices, so have all contained nodes.
    These indices can be used to store additional data about nodes or lines
    by allocating an array `my_data` of length `len(self.nodes)` and then
    accessing it by `my_data[n.index]`. The indices may change iff lines or
    nodes are removed from the circuit.

    Nodes come in two flavors (cells and forks, see `Node`). The names of
    these nodes are kept unique within these two flavors.
    """
    def __init__(self, name=None):
        self.name = name
        self.nodes = IndexList()
        self.lines = IndexList()
        self.interface = GrowingList()
        self.cells = {}
        self.forks = {}

    def get_or_add_fork(self, name):
        return self.forks[name] if name in self.forks else Node(self, name)
    
    def copy(self):
        c = Circuit(self.name)
        for node in self.nodes:
            Node(c, node.name, node.kind)
        for line in self.lines:
            d = c.forks[line.driver.name] if line.driver.kind == '__fork__' else c.cells[line.driver.name]
            r = c.forks[line.reader.name] if line.reader.kind == '__fork__' else c.cells[line.reader.name]
            Line(c, (d, line.driver_pin), (r, line.reader_pin))
        for node in self.interface:
            if node.kind == '__fork__':
                n = c.forks[node.name]
            else:
                n = c.cells[node.name]
            c.interface.append(n)
        return c
    
    def dump(self):
        header = f'{self.name}({",".join([str(n.index) for n in self.interface])})\n'
        return header + '\n'.join([str(n) for n in self.nodes])

    def __repr__(self):
        name = f" '{self.name}'" if self.name else ''
        return f'<Circuit{name} with {len(self.nodes)} nodes, {len(self.lines)} lines, {len(self.interface)} ports>'

    def topological_order(self):
        visit_count = [0] * len(self.nodes)
        queue = deque(n for n in self.nodes if len(n.ins) == 0 or 'DFF' in n.kind)
        while len(queue) > 0:
            n = queue.popleft()
            for line in n.outs:
                if line is None: continue
                succ = line.reader
                visit_count[succ.index] += 1
                if visit_count[succ.index] == len(succ.ins) and 'DFF' not in succ.kind:
                    queue.append(succ)
            yield n

    def topological_line_order(self):
        for n in self.topological_order():
            for line in n.outs:
                if line is not None:
                    yield line

    def reversed_topological_order(self):
        visit_count = [0] * len(self.nodes)
        queue = deque(n for n in self.nodes if len(n.outs) == 0 or 'DFF' in n.kind)
        while len(queue) > 0:
            n = queue.popleft()
            for line in n.ins:
                pred = line.driver
                visit_count[pred.index] += 1
                if visit_count[pred.index] == len(pred.outs) and 'DFF' not in pred.kind:
                    queue.append(pred)
            yield n

    def fanin(self, origin_nodes):
        marks = [False] * len(self.nodes)
        for n in origin_nodes:
            marks[n.index] = True
        for n in self.reversed_topological_order():
            if not marks[n.index]:
                for line in n.outs:
                    if line is not None:
                        marks[n.index] |= marks[line.reader.index]
            if marks[n.index]:
                yield n

    def fanout_free_regions(self):
        for stem in self.reversed_topological_order():
            if len(stem.outs) == 1 and 'DFF' not in stem.kind: continue
            region = []
            if 'DFF' in stem.kind:
                n = stem.ins[0]
                if len(n.driver.outs) == 1 and 'DFF' not in n.driver.kind:
                    queue = deque([n.driver])
                else:
                    queue = deque()
            else:
                queue = deque(n.driver for n in stem.ins
                              if len(n.driver.outs) == 1 and 'DFF' not in n.driver.kind)
            while len(queue) > 0:
                n = queue.popleft()
                preds = [pred.driver for pred in n.ins
                         if len(pred.driver.outs) == 1 and 'DFF' not in pred.driver.kind]
                queue.extend(preds)
                region.append(n)
            yield stem, region
