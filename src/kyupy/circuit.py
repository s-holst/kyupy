"""Data structures for representing non-hierarchical gate-level circuits.

The class :class:`Circuit` is a container of nodes connected by lines.
A node is an instance of class :class:`Node`,
and a line is an instance of class :class:`Line`.
"""

from collections import deque


class GrowingList(list):
    def __setitem__(self, index, value):
        if index >= len(self):
            self.extend([None] * (index + 1 - len(self)))
        super().__setitem__(index, value)

    def free_index(self):
        return next((i for i, x in enumerate(self) if x is None), len(self))


class IndexList(list):
    def __delitem__(self, index):
        if index == len(self) - 1:
            super().__delitem__(index)
        else:
            replacement = self.pop()
            replacement.index = index
            super().__setitem__(index, replacement)


class Node:
    """A node is a named entity in a circuit (e.g. a gate, a standard cell,
    a named signal, or a fan-out point) that is connected to other nodes via lines.

    The constructor automatically adds the new node to the given circuit.
    """
    def __init__(self, circuit, name, kind='__fork__'):
        if kind == '__fork__':
            assert name not in circuit.forks, f'fork of name {name} already in circuit.'
            circuit.forks[name] = self
        else:
            assert name not in circuit.cells, f'cell of name {name} already in circuit.'
            circuit.cells[name] = self
        circuit.nodes.append(self)
        self.circuit = circuit
        """The :class:`Circuit` object the node is part of.
        """
        self.name = name
        """The name of the node.

        Names must be unique among all forks and all cells in the circuit.
        However, a fork (:py:attr:`kind` is set to '__fork__') and a cell with the same name may coexist.
        """
        self.kind = kind
        """A string describing the type of the node.

        Common types are the names from a standard cell library or general gate names like 'AND' or 'NOR'.
        If :py:attr:`kind` is set to '__fork__', it receives special treatment.
        A `fork` describes a named signal or a fan-out point in the circuit and not a physical `cell` like a gate.
        In the circuit, the namespaces of forks and cells are kept separate.
        While :py:attr:`name` must be unique among all forks and all cells, a fork can have the same name as a cell.
        The :py:attr:`index`, however, is unique among all nodes; a fork cannot have the same index as a cell.
        """
        self.index = len(circuit.nodes) - 1
        """A unique and consecutive integer index of the node within the circuit.

        It can be used to store additional data about the node :code:`n`
        by allocating an array or list :code:`my_data` of length :code:`len(n.circuit.nodes)` and
        accessing it by :code:`my_data[n.index]`.
        """
        self.ins = GrowingList()
        """A list of input connections (:class:`Line` objects).
        """
        self.outs = GrowingList()
        """A list of output connections (:class:`Line` objects).
        """

    def __index__(self):
        return self.index

    def __repr__(self):
        ins = ' '.join([f'<{line.index}' if line is not None else '<None' for line in self.ins])
        outs = ' '.join([f'>{line.index}' if line is not None else '>None' for line in self.outs])
        return f'{self.index}:{self.kind}"{self.name}" {ins} {outs}'

    def remove(self):
        """Removes the node from its circuit.

        Lines may still reference the removed node.
        The user must connect such lines to other nodes or remove the lines from the circuit.
        To keep the indices consecutive, the node with the highest index within the circuit
        will be assigned the index of the removed node.
        """
        if self.circuit is not None:
            del self.circuit.nodes[self.index]
            if self.kind == '__fork__':
                del self.circuit.forks[self.name]
            else:
                del self.circuit.cells[self.name]
            self.circuit = None


class Line:
    """A line is a directional 1:1 connection between two nodes.

    It always connects an output of one `driver` node to an input of one `reader` node.
    If a signal fans out to multiple readers, a '__fork__' node needs to be added.

    The constructor automatically adds the new line to the given circuit and inserts references into the connection
    lists of connected nodes.

    When adding a line, input and output pins can either be specified explicitly
    :code:`Line(circuit, (driver, 2), (reader, 0))`, or implicitly :code:`Line(circuit, driver, reader)`.
    In the implicit case, the line will be connected to the first free pin of the node.
    Use the explicit case only if connections to specific pins are required.
    It may overwrite any previous line references in the connection list of the nodes.
    """
    def __init__(self, circuit, driver, reader):
        self.circuit = circuit
        """The :class:`Circuit` object the line is part of.
        """
        self.circuit.lines.append(self)
        self.index = len(self.circuit.lines) - 1
        """A unique and consecutive integer index of the line within the circuit.

        It can be used to store additional data about the line :code:`l`
        by allocating an array or list :code:`my_data` of length :code:`len(l.circuit.lines)` and
        accessing it by :code:`my_data[l.index]`.
        """
        if not isinstance(driver, tuple): driver = (driver, driver.outs.free_index())
        self.driver = driver[0]
        """The :class:`Node` object that drives this line.
        """
        self.driver_pin = driver[1]
        """The output pin position of the driver node this line is connected to.

        This is the position in the outs-list of the driving node this line referenced from:
        :code:`self.driver.outs[self.driver_pin] == self`.
        """
        if not isinstance(reader, tuple): reader = (reader, reader.ins.free_index())
        self.reader = reader[0]
        """The :class:`Node` object that reads this line.
        """
        self.reader_pin = reader[1]
        """The input pin position of the reader node this line is connected to.

        This is the position in the ins-list of the reader node this line referenced from:
        :code:`self.reader.ins[self.reader_pin] == self`.
        """
        self.driver.outs[self.driver_pin] = self
        self.reader.ins[self.reader_pin] = self

    def remove(self):
        """Removes the line from its circuit and its referencing nodes.

        To keep the indices consecutive, the line with the highest index within the circuit
        will be assigned the index of the removed line.
        """
        if self.driver is not None: self.driver.outs[self.driver_pin] = None
        if self.reader is not None: self.reader.ins[self.reader_pin] = None
        if self.circuit is not None: del self.circuit.lines[self.index]
        self.driver = None
        self.reader = None
        self.circuit = None

    def __index__(self):
        return self.index

    def __repr__(self):
        return f'{self.index}'

    def __lt__(self, other):
        return self.index < other.index


class Circuit:
    """A Circuit is a container for interconnected nodes and lines.

    It provides access to lines by index and to nodes by index and by name.
    Nodes come in two flavors: `cells` and `forks` (see :py:attr:`Node.kind`).
    The name spaces of cells and forks are kept separate.

    The indices of nodes and lines are kept consecutive and unique.
    Whenever lines or nodes are removed from the circuit, the indices of some other lines or nodes may change
    to enforce consecutiveness.

    A subset of nodes can be designated as primary input- or output-ports of the circuit.
    This is done by adding them to the :py:attr:`interface` list.
    """
    def __init__(self, name=None):
        self.name = name
        """The name of the circuit.
        """
        self.nodes = IndexList()
        """A list of all :class:`Node` objects contained in the circuit.

        The position of a node in this list equals its index :code:`self.nodes[42].index == 42`.
        """
        self.lines = IndexList()
        """A list of all :class:`Line` objects contained in the circuit.

        The position of a line in this list equals its index :code:`self.lines[42].index == 42`.
        """
        self.interface = GrowingList()
        """A list of nodes that are designated as primary input- or output-ports.

        Port-nodes are contained in :py:attr:`nodes` as well as :py:attr:`interface`.
        The position of a node in the interface list corresponds to positions of logic values in test vectors.
        The port direction is not stored explicitly.
        Usually, nodes in the interface list without any lines in their :py:attr:`Node.ins` list are primary inputs,
        and nodes without any lines in their :py:attr:`Node.outs` list are regarded as primary outputs.
        """
        self.cells = {}
        """A dictionary to access cells by name.
        """
        self.forks = {}
        """A dictionary to access forks by name.
        """

    def get_or_add_fork(self, name):
        return self.forks[name] if name in self.forks else Node(self, name)

    def copy(self):
        """Returns a deep copy of the circuit.
        """
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
        """Returns a string representation of the circuit and all its nodes.
        """
        header = f'{self.name}({",".join([str(n.index) for n in self.interface])})\n'
        return header + '\n'.join([str(n) for n in self.nodes])

    def __repr__(self):
        name = f' {self.name}' if self.name else ''
        return f'<Circuit{name} cells={len(self.cells)} forks={len(self.forks)} ' + \
               f'lines={len(self.lines)} ports={len(self.interface)}>'

    def topological_order(self):
        """Generator function to iterate over all nodes in topological order.

        Nodes without input lines and nodes whose :py:attr:`Node.kind` contains the substring 'DFF' are
        yielded first.
        """
        visit_count = [0] * len(self.nodes)
        queue = deque(n for n in self.nodes if len(n.ins) == 0 or 'DFF' in n.kind)
        while len(queue) > 0:
            n = queue.popleft()
            for line in n.outs:
                if line is None: continue
                succ = line.reader
                visit_count[succ] += 1
                if visit_count[succ] == len(succ.ins) and 'DFF' not in succ.kind:
                    queue.append(succ)
            yield n

    def topological_line_order(self):
        """Generator function to iterate over all lines in topological order.
        """
        for n in self.topological_order():
            for line in n.outs:
                if line is not None:
                    yield line

    def reversed_topological_order(self):
        """Generator function to iterate over all nodes in reversed topological order.

        Nodes without output lines and nodes whose :py:attr:`Node.kind` contains the substring 'DFF' are
        yielded first.
        """
        visit_count = [0] * len(self.nodes)
        queue = deque(n for n in self.nodes if len(n.outs) == 0 or 'DFF' in n.kind)
        while len(queue) > 0:
            n = queue.popleft()
            for line in n.ins:
                pred = line.driver
                visit_count[pred] += 1
                if visit_count[pred] == len(pred.outs) and 'DFF' not in pred.kind:
                    queue.append(pred)
            yield n

    def fanin(self, origin_nodes):
        """Generator function to iterate over the fan-in cone of a given list of origin nodes.

        Nodes are yielded in reversed topological order.
        """
        marks = [False] * len(self.nodes)
        for n in origin_nodes:
            marks[n] = True
        for n in self.reversed_topological_order():
            if not marks[n]:
                for line in n.outs:
                    if line is not None:
                        marks[n] |= marks[line.reader]
            if marks[n]:
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
