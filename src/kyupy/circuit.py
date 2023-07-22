"""Core module for handling non-hierarchical gate-level circuits.

The class :class:`Circuit` is a container of nodes connected by lines.
A node is an instance of class :class:`Node`,
and a line is an instance of class :class:`Line`.

The data structures are designed to work together nicely with numpy arrays.
For example, all the nodes and connections in the circuit graph have consecutive integer indices that can be used to access ndarrays with associated data.
Circuit graphs also define an ordering of inputs, outputs and other nodes to easily process test vector data and alike.

"""

from collections import deque, defaultdict
import re

import numpy as np


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

        It can be used to associate additional data to a node :code:`n`
        by allocating an array or list :code:`my_data` of length :code:`len(n.circuit.nodes)` and
        accessing it by :code:`my_data[n.index]` or simply by :code:`my_data[n]`.
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
        ins = ' ' + ins if len(ins) else ''
        outs = ' ' + outs if len(outs) else ''
        return f'{self.index}:{self.kind}"{self.name}"{ins}{outs}'

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

    def __eq__(self, other):
        """Checks equality of node name and kind. Does not check pin connections.

        This is ok, because (name, kind) is unique within a circuit.
        """
        return self.name == other.name and self.kind == other.kind

    def __hash__(self):
        return hash((self.name, self.kind))


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
        accessing it by :code:`my_data[l.index]` or simply by :code:`my_data[l]`.
        """
        if not isinstance(driver, tuple): driver = (driver, driver.outs.free_index())
        self.driver = driver[0]
        """The :class:`Node` object that drives this line.
        """
        self.driver_pin = driver[1]
        """The output pin position of the driver node this line is connected to.

        This is the position in the list :py:attr:`Node.outs` of the driving node this line referenced from:
        :code:`self.driver.outs[self.driver_pin] == self`.
        """
        if not isinstance(reader, tuple): reader = (reader, reader.ins.free_index())
        self.reader = reader[0]
        """The :class:`Node` object that reads this line.
        """
        self.reader_pin = reader[1]
        """The input pin position of the reader node this line is connected to.

        This is the position in the list :py:attr:`Node.ins` of the reader node this line referenced from:
        :code:`self.reader.ins[self.reader_pin] == self`.
        """
        self.driver.outs[self.driver_pin] = self
        self.reader.ins[self.reader_pin] = self

    def remove(self):
        """Removes the line from its circuit and its referencing nodes.

        To keep the indices consecutive, the line with the highest index within the circuit
        will be assigned the index of the removed line.
        """
        if self.driver is not None:
            self.driver.outs[self.driver_pin] = None
            if self.driver.kind == '__fork__':  # squeeze outputs
                del self.driver.outs[self.driver_pin]
                for i, l in enumerate(self.driver.outs): l.driver_pin = i
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

    def __eq__(self, other):
        return self.driver == other.driver and self.driver_pin == other.driver_pin and \
               self.reader == other.reader and self.reader_pin == other.reader_pin

    def __hash__(self):
        return hash((self.driver, self.driver_pin, self.reader, self.reader_pin))


class Circuit:
    """A Circuit is a container for interconnected nodes and lines.

    It provides access to lines by index and to nodes by index and by name.
    Nodes come in two flavors: `cells` and `forks` (see :py:attr:`Node.kind`).
    The name spaces of cells and forks are kept separate.

    The indices of nodes and lines are kept consecutive and unique.
    Whenever lines or nodes are removed from the circuit, the indices of some other lines or nodes may change
    to enforce consecutiveness.

    A subset of nodes can be designated as primary input- or output-ports of the circuit.
    This is done by adding them to the :py:attr:`io_nodes` list.
    """
    def __init__(self, name=None):
        self.name = name
        """The name of the circuit.
        """
        self.nodes : list[Node] = IndexList()
        """A list of all :class:`Node` objects contained in the circuit.

        The position of a node in this list equals its index :code:`self.nodes[42].index == 42`.
        This list must not be changed directly.
        Use the :class:`Node` constructor and :py:attr:`Node.remove()` to add and remove nodes.
        """
        self.lines : list[Line] = IndexList()
        """A list of all :class:`Line` objects contained in the circuit.

        The position of a line in this list equals its index :code:`self.lines[42].index == 42`.
        This list must not be changed directly.
        Use the :class:`Line` constructor and :py:attr:`Line.remove()` to add and remove lines.
        """
        self.io_nodes : list[Node] = GrowingList()
        """A list of nodes that are designated as primary input- or output-ports.

        Port-nodes are contained in :py:attr:`nodes` as well as :py:attr:`io_nodes`.
        The position of a node in the io_nodes list corresponds to positions of logic values in test vectors.
        The port direction is not stored explicitly.
        Usually, nodes in the io_nodes list without any lines in their :py:attr:`Node.ins` list are primary inputs,
        and all other nodes in the io_nodes list are regarded as primary outputs.
        """
        self.cells : dict[str, Node] = {}
        """A dictionary to access cells by name.

        This dictionary must not be changed directly.
        Use the :class:`Node` constructor and :py:attr:`Node.remove()` to add and remove nodes.
        """
        self.forks : dict[str, Node] = {}
        """A dictionary to access forks by name.

        This dictionary must not be changed directly.
        Use the :class:`Node` constructor and :py:attr:`Node.remove()` to add and remove nodes.
        """

    @property
    def s_nodes(self):
        """A list of all primary I/Os as well as all flip-flops and latches in the circuit (in that order).

        The s_nodes list defines the order of all ports and all sequential elements in the circuit.
        This list is constructed on-the-fly. If used in some inner toop, consider caching the list for better performance.
        """
        return list(self.io_nodes) + [n for n in self.nodes if 'dff' in n.kind.lower()] + [n for n in self.nodes if 'latch' in n.kind.lower()]

    def io_locs(self, prefix):
        """Returns the indices of primary I/Os that start with given name prefix.

        The returned values are used to index into the :py:attr:`io_nodes` array.
        If only one I/O cell matches the given prefix, a single integer is returned.
        If a bus matches the given prefix, a sorted list of indices is returned.
        Busses are identified by integers in the cell names following the given prefix.
        Lists for bus indices are sorted from LSB (e.g. :code:`data[0]`) to MSB (e.g. :code:`data[31]`).
        If a prefix matches multiple different signals or busses, alphanumerically sorted
        lists of lists are returned. Therefore, higher-dimensional busses
        (e.g. :code:`data0[0], data0[1], ...`, :code:`data1[0], data1[1], ...`) are supported as well.
        """
        return self._locs(prefix, list(self.io_nodes))

    def s_locs(self, prefix):
        """Returns the indices of I/Os and sequential elements that start with given name prefix.

        The returned values are used to index into the :py:attr:`s_nodes` list.
        It works the same as :py:attr:`io_locs`. See there for more details.
        """
        return self._locs(prefix, self.s_nodes)

    def _locs(self, prefix, nodes):
        d_top = dict()
        for i, n in enumerate(nodes):
            if m := re.match(fr'({prefix}.*?)((?:[\d_\[\]])*$)', n.name):
                path = [m[1]] + [int(v) for v in re.split(r'[_\[\]]+', m[2]) if len(v) > 0]
                d = d_top
                for j in path[:-1]:
                    d[j] = d.get(j, dict())
                    d = d[j]
                d[path[-1]] = i

        # sort recursively for multi-dimensional lists.
        def sorted_values(d): return [sorted_values(v) for k, v in sorted(d.items())] if isinstance(d, dict) else d
        l = sorted_values(d_top)
        while isinstance(l, list) and len(l) == 1: l = l[0]
        return None if isinstance(l, list) and len(l) == 0 else l

    @property
    def stats(self):
        """A dictionary with the counts of all different elements in the circuit.

        The dictionary contains the number of all different kinds of nodes, the number
        of lines, as well various sums like number of combinational gates, number of
        primary I/Os, number of sequential elements, and so on.

        The count of regular cells use their :py:attr:`Node.kind` as key, other statistics use
        dunder-keys like: `__comb__`, `__io__`, `__seq__`, and so on.
        """
        stats = defaultdict(int)
        stats['__node__'] = len(self.nodes)
        stats['__cell__'] = len(self.cells)
        stats['__fork__'] = len(self.forks)
        stats['__io__'] = len(self.io_nodes)
        stats['__line__'] = len(self.lines)
        for n in self.cells.values():
            stats[n.kind] += 1
            if 'dff' in n.kind.lower(): stats['__dff__'] += 1
            elif 'latch' in n.kind.lower(): stats['__latch__'] += 1
            elif 'put' not in n.kind.lower(): stats['__comb__'] += 1 # no input or output
        stats['__seq__'] = stats['__dff__'] + stats['__latch__']
        return dict(stats)

    def get_or_add_fork(self, name):
        return self.forks[name] if name in self.forks else Node(self, name)

    def remove_dangling_nodes(self, root_node:Node):
        if len([l for l in root_node.outs if l is not None]) > 0: return
        lines = [l for l in root_node.ins if l is not None]
        drivers = [l.driver for l in lines]
        root_node.remove()
        for l in lines:
            l.remove()
        for d in drivers:
            self.remove_dangling_nodes(d)

    def eliminate_1to1_forks(self):
        """Removes all forks that drive only one node.

        Such forks are inserted by parsers to annotate signal names. If this
        information is not needed, such forks can be removed and the two neighbors
        can be connected directly using one line. Forks that drive more than one node
        are not removed by this function.

        This function may remove some nodes and some lines from the circuit.
        Therefore that indices of other nodes and lines may change to keep the indices consecutive.
        It may therefore invalidate external data for nodes and lines.
        """
        ios = set(self.io_nodes)
        for n in list(self.forks.values()):
            if n in ios: continue
            if len(n.outs) != 1: continue
            in_line = n.ins[0]
            out_line = n.outs[0]
            out_reader = out_line.reader
            out_reader_pin = out_line.reader_pin
            n.remove()
            out_line.remove()
            in_line.reader = out_reader
            in_line.reader_pin = out_reader_pin
            in_line.reader.ins[in_line.reader_pin] = in_line

    def substitute(self, node, impl):
        """Replaces a given node with the given implementation circuit.

        The given node will be removed, the implementation is copied in and
        the signal lines are connected appropriately. The number and arrangement
        of the input and output ports must match the pins of the replaced node.

        This function tries to preserve node and line indices as much as possible.
        Usually, it only adds additional nodes and lines, preserving the order of
        all existing nodes and lines. If an implementation is empty, however, nodes
        and lines may get removed, changing indices and invalidating external data.
        """
        ios = set(impl.io_nodes)
        impl_in_nodes = [n for n in impl.io_nodes if len(n.ins) == 0]
        impl_out_lines = [n.ins[0] for n in impl.io_nodes if len(n.ins) > 0]
        designated_cell = None
        if len(impl_out_lines) > 0:
            n = impl_out_lines[0].driver
            while n.kind == '__fork__' and n not in ios:
                n = n.ins[0].driver
            designated_cell = n
        node_in_lines = list(node.ins) + [None] * (len(impl_in_nodes)-len(node.ins))
        node_out_lines = list(node.outs) + [None] * (len(impl_out_lines)-len(node.outs))
        assert len(node_in_lines) == len(impl_in_nodes)
        assert len(node_out_lines) == len(impl_out_lines)
        node_map = dict()
        if designated_cell is not None:
            node.kind = designated_cell.kind
            node_map[designated_cell] = node
            node.ins = GrowingList()
            node.outs = GrowingList()
        else:
            node.remove()
        ios = set(impl.io_nodes)
        for n in impl.nodes:  # add all nodes to main circuit
            if n not in ios:
                if n != designated_cell:
                    node_map[n] = Node(self, f'{node.name}~{n.name}', n.kind)
            elif len(n.outs) > 0 and len(n.ins) > 0:  # output is also read by impl. circuit, need to add a fork.
                node_map[n] = Node(self, f'{node.name}~{n.name}')
            elif len(n.ins) == 0 and len(n.outs) > 1:  # input is read by multiple nodes, need to add fork.
                node_map[n] = Node(self, f'{node.name}~{n.name}')
        for l in impl.lines:  # add all internal lines to main circuit
            if l.reader in node_map and l.driver in node_map:
                Line(self, (node_map[l.driver], l.driver_pin), (node_map[l.reader], l.reader_pin))
        for inn, ll in zip(impl_in_nodes, node_in_lines):  # connect inputs
            if ll is None: continue
            if len(inn.outs) == 1:
                l = inn.outs[0]
                ll.reader = node_map[l.reader]
                ll.reader_pin = l.reader_pin
            else:
                ll.reader = node_map[inn]  # connect to existing fork
                ll.reader_pin = 0
            ll.reader.ins[ll.reader_pin] = ll
        for l, ll in zip(impl_out_lines, node_out_lines):  # connect outputs
            if ll is None:
                if l.driver in node_map:
                    self.remove_dangling_nodes(node_map[l.driver])
                continue
            if len(l.reader.outs) > 0:  # output is also read by impl. circuit, connect to fork.
                ll.driver = node_map[l.reader]
                ll.driver_pin = len(l.reader.outs)
            else:
                ll.driver = node_map[l.driver]
                ll.driver_pin = l.driver_pin
            ll.driver.outs[ll.driver_pin] = ll

    def resolve_tlib_cells(self, tlib):
        """Substitute all technology library cells with kyupy native simulation primitives.

        See :py:attr:`substitute()` for more detail.
        """
        for n in list(self.nodes):
            if n.kind in tlib.cells:
                self.substitute(n, tlib.cells[n.kind][0])

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
        for node in self.io_nodes:
            if node.kind == '__fork__':
                n = c.forks[node.name]
            else:
                n = c.cells[node.name]
            c.io_nodes.append(n)
        return c

    def __getstate__(self):
        nodes = [(node.name, node.kind) for node in self.nodes]
        lines = [(line.driver.index, line.driver_pin, line.reader.index, line.reader_pin) for line in self.lines]
        io_nodes = [n.index for n in self.io_nodes]
        return {'name': self.name,
                'nodes': nodes,
                'lines': lines,
                'io_nodes': io_nodes }

    def __setstate__(self, state):
        self.name = state['name']
        self.nodes = IndexList()
        self.lines = IndexList()
        self.io_nodes = GrowingList()
        self.cells = {}
        self.forks = {}
        for s in state['nodes']:
            Node(self, *s)
        for driver, driver_pin, reader, reader_pin in state['lines']:
            Line(self, (self.nodes[driver], driver_pin), (self.nodes[reader], reader_pin))
        for n in state['io_nodes']:
            self.io_nodes.append(self.nodes[n])

    def __eq__(self, other):
        return self.nodes == other.nodes and self.lines == other.lines and self.io_nodes == other.io_nodes

    def __repr__(self):
        return f'{{name: "{self.name}", cells: {len(self.cells)}, forks: {len(self.forks)}, lines: {len(self.lines)}, io_nodes: {len(self.io_nodes)}}}'

    def topological_order(self):
        """Generator function to iterate over all nodes in topological order.

        Nodes without input lines and nodes whose :py:attr:`Node.kind` contains the
        substrings 'dff' or 'latch' are yielded first.
        """
        visit_count = np.zeros(len(self.nodes), dtype=np.uint32)
        queue = deque(n for n in self.nodes if len(n.ins) == 0 or 'dff' in n.kind.lower() or 'latch' in n.kind.lower())
        while len(queue) > 0:
            n = queue.popleft()
            for line in n.outs:
                if line is None: continue
                succ = line.reader
                visit_count[succ] += 1
                if visit_count[succ] == len(succ.ins) and 'dff' not in succ.kind.lower() and 'latch' not in succ.kind.lower():
                    queue.append(succ)
            yield n

    def topological_order_with_level(self):
        level = np.zeros(len(self.nodes), dtype=np.int32) - 1
        for n in self.topological_order():
            if len(n.ins) == 0 or 'dff' in n.kind.lower() or 'latch' in n.kind.lower():
                l = 0
            else:
                l = level[[l.driver.index for l in n.ins if l is not None]].max() + 1
            level[n] = l
            yield n, l

    def topological_line_order(self):
        """Generator function to iterate over all lines in topological order.
        """
        for n in self.topological_order():
            for line in n.outs:
                if line is not None:
                    yield line

    def reversed_topological_order(self):
        """Generator function to iterate over all nodes in reversed topological order.

        Nodes without output lines and nodes whose :py:attr:`Node.kind` contains the
        substrings 'dff' or 'latch' are yielded first.
        """
        visit_count = [0] * len(self.nodes)
        queue = deque(n for n in self.nodes if len(n.outs) == 0 or 'dff' in n.kind.lower() or 'latch' in n.kind.lower())
        while len(queue) > 0:
            n = queue.popleft()
            for line in n.ins:
                pred = line.driver
                visit_count[pred] += 1
                if visit_count[pred] == len(pred.outs) and 'dff' not in pred.kind.lower() and 'latch' not in pred.kind.lower():
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
            if len(stem.outs) == 1 and 'dff' not in stem.kind.lower(): continue
            region = []
            if 'dff' in stem.kind.lower():
                n = stem.ins[0]
                if len(n.driver.outs) == 1 and 'dff' not in n.driver.kind.lower():
                    queue = deque([n.driver])
                else:
                    queue = deque()
            else:
                queue = deque(n.driver for n in stem.ins
                              if len(n.driver.outs) == 1 and 'dff' not in n.driver.kind.lower())
            while len(queue) > 0:
                n = queue.popleft()
                preds = [pred.driver for pred in n.ins
                         if len(pred.driver.outs) == 1 and 'dff' not in pred.driver.kind.lower()]
                queue.extend(preds)
                region.append(n)
            yield stem, region

    def dot(self, format='svg'):
        from graphviz import Digraph
        dot = Digraph(format=format, graph_attr={'rankdir': 'LR', 'splines': 'true'})

        s_dict = dict((n, i) for i, n in enumerate(self.s_nodes))
        node_level = np.zeros(len(self.nodes), dtype=np.uint32)
        level_nodes = defaultdict(list)
        for n, lv in self.topological_order_with_level():
            level_nodes[lv].append(n)
            node_level[n] = lv

        for lv in level_nodes:
            with dot.subgraph() as s:
                s.attr(rank='same')
                for n in level_nodes[lv]:
                    ins = '|'.join([f'<i{i}>{i}' for i in range(len(n.ins))])
                    outs = '|'.join([f'<o{i}>{i}' for i in range(len(n.outs))])
                    io = f' [{s_dict[n]}]' if n in s_dict else ''
                    s.node(name=str(n.index), label = f'{{{{{ins}}}|{n.index}{io}\n{n.kind}\n{n.name}|{{{outs}}}}}', shape='record')

        for l in self.lines:
            driver, reader = f'{l.driver.index}:o{l.driver_pin}', f'{l.reader.index}:i{l.reader_pin}'
            if node_level[l.driver] >= node_level[l.reader]:
                dot.edge(driver, reader, style='dotted', label=str(l.index))
                pass
            else:
                dot.edge(driver, reader, label=str(l.index))

        return dot
