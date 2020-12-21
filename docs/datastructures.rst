Data Structures
===============

KyuPy provides two types of core data structures, one for gate-level circuits, and a few others for representing and storing logic data and signal values.
The data structures are designed to work together nicely with numpy arrays.
For example, all the nodes and connections in the circuit graph have consecutive integer indices that can be used to access ndarrays with associated data.
Circuit graphs also define an ordering of inputs, outputs and other nodes to easily process test vector data and alike.

Circuit Graph - :mod:`kyupy.circuit`
------------------------------------

.. automodule:: kyupy.circuit

.. autoclass:: kyupy.circuit.Node
   :members:

.. autoclass:: kyupy.circuit.Line
   :members:

.. autoclass:: kyupy.circuit.Circuit
   :members:

Multi-Valued Logic - :mod:`kyupy.logic`
---------------------------------------

.. automodule:: kyupy.logic
   :members:


