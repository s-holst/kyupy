Simulators
==========

KyuPy's simulators are optimized for cells with at most 4 inputs and 1 output.

More complex cells must be mapped to simulation primitives first.


Logic Simulation - :mod:`kyupy.logic_sim`
-----------------------------------------

.. automodule:: kyupy.logic_sim

.. autoclass:: kyupy.logic_sim.LogicSim
   :members:


Timing Simulation - :mod:`kyupy.wave_sim`
-----------------------------------------

.. automodule:: kyupy.wave_sim
   :members: TMAX, TMAX_OVL, TMIN

.. autoclass:: kyupy.wave_sim.WaveSim
   :members:

.. autoclass:: kyupy.wave_sim.WaveSimCuda
   :members:
