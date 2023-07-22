KyuPy - Pythonic Processing of VLSI Circuits
============================================

KyuPy is a Python package for processing and analysis of non-hierarchical gate-level VLSI designs.
It contains fundamental building blocks for research software in the fields of VLSI test, diagnosis and reliability:

* Efficient data structures for gate-level circuits and related design data.
* Partial `lark <https://github.com/lark-parser/lark>`_ parsers for common design files like
  bench, gate-level Verilog, standard delay format (SDF), standard test interface language (STIL), design exchange format (DEF).
* Bit-parallel gate-level 2-, 4-, and 8-valued logic simulation.
* GPU-accelerated high-throughput gate-level timing simulation.
* High-performance through the use of `numpy <https://numpy.org>`_ and `numba <https://numba.pydata.org>`_.


Getting Started
---------------

KyuPy is available in `PyPI <https://pypi.org/project/kyupy>`_.
It requires Python 3.8 or newer, `lark-parser <https://pypi.org/project/lark-parser>`_, and `numpy`_.
Although optional, `numba`_ should be installed for best performance.
GPU/CUDA support in numba may `require some additional setup <https://numba.readthedocs.io/en/stable/cuda/index.html>`_.
If numba is not available, KyuPy will automatically fall back to slow, pure Python execution.

The Jupyter Notebook `Introduction.ipynb <https://github.com/s-holst/kyupy/blob/main/examples/Introduction.ipynb>`_ contains some useful examples to get familiar with the API.


Development
-----------

To work with the latest pre-release source code, clone the `KyuPy GitHub repository <https://github.com/s-holst/kyupy>`_.
Run ``pip install -e .`` within your local checkout to make the package available in your Python environment.
The source code comes with tests that can be run with ``pytest``.
