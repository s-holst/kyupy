KyuPy - Processing VLSI Circuits With Ease
==========================================

KyuPy is a python package for high-performance processing and analysis of
non-hierarchical VLSI designs. Its purpose is to provide a rapid prototyping
platform to aid and accelerate research in the fields of VLSI test, diagnosis
and reliability. KyuPy is freely available under the MIT license.


Main Features
-------------

* Partial [lark](https://github.com/lark-parser/lark) parsers for common files used with synthesized designs:
  bench, gate-level verilog, standard delay format (SDF), standard test interface language (STIL)
* Bit-parallel gate-level 2-, 4-, and 8-valued logic simulation
* GPU-accelerated high-throughput gate-level timing simulation
* High-performance through the use of [numpy](https://numpy.org) and [numba](https://numba.pydata.org)


Getting Started
---------------

KyuPy requires Python 3.6 or newer.
Install the latest release by running:
```commandline
pip3 install --user kyupy
```
For best performance, ensure you have [numba](https://pypi.org/project/numba) installed:
```commandline
pip3 install --user numba
```
GPU/CUDA support may [require some additional setup](https://numba.pydata.org/numba-doc/latest/cuda/index.html).
If CUDA or numba is not available, KyuPy will automatically fall back to slow, pure python execution.

The Jupyter Notebook [UsageExamples.ipynb](https://github.com/s-holst/kyupy/blob/main/UsageExamples.ipynb) on GitHub
contains some useful examples to get familiar with the API.


Development
-----------

To contribute to KyuPy or simply explore the source code, clone the KyuPy [repository](https://github.com/s-holst/kyupy) on GitHub.
Within your local checkout, run:
```commandline
pip3 install --user -e .
```
to make the kyupy package available in your python environment.
The source code comes with tests that can be run with:
```
pytest
```

KyuPy depends on the following packages:
* [lark-parser](https://pypi.org/project/lark-parser)
* [numpy](https://pypi.org/project/numpy)
* [numba](https://pypi.org/project/numba) (optional, required only for GPU/CUDA support)
