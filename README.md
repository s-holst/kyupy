KyuPy - Processing VLSI Circuits With Ease
==========================================

KyuPy is a python package for high-performance processing and analysis of
non-hierarchical VLSI designs. Its purpose is to provide a rapid prototyping
platform to aid and accelerate research in the fields of VLSI test, diagnosis
and reliability. KyuPy is freely available under the MIT license.

Main Features
-------------

* Partial [lark](https://github.com/lark-parser/lark)-parsers for common files used with synthesized designs: bench, gate-level verilog, standard delay format (SDF), standard test interface language (STIL)
* Bit-parallel gate-level 2-, 4-, and 8-valued logic simulation
* GPU-accelerated high-throughput gate-level timing simulation
* High-performance through the use of [numpy](https://numpy.org) and [numba](https://numba.pydata.org)


Getting Started
---------------

KyuPy requires python 3.6+ and the following packages:
* [lark-parser](https://pypi.org/project/lark-parser)
* [numpy](https://pypi.org/project/numpy)
* [numba](https://pypi.org/project/numba) (required only for GPU/CUDA support)

GPU/CUDA support may [require some additional setup](https://numba.pydata.org/numba-doc/latest/cuda/index.html). If CUDA or numba is not available, the package will automatically fall back to pure python execution.

This repository contains tests that can be run with:
```
pytest
```

Usage examples to get familiar with the API can be found in the Jupyter Notebook [UsageExamples.ipynb](UsageExamples.ipynb).
