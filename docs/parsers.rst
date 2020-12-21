Parsers
=======

KyuPy contains simple (and often incomplete) parsers for common file formats.
These parsers are tailored to the most common use-cases to keep the grammars and the code-base as simple as possible.

Each of the modules export a function ``parse()`` for parsing a string directly and a function
``load()`` for loading a file. Files with a '.gz' extension are uncompressed on-the-fly.


Verilog - :mod:`kyupy.verilog`
------------------------------

.. automodule:: kyupy.verilog
   :members: parse, load


Bench Format - :mod:`kyupy.bench`
---------------------------------

.. automodule:: kyupy.bench
   :members: parse, load


Standard Test Interface Language - :mod:`kyupy.stil`
----------------------------------------------------

.. automodule:: kyupy.stil
   :members: parse, load

.. autoclass:: kyupy.stil.StilFile
   :members:


Standard Delay Format - :mod:`kyupy.sdf`
----------------------------------------

.. automodule:: kyupy.sdf
   :members: parse, load

.. autoclass:: kyupy.sdf.DelayFile
   :members:
