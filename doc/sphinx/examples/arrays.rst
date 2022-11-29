Arrays
======

Sends and receives arrays (C arrays and ``std::array``) of fixed size, which must be known at compile time. The types of the array elements must be suited for communication. These rules may be applied recursively.

.. literalinclude:: ../../../examples/arrays.cc
   :language: c++
