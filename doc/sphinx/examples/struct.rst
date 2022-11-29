Using structs
=============

Demonstrates how to use ``class mpl::struct_builder`` to enable communication using structures and classes. Class members must be of fixed size (no dynamic memory allocation). All types of the class members must be suited for communication. These rules may be applied recursively.

.. literalinclude:: ../../../examples/struct.cc
   :language: c++
