Custom initialization of the MPI environment
============================================

MPL initializes the MPI environment automatically internally.  This example program shows how to write a custom MPI initializer when more control over the initialization of the MPI environment is needed.  This can be useful, when combining MPL with other MPI-based libraries.

.. literalinclude:: ../../../examples/custom_initialization.cc
   :language: c++
