Process creation
================

The following two programs illustrate dynamic creation of new processes and establishing a communication channel in the form of inter-communicator.

.. literalinclude:: ../../../examples/process_creation.cc
   :language: c++

.. literalinclude:: ../../../examples/process_creation_multiple.cc
   :language: c++


The corresponding source of the spawned process is given as shown below:

.. literalinclude:: ../../../examples/process_creation_client.cc
   :language: c++
