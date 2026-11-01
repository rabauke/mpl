Communicators
=============

A communicator consists of a group of processed and defines a communication context that partitions the communication space. A message sent in one context cannot be received in another context. Furthermore, where permitted, collective operations are independent of pending point-to-point operations.

MPL defines several kinds of communicators:

-  standard communicators,

-  communicators with a process topology (Cartesian communicators, graph communicators, distributed graph communicators) and

- inter-communicators.

An inter-communicator identifies two distinct groups of processes linked with a communication context.


Standard communicators
----------------------

.. doxygenclass:: mpl::communicator


Standard communicators for MPI interoperability
-----------------------------------------------

.. doxygenclass:: mpl::mpi_communicator


Cartesian communicators
-----------------------

.. doxygenclass:: mpl::cartesian_communicator


Auxiliary functions and classes for cartesian communicators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For constructing communicators with a cartesian process topology the following utility function can be used.

.. doxygenfunction:: mpl::dims_create

.. doxygenstruct:: mpl::shift_ranks


Graph communicators
-------------------

.. doxygenclass:: mpl::graph_communicator


Distributed graph communicators
-------------------------------

.. doxygenclass:: mpl::distributed_graph_communicator


Inter-communicators
-------------------

.. doxygenclass:: mpl::inter_communicator


Inter-communicators for MPI interoperability
-----------------------------------------------

.. doxygenclass:: mpl::mpi_inter_communicator
